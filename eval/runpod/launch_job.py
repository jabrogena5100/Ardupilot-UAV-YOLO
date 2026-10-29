#!/usr/bin/env python3
"""Cloud/laptop side of RunPod job orchestration. Subcommands:

  check-access  read-only: API reads, volume present, billing, and a NON-MUTATING stop probe on a nonexistent pod id
  notify-test   send one test notification through $NOTIFY_URL (no RunPod call)
  preflight     read-only: no pod running, budget check -> plan (exit 0 ok / 3 refused)
  launch        preflight, then (only with --confirm-paid RUN_NAME) create ONE pod. Dry run otherwise.
                --watch keeps this process alive as the off-pod watchdog until the pod reads stopped.
  lifecycle-test  launch preset: disposable job (no model, no dataset), 60 s hard timeout, run name lifecycle_test_*
  monitor       off-pod watchdog for an existing pod: stop it once past its deadline
  sweep         stop every RUNNING pod whose name starts with --name-prefix (for cron / scheduled safety nets)

The pod needs no files on the network volume: this launcher ships the supervisor (and the job files) inside the pod's
start command as a base64 tar.gz in $ORCH_BUNDLE_B64. No delete/terminate call exists in this tooling; pods are stopped.
"""
import argparse
import base64
import io
import json
import os
import shlex
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rp_common as C  # noqa: E402

DEFAULTS = {"gpu": "NVIDIA GeForce RTX 4090", "data_center": "US-IL-1", "volume_id": "sej40ez3ia", "mount": "/workspace",
            "image": "runpod/pytorch:2.8.0-py3.12-cuda12.8.1-cudnn-devel-ubuntu22.04", "container_disk_gb": 30}
CORE_FILES = ["rp_common.py", "pod_supervisor.py"]
MAX_BUNDLE_B64 = 120_000  # keep the env var / start command comfortably small


def make_bundle(extra_files=()):
    """tar.gz (base64) holding the supervisor + extra files, flat names, deterministic. extra_files are paths inside eval/runpod/."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tf:
        for name in sorted(set(CORE_FILES) | set(extra_files)):
            path = os.path.realpath(os.path.join(HERE, name))
            if os.path.dirname(path) != os.path.realpath(HERE):
                raise ValueError(f"bundle files must live in eval/runpod/: {name}")
            ti = tarfile.TarInfo(os.path.basename(name))
            with open(path, "rb") as fh:
                data = fh.read()
            ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
            tf.addfile(ti, io.BytesIO(data))
    b64 = base64.b64encode(buf.getvalue()).decode()
    if len(b64) > MAX_BUNDLE_B64:
        raise ValueError(f"bundle too large ({len(b64)} chars)")
    return b64


def build_spec(a, notify_url, git_commit, bundle_b64):
    result_dir = f"{a.mount}/runs/{a.run_name}_orch"
    sup = ["python3", "pod_supervisor.py", "--run-name", a.run_name, "--result-dir", result_dir,
           "--hard-timeout-s", str(a.hard_timeout_s)]
    for art in a.artifact:
        sup += ["--artifact", art]
    sup += ["--"] + shlex.split(a.job)
    boot = ('set -e; D=$(mktemp -d); printf %s "$ORCH_BUNDLE_B64" | base64 -d | tar xz -C "$D"; cd "$D"; exec '
            + shlex.join(sup))
    return {"name": a.run_name[:60], "imageName": a.image, "cloudType": "SECURE", "gpuCount": 1,
            "gpuTypeIds": [a.gpu], "dataCenterIds": [a.data_center], "networkVolumeId": a.volume_id,
            "volumeMountPath": a.mount, "containerDiskInGb": a.container_disk_gb,
            "env": {"NOTIFY_URL": notify_url, "NOTIFY_KIND": os.environ.get("NOTIFY_KIND", "ntfy"),
                    "GIT_COMMIT": git_commit, "ORCH_BUNDLE_B64": bundle_b64},
            "dockerStartCmd": ["bash", "-lc", boot]}


def redact(spec):
    s = json.loads(json.dumps(spec))
    s["env"] = {k: ("<set>" if v else "<unset>") for k, v in s["env"].items()}
    return s


def preflight(api, a):
    pods = api.list_pods()
    running = [p for p in pods if C.is_running(p)]
    ok, reasons, worst = C.check_budget(price_per_hr=a.max_hourly_price, hard_timeout_s=a.hard_timeout_s,
                                        boot_allowance_s=a.boot_allowance_s, billed_usd=api.billed_total(),
                                        ledger_floor_usd=a.ledger_floor_usd, other_running=len(running))
    if not C.RUN_NAME_RE.match(a.run_name):
        ok, reasons = False, reasons + ["run name must match " + C.RUN_NAME_RE.pattern]
    if any(p.get("name") == a.run_name for p in pods):
        ok, reasons = False, reasons + ["a pod with this name already exists"]
    return {"ok": ok, "reasons": reasons, "worst_case_usd": round(worst, 2), "running_pods": len(running),
            "stopped_pods": len(pods) - len(running)}


def launch(api, a, notify_url, git_commit, sleep=time.sleep, now=time.time):
    bundle = make_bundle(getattr(a, "bundle_files", []))
    plan = preflight(api, a)
    spec = build_spec(a, notify_url, git_commit, bundle)
    out = {"preflight": plan, "spec": redact(spec), "bundle_chars": len(bundle), "created": False}
    if not plan["ok"]:
        return 3, out
    if a.confirm_paid != a.run_name:
        out["note"] = "dry run: pass --confirm-paid RUN_NAME (after owner approval) to create the pod"
        return 0, out
    t_create = now()
    pod = api.create_pod(spec)
    out.update(created=True, pod_id=pod.get("id"),
               deadline_epoch=t_create + a.boot_allowance_s + a.hard_timeout_s)  # off-pod watchdog stops the pod here
    price = pod.get("costPerHr")
    if pod.get("id") and price is not None and price > a.max_hourly_price:  # price is only known after creation
        api.stop_pod(pod["id"])
        out.update(stopped_for_price=True, cost_per_hr=price)
        return 3, out
    out["cost_per_hr"] = price
    return 0, out


def monitor(api, pod_id, deadline_epoch, notify_url, grace_s=60, poll_s=15, give_up_s=1800, sleep=time.sleep, now=time.time):
    """Off-pod watchdog. Stops the pod grace_s past the deadline and keeps retrying; returns 'stopped', 'gone' or 'UNCONFIRMED'."""
    while True:
        try:
            if C.is_stopped(api.get_pod(pod_id)):
                return "stopped"
        except C.RunPodError as e:
            if e.code == 404:
                return "gone"
        t = now()
        if t > deadline_epoch + grace_s:
            try:
                api.stop_pod(pod_id)
                C.notify(notify_url, {"title": "RunPod external watchdog stopped pod", "pod_id": pod_id})
            except C.RunPodError:
                pass
            if t > deadline_epoch + grace_s + give_up_s:
                C.notify(notify_url, {"title": "RunPod STOP UNCONFIRMED", "urgent": True, "pod_id": pod_id})
                return "UNCONFIRMED"
        sleep(poll_s)


def sweep(api, prefix, notify_url):
    """Stop every running pod whose name starts with prefix. Returns list of stopped ids."""
    if not prefix:
        raise ValueError("--name-prefix is required")
    stopped = []
    for p in api.list_pods():
        if C.is_running(p) and str(p.get("name", "")).startswith(prefix):
            api.stop_pod(p["id"])
            stopped.append(p["id"])
    if stopped:
        C.notify(notify_url, {"title": "RunPod sweep stopped pods", "urgent": True, "pod_ids": ",".join(stopped)})
    return stopped


def check_access(api, volume_id=DEFAULTS["volume_id"]):
    """Read-only access report. The stop probe targets a nonexistent id, so it can never stop a real pod."""
    rep = {}
    for name, fn in (("list_pods", api.list_pods), ("billing", api.billed_total)):
        try:
            fn()
            rep[name] = "ok"
        except C.RunPodError as e:
            rep[name] = f"FAIL {e.code}"
    try:
        rep["volume_present"] = any(v.get("id") == volume_id for v in api.list_network_volumes())
    except C.RunPodError as e:
        rep["volume_present"] = f"FAIL {e.code}"
    try:
        api.stop_pod("nonexistent000probe")
        rep["stop_probe"] = "UNEXPECTED success"
    except C.RunPodError as e:
        rep["stop_probe"] = ("NO WRITE SCOPE" if e.code in (401, 403) else
                             "authorized (server reports pod does not exist)" if "not exist" in e.detail else f"inconclusive {e.code}")
    return rep


def add_job_args(p, job_required=True):
    p.add_argument("--run-name", required=job_required)
    p.add_argument("--job", required=job_required, help="command run on the pod from the bundle dir")
    p.add_argument("--hard-timeout-s", type=int, required=job_required)
    p.add_argument("--max-hourly-price", type=float, required=True, help="USD/h you accept; bounds the budget check")
    p.add_argument("--ledger-floor-usd", type=float, default=0.0, help="conservative spend from docs/project-status.md")
    p.add_argument("--boot-allowance-s", type=int, default=600, help="time for pod start/image pull before the supervisor runs")
    p.add_argument("--artifact", action="append", default=[])
    p.add_argument("--bundle-file", action="append", default=[], dest="bundle_files", help="extra file from eval/runpod/")
    for k, v in (("gpu", DEFAULTS["gpu"]), ("data-center", DEFAULTS["data_center"]), ("volume-id", DEFAULTS["volume_id"]),
                 ("mount", DEFAULTS["mount"]), ("image", DEFAULTS["image"])):
        p.add_argument("--" + k, default=v)
    p.add_argument("--container-disk-gb", type=int, default=DEFAULTS["container_disk_gb"])
    p.add_argument("--confirm-paid", default="", help="must equal --run-name to create a pod")
    p.add_argument("--watch", action="store_true", help="after creating, stay as the off-pod watchdog until the pod stops")


def lifecycle_args(ns):
    ns.run_name = ns.run_name or "lifecycle_test_" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    if not ns.run_name.startswith("lifecycle_test_"):
        raise SystemExit("lifecycle test run names must start with lifecycle_test_")
    ns.job = f"python3 lifecycle_test_job.py --out {ns.mount}/runs/{ns.run_name}_job"
    ns.hard_timeout_s = 60
    ns.artifact = [f"{ns.mount}/runs/{ns.run_name}_job/*"]
    ns.bundle_files = ["lifecycle_test_job.py"]
    return ns


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-access")
    sub.add_parser("notify-test")
    for name in ("preflight", "launch"):
        add_job_args(sub.add_parser(name))
    lt = sub.add_parser("lifecycle-test")
    add_job_args(lt, job_required=False)
    lt.set_defaults(run_name="", job="", hard_timeout_s=60)
    m = sub.add_parser("monitor")
    m.add_argument("--pod-id", required=True)
    m.add_argument("--deadline-epoch", type=float, required=True)
    sw = sub.add_parser("sweep")
    sw.add_argument("--name-prefix", required=True)
    a = ap.parse_args(argv)
    notify_url = os.environ.get("NOTIFY_URL", "")
    if a.cmd == "notify-test":
        ok = C.notify(notify_url, {"title": "RunPod notify test", "message": "If you can read this on your phone, the channel works."})
        print(json.dumps({"notify_url_set": bool(notify_url), "delivered_http_2xx": ok}))
        return 0 if ok else 2
    api = C.RunPodAPI()
    if a.cmd == "check-access":
        rep = check_access(api)
        print(json.dumps(rep, indent=2))
        return 0 if rep.get("stop_probe", "").startswith("authorized") else 3
    if a.cmd == "monitor":
        print(monitor(api, a.pod_id, a.deadline_epoch, notify_url))
        return 0
    if a.cmd == "sweep":
        print(json.dumps({"stopped": sweep(api, a.name_prefix, notify_url)}))
        return 0
    if a.cmd == "lifecycle-test":
        a = lifecycle_args(a)
    commit = os.environ.get("GIT_COMMIT", "unknown")
    if a.cmd == "preflight":
        out = {"preflight": preflight(api, a)}
        print(json.dumps(out, indent=2))
        return 0 if out["preflight"]["ok"] else 3
    rc, out = launch(api, a, notify_url, commit)
    print(json.dumps(out, indent=2))
    if rc == 0 and out.get("created") and a.watch:
        res = monitor(api, out["pod_id"], out["deadline_epoch"], notify_url)
        print(json.dumps({"watchdog_final": res}))
        return 0 if res in ("stopped", "gone") else 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
