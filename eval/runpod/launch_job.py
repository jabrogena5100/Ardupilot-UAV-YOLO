#!/usr/bin/env python3
"""Cloud/laptop side of RunPod job orchestration. Subcommands:

  preflight  read-only: API reachable, no pod running, budget check -> prints the plan (exit 0 ok / 3 refused)
  launch     preflight, then (only with --confirm-paid RUN_NAME) create ONE pod whose start command is pod_supervisor.py
             Without the flag it prints the redacted pod spec and creates nothing (dry run is the default).
  monitor    external watchdog: poll a pod; stop it (never delete) once past its deadline; exit when it reads stopped

No delete/terminate call exists in this tooling. Stopped pods keep the network volume, checkpoints and logs; a stopped pod's
container disk still accrues a small storage charge, so remove idle stopped pods by hand once artifacts are verified.
"""
import argparse
import json
import os
import shlex
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rp_common as C  # noqa: E402

DEFAULTS = {"gpu": "NVIDIA GeForce RTX 4090", "data_center": "US-IL-1", "volume_id": "sej40ez3ia", "mount": "/workspace",
            "image": "runpod/pytorch:2.8.0-py3.12-cuda12.8.1-cudnn-devel-ubuntu22.04", "container_disk_gb": 30}


def build_spec(a, notify_url, git_commit):
    result_dir = f"{a.mount}/runs/{a.run_name}_orch"
    sup = ["python3", "eval/runpod/pod_supervisor.py", "--run-name", a.run_name, "--result-dir", result_dir,
           "--hard-timeout-s", str(a.hard_timeout_s)]
    for art in a.artifact:
        sup += ["--artifact", art]
    sup += ["--"] + shlex.split(a.job)
    return {"name": f"{a.run_name}"[:60], "imageName": a.image, "cloudType": "SECURE", "gpuCount": 1,
            "gpuTypeIds": [a.gpu], "dataCenterIds": [a.data_center], "networkVolumeId": a.volume_id,
            "volumeMountPath": a.mount, "containerDiskInGb": a.container_disk_gb,
            "env": {"NOTIFY_URL": notify_url, "GIT_COMMIT": git_commit},
            "dockerStartCmd": ["bash", "-lc", f"cd {shlex.quote(a.repo_dir)} && " + shlex.join(sup)]}


def redact(spec):
    s = json.loads(json.dumps(spec))
    s["env"] = {k: ("<set>" if v else "<unset>") for k, v in s["env"].items()}
    return s


def preflight(api, a):
    pods = api.list_pods()
    running = [p for p in pods if C.is_running(p)]
    ok, reasons, worst = C.check_budget(price_per_hr=a.max_hourly_price, hard_timeout_s=a.hard_timeout_s,
                                        billed_usd=api.billed_total(), ledger_floor_usd=a.ledger_floor_usd,
                                        other_running=len(running))
    if not C.RUN_NAME_RE.match(a.run_name):
        ok, reasons = False, reasons + ["run name must match " + C.RUN_NAME_RE.pattern]
    if any(p.get("name") == a.run_name for p in pods):
        ok, reasons = False, reasons + ["a pod with this name already exists"]
    return {"ok": ok, "reasons": reasons, "worst_case_usd": round(worst, 2), "running_pods": len(running),
            "stopped_pods": len(pods) - len(running)}


def launch(api, a, notify_url, git_commit, sleep=time.sleep):
    plan = preflight(api, a)
    spec = build_spec(a, notify_url, git_commit)
    out = {"preflight": plan, "spec": redact(spec), "created": False}
    if not plan["ok"]:
        return 3, out
    if a.confirm_paid != a.run_name:
        out["note"] = "dry run: pass --confirm-paid RUN_NAME (after owner approval) to create the pod"
        return 0, out
    pod = api.create_pod(spec)
    out.update(created=True, pod_id=pod.get("id"))
    price = pod.get("costPerHr")
    if pod.get("id") and price is not None and price > a.max_hourly_price:  # price is only known after creation
        api.stop_pod(pod["id"])
        out.update(stopped_for_price=True, cost_per_hr=price)
        return 3, out
    out["cost_per_hr"] = price
    return 0, out


def monitor(api, pod_id, deadline_epoch, notify_url, grace_s=120, poll_s=30, sleep=time.sleep, now=time.time):
    """Stop the pod if it is still running grace_s past the deadline. Returns final status string."""
    while True:
        try:
            pod = api.get_pod(pod_id)
        except C.RunPodError:
            sleep(poll_s)
            continue
        if C.is_stopped(pod):
            C.notify(notify_url, {"title": "RunPod pod stopped", "pod_id": pod_id})
            return "stopped"
        if now() > deadline_epoch + grace_s:
            try:
                api.stop_pod(pod_id)
            except C.RunPodError:
                pass
            C.notify(notify_url, {"title": "RunPod external watchdog stopped pod", "pod_id": pod_id})
        sleep(poll_s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "launch"):
        p = sub.add_parser(name)
        p.add_argument("--run-name", required=True)
        p.add_argument("--job", required=True, help="command to run on the pod, e.g. 'bash eval/runpod/my_job.sh'")
        p.add_argument("--hard-timeout-s", type=int, required=True)
        p.add_argument("--max-hourly-price", type=float, required=True, help="USD/h you accept; used for the budget bound")
        p.add_argument("--ledger-floor-usd", type=float, default=0.0, help="conservative spend from docs/project-status.md")
        p.add_argument("--artifact", action="append", default=[])
        p.add_argument("--repo-dir", default="/workspace/repo_orch")
        p.add_argument("--gpu", default=DEFAULTS["gpu"])
        p.add_argument("--data-center", default=DEFAULTS["data_center"])
        p.add_argument("--volume-id", default=DEFAULTS["volume_id"])
        p.add_argument("--mount", default=DEFAULTS["mount"])
        p.add_argument("--image", default=DEFAULTS["image"])
        p.add_argument("--container-disk-gb", type=int, default=DEFAULTS["container_disk_gb"])
        p.add_argument("--confirm-paid", default="", help="must equal --run-name to create a pod")
    m = sub.add_parser("monitor")
    m.add_argument("--pod-id", required=True)
    m.add_argument("--deadline-epoch", type=float, required=True)
    a = ap.parse_args(argv)
    api = C.RunPodAPI()
    notify_url = os.environ.get("NOTIFY_URL", "")
    if a.cmd == "monitor":
        print(monitor(api, a.pod_id, a.deadline_epoch, notify_url))
        return 0
    rc, out = (0, {"preflight": preflight(api, a)}) if a.cmd == "preflight" else launch(
        api, a, notify_url, os.environ.get("GIT_COMMIT", "unknown"))
    if a.cmd == "preflight" and not out["preflight"]["ok"]:
        rc = 3
    print(json.dumps(out, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
