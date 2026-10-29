#!/usr/bin/env python3
"""Cloud/laptop side of RunPod job orchestration. Subcommands:

  check-access    read-only: API reads, volume present, billing, and a NON-MUTATING stop probe on a nonexistent pod id
  preflight       read-only: no pod running, budget check -> plan (exit 0 ok / 3 refused)
  launch          preflight, then (only with --confirm-paid RUN_NAME and --watchdog-armed) create ONE pod. Dry run otherwise.
  transfer-test   launch preset: serve a 256 KiB disposable artifact through the RunPod HTTPS proxy (tests the return path), 600 s hard timeout
  lifecycle-test  launch preset: disposable job (no model, no dataset), 60 s hard timeout, run name lifecycle_test_*
  enforce         one pass of the off-pod watchdog (same as watchdog.py)
  status          what is running, deadlines, spend, and the append-only watchdog log (works from any new session)

The pod needs no files on the network volume: this launcher ships the supervisor (and the job files) inside the pod's
start command as a base64 tar.gz split over $ORCH_BUNDLE_B64_<i> (repo files under repo/). The pod's name carries its hard deadline (`<run>__dl<epoch>`), which the
independent GitHub Actions watchdog (.github/workflows/runpod-watchdog.yml) enforces. No delete/terminate call exists here.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import shlex
import sys
import subprocess
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rp_common as C  # noqa: E402

DEFAULTS = {"gpu": "NVIDIA GeForce RTX 4090", "data_center": "US-IL-1", "volume_id": "sej40ez3ia", "mount": "/workspace",
            "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404", "container_disk_gb": 30}
CORE_FILES = ["rp_common.py", "pod_supervisor.py"]
CHUNK_B64 = 50_000   # one env var per chunk (execve limit is 128 KiB per string)
MAX_CHUNKS = 8


def make_bundle(extra_files=(), repo_paths=()):
    """Deterministic tar.gz of the supervisor (+ extra eval/runpod files at the top level, + repo files under repo/), base64, split
    into chunks of at most CHUNK_B64 characters (one environment variable each). Returns (chunks, sha256 of the tarball).
    repo_paths are repo-relative files or directories (only *.py/*.sh/*.yaml/*.json/*.md files; no caches, weights or data)."""
    root = os.path.dirname(os.path.dirname(HERE))
    members = {}
    for name in sorted(set(CORE_FILES) | set(extra_files)):
        path = os.path.realpath(os.path.join(HERE, name))
        if os.path.dirname(path) != os.path.realpath(HERE):
            raise ValueError(f"bundle files must live in eval/runpod/: {name}")
        members[os.path.basename(name)] = path
    for rel in sorted(repo_paths):
        path = os.path.realpath(os.path.join(root, rel))
        if not path.startswith(os.path.realpath(root) + os.sep):
            raise ValueError(f"repo path outside the repository: {rel}")
        walk = [path] if os.path.isfile(path) else [os.path.join(d, f) for d, _, fs in os.walk(path) for f in fs]
        for fp in sorted(walk):
            if "__pycache__" in fp or os.path.splitext(fp)[1] not in (".py", ".sh", ".yaml", ".json", ".md"):
                continue
            members["repo/" + os.path.relpath(fp, root)] = fp
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tf:
        for arc in sorted(members):
            with open(members[arc], "rb") as fh:
                data = fh.read()
            ti = tarfile.TarInfo(arc)
            ti.size, ti.mtime, ti.mode = len(data), 0, 0o755 if arc.endswith(".sh") else 0o644
            tf.addfile(ti, io.BytesIO(data))
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode()
    chunks = [b64[i:i + CHUNK_B64] for i in range(0, len(b64), CHUNK_B64)]
    if len(chunks) > MAX_CHUNKS:
        raise ValueError(f"bundle too large ({len(b64)} chars, {len(chunks)} chunks)")
    return chunks, hashlib.sha256(raw).hexdigest()


def build_spec(a, git_commit, bundle, deadline_epoch):
    chunks, bundle_sha = bundle
    result_dir = f"{a.mount}/runs/{a.run_name}_orch"
    sup = ["python3", "pod_supervisor.py", "--run-name", a.run_name, "--result-dir", result_dir,
           "--hard-timeout-s", str(a.hard_timeout_s)]
    for art in a.artifact:
        sup += ["--artifact", art]
    sup += ["--"] + shlex.split(a.job)
    names = [f"ORCH_BUNDLE_B64_{i}" for i in range(len(chunks))]
    boot = ('set -e; D=$(mktemp -d); { ' + " ".join(f'printf %s "${n}";' for n in names)
            + ' } | base64 -d | tar xz -C "$D"; cd "$D"; exec ' + shlex.join(sup))
    env = {"GIT_COMMIT": git_commit, "BUNDLE_SHA256": bundle_sha, **dict(zip(names, chunks)), **dict(getattr(a, "extra_env", {}) or {})}
    if getattr(a, "results_token", None):
        env["RESULTS_TOKEN"] = a.results_token
    spec = {"name": C.tag_name(a.run_name, deadline_epoch), "imageName": a.image, "cloudType": "SECURE", "gpuCount": 1,
            "gpuTypeIds": [a.gpu], "dataCenterIds": [a.data_center], "networkVolumeId": a.volume_id,
            "volumeMountPath": a.mount, "containerDiskInGb": a.container_disk_gb, "env": env,
            "dockerStartCmd": ["bash", "-lc", boot]}
    if getattr(a, "ports", None):
        spec["ports"] = list(a.ports)
    return spec


def redact(spec):
    s = json.loads(json.dumps(spec))
    s["env"] = {k: ("<set>" if v else "<unset>") for k, v in s["env"].items()}
    return s


def preflight(api, a):
    pods = api.list_pods()
    running = [p for p in pods if C.is_running(p)]
    ok, reasons, worst = C.check_budget(price_per_hr=a.max_hourly_price, hard_timeout_s=a.hard_timeout_s,
                                        boot_allowance_s=a.boot_allowance_s,
                                        watchdog_latency_s=a.watchdog_latency_s, billed_usd=api.billed_total(),
                                        ledger_floor_usd=a.ledger_floor_usd, other_running=len(running))
    if not C.RUN_NAME_RE.match(a.run_name):
        ok, reasons = False, reasons + ["run name must match " + C.RUN_NAME_RE.pattern]
    if any(str(p.get("name", "")).startswith(a.run_name + C.DEADLINE_TAG) for p in pods):
        ok, reasons = False, reasons + ["a pod with this name already exists"]
    return {"ok": ok, "reasons": reasons, "worst_case_usd": round(worst, 2), "running_pods": len(running),
            "stopped_pods": len(pods) - len(running)}


def launch(api, a, git_commit, now=time.time):
    bundle = make_bundle(getattr(a, "bundle_files", []), getattr(a, "repo_paths", []))
    plan = preflight(api, a)
    t_create = now()
    deadline = t_create + a.boot_allowance_s + a.hard_timeout_s  # off-pod watchdog stops the pod after this
    spec = build_spec(a, git_commit, bundle, deadline)
    out = {"preflight": plan, "spec": redact(spec), "bundle_chunks": len(bundle[0]), "bundle_sha256": bundle[1], "created": False,
           "deadline_epoch": int(deadline), "deadline_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline))}
    if not plan["ok"]:
        return 3, out
    if a.confirm_paid != a.run_name:
        out["note"] = "dry run: pass --confirm-paid RUN_NAME (after owner approval) to create the pod"
        return 0, out
    if not getattr(a, "watchdog_armed", False):
        out["note"] = ("refused: confirm the GitHub watchdog is live (secret RUNPOD_API_KEY set, workflow on the default branch, "
                       "a successful run seen) and pass --watchdog-armed")
        return 3, out
    pod = api.create_pod(spec)
    out.update(created=True, pod_id=pod.get("id"))
    price = pod.get("costPerHr")
    if pod.get("id") and price is not None and price > a.max_hourly_price:  # price is only known after creation
        api.stop_pod(pod["id"])
        out.update(stopped_for_price=True, cost_per_hr=price)
        return 3, out
    out["cost_per_hr"] = price
    if getattr(a, "ports", None) and pod.get("id"):
        out["proxy_urls"] = {p.split("/")[0]: f"https://{pod['id']}-{p.split('/')[0]}.proxy.runpod.net" for p in a.ports if p.endswith("/http")}
    return 0, out


def status(api, now=time.time, log_lines=20):
    """Everything a new session needs: pods with deadlines, spend, and the tail of the watchdog log branch (if fetchable)."""
    t = now()
    pods = []
    for p in api.list_pods():
        dl = C.parse_deadline(p.get("name"))
        pods.append({"id": p.get("id"), "name": p.get("name"), "state": p.get("desiredStatus"),
                     "deadline_utc": None if dl is None else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(dl)),
                     "overdue_s": None if dl is None else max(0, int(t - dl)) if C.is_running(p) else 0})
    out = {"pods": pods, "running": sum(1 for p in pods if p["state"] == "RUNNING"), "billed_total_usd": round(api.billed_total(), 2)}
    try:
        subprocess.run(["git", "fetch", "-q", "origin", "runpod-logs"], cwd=HERE, capture_output=True, timeout=30)
        r = subprocess.run(["git", "show", "FETCH_HEAD:enforce_events.jsonl"], cwd=HERE, capture_output=True, text=True, timeout=30)
        out["watchdog_log_tail"] = r.stdout.strip().splitlines()[-log_lines:] if r.returncode == 0 else "no runpod-logs branch yet"
    except Exception as e:
        out["watchdog_log_tail"] = f"unavailable ({type(e).__name__})"
    out["note"] = "DONE.json / events.jsonl for each run live on the volume under /workspace/runs/<run>_orch (readable only from a pod or S3 keys)"
    return out


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
    p.add_argument("--watchdog-latency-s", type=int, default=600, help="allowed delay of the 5-minute GitHub watchdog (cron can lag)")
    p.add_argument("--watchdog-armed", action="store_true", help="owner/agent attestation that the GitHub watchdog is live")
    p.add_argument("--artifact", action="append", default=[])
    p.add_argument("--bundle-file", action="append", default=[], dest="bundle_files", help="extra file from eval/runpod/")
    p.add_argument("--repo-path", action="append", default=[], dest="repo_paths", help="repo-relative file/dir shipped under repo/")
    for k, v in (("gpu", DEFAULTS["gpu"]), ("data-center", DEFAULTS["data_center"]), ("volume-id", DEFAULTS["volume_id"]),
                 ("mount", DEFAULTS["mount"]), ("image", DEFAULTS["image"])):
        p.add_argument("--" + k, default=v)
    p.add_argument("--container-disk-gb", type=int, default=DEFAULTS["container_disk_gb"])
    p.add_argument("--confirm-paid", default="", help="must equal --run-name to create a pod")
    p.add_argument("--port", action="append", default=[], dest="ports", help="expose a pod port, e.g. 8000/http (RunPod HTTPS proxy)")
    p.add_argument("--results-token-file", default="", help="file holding the results-endpoint token (created 0600 if missing; must be outside the repo)")


def lifecycle_args(ns):
    ns.run_name = ns.run_name or "lifecycle_test_" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    if not ns.run_name.startswith("lifecycle_test_"):
        raise SystemExit("lifecycle test run names must start with lifecycle_test_")
    ns.job = f"python3 lifecycle_test_job.py --out {ns.mount}/runs/{ns.run_name}_job"
    ns.hard_timeout_s = 60
    ns.artifact = [f"{ns.mount}/runs/{ns.run_name}_job/*"]
    ns.bundle_files = ["lifecycle_test_job.py"]
    return ns


def transfer_args(ns):
    ns.run_name = ns.run_name or "transfer_test_" + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    if not ns.run_name.startswith("transfer_test_"):
        raise SystemExit("transfer test run names must start with transfer_test_")
    ns.job = f"python3 transfer_test_job.py --out {ns.mount}/runs/{ns.run_name}_job --serve-seconds 420"
    ns.hard_timeout_s = 600
    ns.artifact = []
    ns.bundle_files = ["transfer_test_job.py", "results_server.py"]
    ns.ports = ns.ports or ["8000/http"]
    return ns


def load_token(path):
    """Read (or create with mode 0600) the results token. Refuses a path inside the repository."""
    import secrets
    root = os.path.realpath(os.path.dirname(os.path.dirname(HERE)))
    path = os.path.realpath(path)
    if path == root or path.startswith(root + os.sep):
        raise SystemExit("token file must live outside the repository")
    if not os.path.exists(path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(32) + "\n")
    return open(path).read().strip()


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-access")
    sub.add_parser("status")
    sub.add_parser("enforce")
    for name in ("preflight", "launch"):
        add_job_args(sub.add_parser(name))
    tt = sub.add_parser("transfer-test")
    add_job_args(tt, job_required=False)
    tt.set_defaults(run_name="", job="", hard_timeout_s=600)
    lt = sub.add_parser("lifecycle-test")
    add_job_args(lt, job_required=False)
    lt.set_defaults(run_name="", job="", hard_timeout_s=60)
    a = ap.parse_args(argv)
    api = C.RunPodAPI()
    if a.cmd == "check-access":
        rep = check_access(api)
        print(json.dumps(rep, indent=2))
        return 0 if rep.get("stop_probe", "").startswith("authorized") else 3
    if a.cmd == "status":
        print(json.dumps(status(api), indent=2))
        return 0
    if a.cmd == "enforce":
        import watchdog
        return watchdog.enforce(api, lambda n, **f: print(C.utc_now(), n, f))
    if a.cmd == "lifecycle-test":
        a = lifecycle_args(a)
    if a.cmd == "transfer-test":
        a = transfer_args(a)
    a.results_token = load_token(a.results_token_file) if getattr(a, "results_token_file", "") else None
    commit = os.environ.get("GIT_COMMIT", "unknown")
    if a.cmd == "preflight":
        out = {"preflight": preflight(api, a)}
        print(json.dumps(out, indent=2))
        return 0 if out["preflight"]["ok"] else 3
    if a.cmd == "transfer-test":
        a.cmd = "launch"
    rc, out = launch(api, a, commit)
    print(json.dumps(out, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
