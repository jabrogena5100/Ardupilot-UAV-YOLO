#!/usr/bin/env python3
"""Runs ON the pod. Wraps one job so the GPU is always stopped afterwards.

    python3 pod_supervisor.py --run-name NAME --result-dir /workspace/runs/NAME_orch --hard-timeout-s 3600 \
        --artifact '/workspace/runs/NAME/weights/best.pt' -- bash my_job.sh args...

Lifecycle: (1) refuse if result dir exists; (2) start an independent watchdog process (own session) that stops the pod at
the hard deadline even if this process hangs; (3) run the job with a timeout; (4) write DONE.json (status success | failed |
timeout, exit code, wall time, artifact SHA-256s) to the volume; (5) notify; (6) stop the pod (never delete) and verify.
Stopping is retried; if it still fails the failure is logged and notified and the watchdog keeps retrying.

Environment (injected by RunPod / the launcher): RUNPOD_POD_ID, RUNPOD_API_KEY (a pod-scoped key), NOTIFY_URL (optional).
"""
import argparse
import glob
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rp_common as C  # noqa: E402


def stop_and_verify(api, pod_id, log, attempts=6, sleep=time.sleep):
    """Stop (not delete) the pod, then confirm via GET. Returns True when the pod reads as stopped."""
    for i in range(attempts):
        try:
            api.stop_pod(pod_id)
            sleep(1)
            if C.is_stopped(api.get_pod(pod_id)):
                log.emit("pod_stopped", attempt=i + 1)
                return True
            log.emit("stop_not_yet_confirmed", attempt=i + 1)
        except C.RunPodError as e:
            log.emit("stop_error", attempt=i + 1, error=str(e))
        sleep(min(2 ** (i + 1), 30))
    log.emit("STOP_FAILED", pod_id=pod_id)
    return False


def run_watchdog(api, pod_id, deadline_epoch, log_path, notify_url, poll=5, sleep=time.sleep, now=time.time):
    """Independent process: at the deadline stop the pod and keep retrying until it reads stopped."""
    log = C.EventLog(log_path)
    while now() < deadline_epoch:
        sleep(min(poll, max(0.0, deadline_epoch - now())))
    log.emit("watchdog_deadline_reached", pod_id=pod_id)
    ok = stop_and_verify(api, pod_id, log, attempts=12, sleep=sleep)
    C.notify(notify_url, {"title": "RunPod watchdog fired", "urgent": not ok, "pod_id": pod_id, "stopped": ok})
    return ok


def run_job(cmd, timeout_s, job_log_path):
    """Run cmd in its own process group; kill the group on timeout. Returns (status, exit_code)."""
    with open(job_log_path, "ab") as out:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            return "timeout", None
    return ("success" if rc == 0 else "failed"), rc


def collect_artifacts(patterns):
    out = {}
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            if os.path.isfile(path):
                out[path] = {"sha256": C.sha256_file(path), "bytes": os.path.getsize(path)}
    return out


def supervise(args, cmd, api, spawn_watchdog=None, sleep=time.sleep):
    if os.path.exists(args.result_dir):
        print(f"ABORT: {args.result_dir} exists (refusing to overwrite)", file=sys.stderr)
        return 4
    os.makedirs(args.result_dir)
    log = C.EventLog(os.path.join(args.result_dir, "events.jsonl"))
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    notify_url = os.environ.get("NOTIFY_URL", "")
    t0 = time.time()
    deadline = t0 + args.hard_timeout_s
    log.emit("start", run_name=args.run_name, pod_id=pod_id, hard_timeout_s=args.hard_timeout_s, cmd_argv0=cmd[0],
             git_commit=os.environ.get("GIT_COMMIT", "unknown"))
    wd = (spawn_watchdog or spawn_watchdog_process)(args, deadline) if pod_id else None
    status, rc = "failed", None
    try:
        # leave the watchdog a margin: the job gets killed before the deadline so DONE.json and the stop can happen first
        status, rc = run_job(cmd, max(1, args.hard_timeout_s - min(args.finish_margin_s, args.hard_timeout_s // 4)), os.path.join(args.result_dir, "job.log"))
    except Exception as e:  # supervisor must still stop the pod
        log.emit("job_exception", error=type(e).__name__)
    finally:
        wall = int(time.time() - t0)
        result = {"run_name": args.run_name, "status": status, "exit_code": rc, "wall_seconds": wall,
                  "pod_id": pod_id, "finished_utc": C.utc_now(), "hard_timeout_s": args.hard_timeout_s,
                  "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
                  "artifacts": collect_artifacts(args.artifact)}
        try:
            C.write_json_new(os.path.join(args.result_dir, "DONE.json"), result)
        except Exception as e:
            log.emit("done_write_failed", error=type(e).__name__)
        log.emit("job_finished", status=status, exit_code=rc, wall_seconds=wall)
        C.notify(notify_url, {"title": f"RunPod job {status}: {args.run_name}", "status": status, "exit_code": rc,
                              "wall_seconds": wall, "pod_id": pod_id, "artifacts": len(result["artifacts"])})
        stopped = stop_and_verify(api, pod_id, log, sleep=sleep) if pod_id else False
        if not stopped:
            C.notify(notify_url, {"title": "RunPod STOP FAILED", "urgent": True, "pod_id": pod_id, "run_name": args.run_name})
        if wd is not None:
            try:
                wd.terminate()
            except Exception:
                pass
    return 0 if status == "success" else 1


def spawn_watchdog_process(args, deadline):
    return subprocess.Popen([sys.executable, os.path.abspath(__file__), "--watchdog", "--deadline-epoch", str(deadline),
                             "--result-dir", args.result_dir], start_new_session=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name")
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--hard-timeout-s", type=int)
    ap.add_argument("--finish-margin-s", type=int, default=300)
    ap.add_argument("--artifact", action="append", default=[], help="glob of files to hash into DONE.json")
    ap.add_argument("--watchdog", action="store_true", help="internal: run as the independent watchdog")
    ap.add_argument("--deadline-epoch", type=float)
    argv = sys.argv[1:] if argv is None else argv
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    args = ap.parse_args(argv)
    try:
        api = C.RunPodAPI()
    except C.RunPodError as e:  # fail safe: without a key this pod cannot stop itself, so do not start the job
        print(f"ABORT: {e}; refusing to run a job that cannot self-stop", file=sys.stderr)
        return 5
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    if args.watchdog:
        ok = run_watchdog(api, pod_id, args.deadline_epoch, os.path.join(args.result_dir + "_watchdog.jsonl"),
                          os.environ.get("NOTIFY_URL", ""))
        return 0 if ok else 2
    if not (cmd and args.run_name and args.hard_timeout_s):
        ap.error("--run-name, --hard-timeout-s and a command after -- are required")
    if not C.RUN_NAME_RE.match(args.run_name):
        ap.error("bad run name")
    return supervise(args, cmd, api)


if __name__ == "__main__":
    sys.exit(main())
