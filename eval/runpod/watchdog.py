#!/usr/bin/env python3
"""Stateless off-pod watchdog. One pass: stop every RUNNING pod whose name-tagged hard deadline has passed.

A pod is created with name `<run>__dl<epoch>` (rp_common.tag_name), so this needs no state, session or local machine: it only
needs RUNPOD_API_KEY. It runs from .github/workflows/runpod-watchdog.yml (GitHub's servers, every 5 minutes), and can also be run
by hand or from cron. It only ever STOPS pods (no delete). Untagged running pods are reported, and stopped only with --stop-untagged.

Events are appended as JSON lines to --log (the workflow commits that file to the `runpod-logs` branch: append-only history).
Quiet passes (no running pods) write nothing. Exit 0 = nothing wrong; 2 = API error or a stop that could not be confirmed
(the workflow then fails, which GitHub surfaces in the Actions tab/app).
"""
import argparse
import sys
import time

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rp_common as C  # noqa: E402


def enforce(api, log, now=time.time, stop_untagged=False, grace_s=0, sleep=time.sleep, attempts=3):
    """Returns 0 or 2. `log` is a callable event(name, **fields)."""
    try:
        pods = api.list_pods()
    except C.RunPodError as e:
        log("api_error", where="list_pods", error=str(e))
        return 2
    rc = 0
    for p in pods:
        if not C.is_running(p):
            continue
        pid, name, t = p.get("id"), p.get("name"), now()
        dl = C.parse_deadline(name)
        if dl is None and not stop_untagged:
            log("untagged_running_pod", pod_id=pid, name=name, note="no deadline tag; not stopped (use --stop-untagged)")
            continue
        if dl is not None and t <= dl + grace_s:
            log("heartbeat_running_within_deadline", pod_id=pid, name=name, seconds_to_deadline=int(dl - t))
            continue
        log("deadline_exceeded" if dl is not None else "stopping_untagged", pod_id=pid, name=name,
            seconds_over=(int(t - dl) if dl is not None else None))
        stopped = False
        for i in range(attempts):
            try:
                api.stop_pod(pid)
                sleep(5)
                if C.is_stopped(api.get_pod(pid)):
                    stopped = True
                    break
                log("stop_not_yet_confirmed", pod_id=pid, attempt=i + 1)
            except C.RunPodError as e:
                log("stop_error", pod_id=pid, attempt=i + 1, error=str(e))
            sleep(5)
        log("pod_stopped" if stopped else "STOP_FAILED", pod_id=pid, name=name)
        if not stopped:
            rc = 2
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="JSONL file to append events to")
    ap.add_argument("--stop-untagged", action="store_true")
    ap.add_argument("--grace-s", type=int, default=0)
    a = ap.parse_args(argv)
    try:
        api = C.RunPodAPI()
    except C.RunPodError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 2
    ev = C.EventLog(a.log)

    def log(name, **f):
        rec = ev.emit(name, **f)
        print(rec)
    return enforce(api, log, stop_untagged=a.stop_untagged, grace_s=a.grace_s)


if __name__ == "__main__":
    sys.exit(main())
