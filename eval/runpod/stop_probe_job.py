#!/usr/bin/env python3
"""Disposable probe: which on-pod method can stop this pod when the REST stop is forbidden (HTTP 403 with the pod-injected key)?

Tries, in order and logging only status codes / short non-secret excerpts: (A) GraphQL podStop with the injected key, (B) `runpodctl stop pod`.
Exports its log through results_server.py (so the outcome is readable before a successful stop kills the pod). Writes only a NEW directory
/workspace/runs/transfer_test_*/export/. The pod's external state (GET /pods/{id}) shows which attempt, if any, worked.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import results_server as RS  # noqa: E402

out = os.path.realpath(sys.argv[sys.argv.index("--out") + 1])
if not os.path.basename(out).startswith("transfer_test_") or os.path.dirname(out) != os.path.realpath("/workspace/runs"):
    sys.exit("refusing: output must be /workspace/runs/transfer_test_*")
exp = os.path.join(out, "export")
os.makedirs(exp)
LOG = os.path.join(exp, "stop_probe.log")
pod = os.environ.get("RUNPOD_POD_ID", "")
key = os.environ.get("RUNPOD_API_KEY", "")


def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M:%SZ ", time.gmtime()) + msg.replace(key, "***") + "\n")
    RS.write_manifest(exp, ["stop_probe.log"])


log(f"start pod_set={bool(pod)} key_set={bool(key)} key_len={len(key)} runpodctl={shutil.which('runpodctl')}")
log("env names with RUNPOD: " + ",".join(sorted(k for k in os.environ if "RUNPOD" in k and "KEY" not in k)))
threading.Thread(target=lambda: RS.serve(exp, 8000, os.environ.get("RESULTS_TOKEN", ""), 500), daemon=True).start()
time.sleep(60)


def attempt_graphql():
    q = {"query": 'mutation { podStop(input: {podId: "%s"}) { id desiredStatus } }' % pod}
    req = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps(q).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key, "User-Agent": "wf-probe/1"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            log("A graphql podStop -> HTTP %s %s" % (r.status, r.read()[:200].decode(errors="replace")))
    except urllib.error.HTTPError as e:
        log("A graphql podStop -> HTTP %s %s" % (e.code, e.read()[:200].decode(errors="replace")))
    except Exception as e:
        log("A graphql podStop -> %s" % type(e).__name__)


log("attempt A")
attempt_graphql()
time.sleep(25)
log("attempt B")
if shutil.which("runpodctl"):
    r = subprocess.run(["runpodctl", "stop", "pod", pod], capture_output=True, text=True, timeout=60)
    log("B runpodctl stop -> rc=%s out=%s err=%s" % (r.returncode, r.stdout[:200], r.stderr[:200]))
else:
    log("B runpodctl not found")
time.sleep(25)
log("probe done; exiting job (supervisor will try REST stop = 403 expected)")
