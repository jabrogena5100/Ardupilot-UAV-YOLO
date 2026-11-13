#!/usr/bin/env python3
"""Disposable job for the paid retrieval test: export a small synthetic artifact and serve it through results_server.py.

Touches no model, dataset or split. Writes only a NEW directory /workspace/runs/transfer_test_*/ (refuses anything else), then serves
it until the fetcher acknowledges or the window elapses. Run it under pod_supervisor.py with RESULTS_TOKEN in the environment.
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import results_server as RS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--serve-seconds", type=int, default=300)
a = ap.parse_args()
out = os.path.realpath(a.out)
if not os.path.basename(out).startswith("transfer_test_") or os.path.dirname(out) != os.path.realpath("/workspace/runs"):
    sys.exit("refusing: output must be /workspace/runs/transfer_test_*")
exp = os.path.join(out, "export")
os.makedirs(exp)  # raises if it exists: never overwrite
payload = b"".join(hashlib.sha256(str(i).encode()).digest() for i in range(8192))   # 256 KiB, deterministic
open(os.path.join(exp, "payload.dat"), "wb").write(payload)
open(os.path.join(exp, "note.txt"), "w").write("transfer test ok\n")
files = RS.write_manifest(exp, ["payload.dat", "note.txt"])
print("exported", {k: v["bytes"] for k, v in files.items()}, flush=True)
print("serve result:", RS.serve(exp, 8000, os.environ.get("RESULTS_TOKEN", ""), a.serve_seconds), flush=True)
