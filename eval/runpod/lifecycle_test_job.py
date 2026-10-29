#!/usr/bin/env python3
"""Disposable job for the paid one-minute lifecycle test. Trains/evaluates nothing and touches no model or dataset.

Writes one small file into a NEW directory under <mount>/runs/ whose name starts with lifecycle_test_, sleeps briefly (so the
supervisor, watchdog and stop path are exercised on a live pod) and exits. Refuses any other output location.
"""
import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--sleep-s", type=float, default=20.0)
a = ap.parse_args()
out = os.path.realpath(a.out)
if not os.path.basename(out).startswith("lifecycle_test_") or os.path.dirname(out) != os.path.realpath("/workspace/runs"):
    sys.exit("refusing: output must be /workspace/runs/lifecycle_test_*")
os.makedirs(out)  # raises if it exists: never overwrite
time.sleep(a.sleep_s)
with open(os.path.join(out, "payload.txt"), "w") as f:
    f.write(f"lifecycle test ok {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
print("lifecycle job done")
