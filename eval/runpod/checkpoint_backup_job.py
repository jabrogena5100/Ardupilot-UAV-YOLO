#!/usr/bin/env python3
"""Pod-side checkpoint BACKUP job: copy the three Model B v2 checkpoints into a NEW export dir, verify SHA-256, and serve ONLY them (token required).

Run on a pod that mounts the `wildfire-datasets` volume (the OWNER launches it from the Ubuntu VM so the token never leaves the VM):
    python3 eval/runpod/launch_job.py transfer-test --checkpoint-backup --results-token-file ~/.wf_results_token ... --confirm-paid <run name> --watchdog-armed
Reads the originals (never modifies, moves or deletes them), refuses to continue on any hash mismatch, writes SHA256SUMS into the export, and serves via
results_server.py with --allow-v2-checkpoints semantics (exact three names only). Writes only a NEW /workspace/runs/transfer_test_*/export/.
"""
import argparse
import hashlib
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import results_server as RS  # noqa: E402

CKPTS = {
    "real_boreal_v2_best_seed0.pt": ("/workspace/runs/boreal_v2_try3/real_boreal_v2_seed0/real_boreal_v2_best_seed0.pt", "090c0f4cbe2718d2b084a41b8a3736fa27b673bc66b20c64672b13c83c299a05"),
    "real_boreal_v2_best_seed1.pt": ("/workspace/runs/boreal_v2_try4/real_boreal_v2_seed1/real_boreal_v2_best_seed1.pt", "5a9616cefddd3bca6eccf31d0691f7b1a2fadb572f84cf7948c85296644e9388"),
    "real_boreal_v2_best_seed2.pt": ("/workspace/runs/boreal_v2_try4/real_boreal_v2_seed2/real_boreal_v2_best_seed2.pt", "c4d443a8af919c6b00e4319b117a5dfbe90bd8b29750c75e3970a9389ae097ef"),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--serve-seconds", type=int, default=1800)
    a = ap.parse_args(argv)
    out = os.path.realpath(a.out)
    if not os.path.basename(out).startswith("transfer_test_") or os.path.dirname(out) != os.path.realpath("/workspace/runs"):
        sys.exit("refusing: output must be /workspace/runs/transfer_test_*")
    exp = os.path.join(out, "export")
    os.makedirs(exp)
    sums = []
    for name, (src, want) in CKPTS.items():
        got = RS.sha256_file(src) if os.path.isfile(src) else None
        if got != want:
            sys.exit(f"ABORT: {name} missing or hash mismatch at {src}: {got} != {want}")
        shutil.copyfile(src, os.path.join(exp, name))          # copy; the original is only read
        if RS.sha256_file(os.path.join(exp, name)) != want:
            sys.exit(f"ABORT: copy of {name} differs from the original")
        sums.append(f"{want}  {name}")
        print("verified", name, want[:16], flush=True)
    with open(os.path.join(exp, "SHA256SUMS"), "w") as f:
        f.write("\n".join(sums) + "\n")
    files = RS.write_manifest(exp, sorted(os.listdir(exp)))
    print("exported", {k: v["bytes"] for k, v in files.items()}, flush=True)
    print("serve result:", RS.serve(exp, 8000, os.environ.get("RESULTS_TOKEN", ""), a.serve_seconds, allow_v2_ckpt=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
