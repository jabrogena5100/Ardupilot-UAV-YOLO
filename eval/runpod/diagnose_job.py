#!/usr/bin/env python3
"""Read-only diagnostic job: measure the host and export small log files of earlier runs through results_server.py.

Writes ONLY a new directory /workspace/runs/transfer_test_*/export/ (refuses anything else); reads (never modifies) the aborted run's
logs and a few split images. Exports: host.json (CPU, memory, mounts, disk and image read/decode timing), tails of supervisor and
training logs, results.csv of the aborted attempt, directory listings. Never exports checkpoints or images.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import results_server as RS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--serve-seconds", type=int, default=600)
a = ap.parse_args()
out = os.path.realpath(a.out)
if not os.path.basename(out).startswith("transfer_test_") or os.path.dirname(out) != os.path.realpath("/workspace/runs"):
    sys.exit("refusing: output must be /workspace/runs/transfer_test_*")
exp = os.path.join(out, "export")
os.makedirs(exp)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120).stdout.strip()


def tail(path, n=200_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read()
    except OSError:
        return b""


def decode(p):
    from PIL import Image
    t = time.time()
    with open(p, "rb") as f:
        data = f.read()
    t_read = time.time() - t
    t = time.time()
    import io
    Image.open(io.BytesIO(data)).convert("RGB").load()
    return len(data), t_read, time.time() - t


host = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "nproc": os.cpu_count(), "cpu_model": sh("grep -m1 'model name' /proc/cpuinfo"),
        "mem": sh("grep -E 'MemTotal|MemAvailable' /proc/meminfo"), "loadavg": open("/proc/loadavg").read().strip(),
        "workspace_mount": sh("grep -E ' /workspace ' /proc/mounts"), "df": sh("df -h /workspace /root 2>/dev/null"),
        "gpu": sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
        "python": sys.version.split()[0]}
try:
    files = sorted(glob.glob("/workspace/splits/boreal_v2/images/train/*.jpg"))[::55][:64]
    cold = [decode(p) for p in files]
    warm = [decode(p) for p in files]
    from multiprocessing import Pool
    t = time.time()
    with Pool(os.cpu_count() or 4) as pool:
        pool.map(decode, files)
    host["image_probe"] = {"n": len(files), "mean_MB": round(sum(c[0] for c in cold) / len(cold) / 1e6, 2),
                           "cold_read_ms_per_img": round(1000 * sum(c[1] for c in cold) / len(cold), 1),
                           "warm_read_ms_per_img": round(1000 * sum(c[1] for c in warm) / len(warm), 1),
                           "decode_ms_per_img_single_core": round(1000 * sum(c[2] for c in cold) / len(cold), 1),
                           "pool_throughput_img_per_s": round(len(files) / (time.time() - t), 1)}
except Exception as e:
    host["image_probe_error"] = type(e).__name__ + ": " + str(e)[:150]
json.dump(host, open(os.path.join(exp, "host.json"), "w"), indent=1)
tag = os.environ.get("DIAG_TAG", "boreal_v2")
orch = "/workspace/runs/" + ("boreal_v2_train_eval" if tag == "boreal_v2" else tag + "_train_eval") + "_orch"
run = "/workspace/runs/" + tag
pairs = [(orch + "/events.jsonl", "orch_events.jsonl"), (orch + "/DONE.json", "orch_done.json"), (orch + "_watchdog.jsonl", "orch_watchdog.jsonl"),
         (run + "/results_server.log", "results_server.log"), (run + "/.phase", "phase.txt"), (run + "/PROTOCOL_DEVIATION.json", "protocol_deviation.json"),
         (run + "/INTEGRITY.json", "integrity.json")]
for sd in (0, 1, 2):
    pairs += [(f"{run}/real_boreal_v2_seed{sd}/results.csv", f"seed{sd}_results.csv"), (f"{run}/real_boreal_v2_seed{sd}/train_config.json", f"seed{sd}_train_config.json"),
              (f"{run}/real_boreal_v2_seed{sd}/args.yaml", f"seed{sd}_args.yaml")]
for src, dst in pairs:
    if os.path.isfile(src):
        open(os.path.join(exp, dst), "wb").write(tail(src, 400_000))
for src, dst in [(orch + "/job.log", "orch_job_tail.log")] + [(f"{run}/real_boreal_v2_seed{sd}_train.log", f"seed{sd}_train_tail.log") for sd in (0, 1, 2)]:
    if os.path.isfile(src):
        open(os.path.join(exp, dst), "wb").write(tail(src, 120_000))
sums = []
for ck in sorted(glob.glob(run + "/real_boreal_v2_seed*/real_boreal_v2_best_seed*.pt")):
    sums.append(f"{__import__('hashlib').sha256(open(ck, 'rb').read()).hexdigest()}  {ck}  {os.path.getsize(ck)}")
open(os.path.join(exp, "ckpt_sha256.txt"), "w").write("\n".join(sums) + "\n")
open(os.path.join(exp, "ls_runs.txt"), "w").write(sh(f"ls -la /workspace/runs {run} {orch} /workspace/splits 2>&1 | head -120"))
names = [n for n in sorted(os.listdir(exp)) if n not in ("export_manifest.json", "status.json")]
print("exported", {k: v["bytes"] for k, v in RS.write_manifest(exp, names).items()}, flush=True)
print("serve result:", RS.serve(exp, 8000, os.environ.get("RESULTS_TOKEN", ""), a.serve_seconds), flush=True)
