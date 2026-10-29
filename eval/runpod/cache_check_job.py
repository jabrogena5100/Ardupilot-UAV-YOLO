#!/usr/bin/env python3
"""Probe job: is Ultralytics cache=ram (or cache=disk) practical and pixel-equivalent for the Boreal v2 4K frames? Read-only on the volume.

Measures on the real pod: installed Ultralytics cache code (printed), container CPU/RAM limits, exact RAM needed to cache train+val at imgsz 640,
cache=disk size/speed (on LOCAL container disk only; never writes next to the volume images), and whether load_image output is byte-identical with and
without caching. Writes only a NEW /workspace/runs/transfer_test_*/export/ and serves it via results_server.py.
"""
import argparse
import glob
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import results_server as RS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--serve-seconds", type=int, default=420)
a = ap.parse_args()
out = os.path.realpath(a.out)
if not os.path.basename(out).startswith("transfer_test_") or os.path.dirname(out) != os.path.realpath("/workspace/runs"):
    sys.exit("refusing: output must be /workspace/runs/transfer_test_*")
exp = os.path.join(out, "export")
os.makedirs(exp)
res = {"t0": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=300).stdout.strip()


def save():
    json.dump(res, open(os.path.join(exp, "cache_check.json"), "w"), indent=1, default=str)
    RS.write_manifest(exp, [n for n in os.listdir(exp) if n not in ("export_manifest.json", "status.json", "export_manifest.json.tmp")])


res["pip"] = sh("pip install -q --break-system-packages 'ultralytics==8.4.173' 2>&1 | tail -1")
res["limits"] = {"RUNPOD_MEM_GB": os.environ.get("RUNPOD_MEM_GB"), "RUNPOD_CPU_COUNT": os.environ.get("RUNPOD_CPU_COUNT"), "nproc": os.cpu_count(),
                 "affinity": len(os.sched_getaffinity(0)), "cgroup_memory_max": sh("cat /sys/fs/cgroup/memory.max 2>/dev/null"),
                 "cgroup_cpu_max": sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null"), "free_g": sh("free -g | head -2"), "loadavg": open("/proc/loadavg").read().strip(),
                 "local_df": sh("df -h / /tmp | head -3")}
save()
import numpy as np  # noqa: E402
import ultralytics  # noqa: E402
from ultralytics.data import base as UB  # noqa: E402
res["ultralytics"] = ultralytics.__version__
with open(os.path.join(exp, "cache_source.txt"), "w") as f:
    for name in ("cache_images", "cache_images_to_disk", "check_cache_disk", "check_cache_ram", "cache_images_to_disk", "load_image", "set_rectangle"):
        fn = getattr(UB.BaseDataset, name, None)
        if fn is not None:
            f.write(f"\n##### BaseDataset.{name}\n" + inspect.getsource(fn))
    f.write("\n##### module: " + UB.__file__ + "\n")
save()

# exact RAM for train+val at imgsz 640 (same resize rule: r = 640 / max(h, w); w, h = ceil(w*r), ceil(h*r); 3 channels)
from PIL import Image  # noqa: E402
tot = {}
for part in ("train", "val"):
    b, n = 0, 0
    for p in sorted(glob.glob(f"/workspace/splits/boreal_v2/images/{part}/*.jpg")):
        w, h = Image.open(p).size
        r = 640 / max(w, h)
        b += math.ceil(w * r) * math.ceil(h * r) * 3
        n += 1
    tot[part] = {"images": n, "resized_bytes": b, "GB": round(b / 1e9, 3), "full_res_decoded_GB_if_cache_disk": None}
res["ram_estimate"] = tot
save()

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.dataset import YOLODataset  # noqa: E402
src = sorted(glob.glob("/workspace/splits/boreal_v2/images/train/*.jpg"))[::145][:24]
tmp = "/tmp/cachetest"
for sub in ("images/train", "labels/train"):
    os.makedirs(f"{tmp}/{sub}", exist_ok=True)
for p in src:
    shutil.copy(p, f"{tmp}/images/train/" + os.path.basename(p))
    lab = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    if os.path.exists(lab):
        shutil.copy(lab, f"{tmp}/labels/train/" + os.path.basename(lab))
hyp = get_cfg()
data = {"names": {0: "smoke"}, "channels": 3, "nc": 1}


def make(cache):
    return YOLODataset(img_path=f"{tmp}/images/train", imgsz=640, batch_size=16, augment=True, hyp=hyp, rect=False, cache=cache,
                       single_cls=False, stride=32, pad=0.0, prefix="", task="detect", classes=None, data=data, fraction=1.0)


try:
    d0 = make(False)
    t = time.time()
    base = [d0.load_image(i) for i in range(len(d0.im_files))]
    res["uncached_load_image_s_per_img"] = round((time.time() - t) / len(base), 3)
    t = time.time()
    d1 = make("ram")           # the constructor builds the RAM cache (ThreadPool) when cache == "ram"
    res["ram_cache_build_s_for_%d_imgs" % len(base)] = round(time.time() - t, 2)
    res["ram_cache_populated"] = sum(1 for im in d1.ims if im is not None)
    t = time.time()
    cached = [d1.load_image(i) for i in range(len(d1.im_files))]
    res["ram_cached_load_image_s_per_img"] = round((time.time() - t) / len(cached), 5)
    eq = [bool(np.array_equal(a_[0], b_[0])) and a_[1] == b_[1] and a_[2] == b_[2] for a_, b_ in zip(base, cached)]
    res["ram_vs_uncached_identical"] = {"all": all(eq), "n": len(eq), "shape0": list(base[0][0].shape), "dtype": str(base[0][0].dtype)}
except Exception as e:
    res["ram_check_error"] = type(e).__name__ + ": " + str(e)[:300]
save()
try:
    t = time.time()
    dd = make("disk")          # the constructor writes the .npy files (here: next to LOCAL copies in /tmp, never the volume)
    res["disk_cache_build_s_for_%d_imgs" % len(src)] = round(time.time() - t, 2)
    npy = sorted(glob.glob(f"{tmp}/images/train/*.npy"))
    sz = [os.path.getsize(p) for p in npy]
    res["disk_cache"] = {"npy_files": len(npy), "mean_MB": round(sum(sz) / max(1, len(sz)) / 1e6, 1),
                         "extrapolated_GB_train_plus_val": round(sum(sz) / max(1, len(sz)) * (tot["train"]["images"] + tot["val"]["images"]) / 1e9, 1)}
    t = time.time()
    dl = [dd.load_image(i) for i in range(len(dd.im_files))]
    res["disk_cached_load_image_s_per_img"] = round((time.time() - t) / len(dl), 3)
    res["disk_vs_uncached_identical"] = all(bool(np.array_equal(a_[0], b_[0])) for a_, b_ in zip(base, dl))
except Exception as e:
    res["disk_check_error"] = type(e).__name__ + ": " + str(e)[:300]
res["t1"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save()
print("cache check done", json.dumps({k: v for k, v in res.items() if k.startswith(("ram", "disk", "limits"))}, default=str)[:1500], flush=True)
print("serve:", RS.serve(exp, 8000, os.environ.get("RESULTS_TOKEN", ""), a.serve_seconds), flush=True)
