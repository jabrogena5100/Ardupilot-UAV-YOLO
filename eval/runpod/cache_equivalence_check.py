#!/usr/bin/env python3
"""Pod-side guard for the cache=ram protocol deviation: cached and uncached Ultralytics image loads must be byte-identical, and RAM must suffice.

    python3 cache_equivalence_check.py --images /workspace/splits/boreal_v2/images/train --n 16 --out DEV.json --min-mem-gb 16

Copies n images (+labels) to LOCAL /tmp (nothing is written next to the volume), builds YOLODataset with cache=False and cache='ram', compares load_image
arrays, shapes and sizes, and records RUNPOD_MEM_GB and the exact RAM estimate. Writes the result JSON (the protocol-deviation record). Exit 0 ok, 3 on any mismatch.
"""
import argparse
import glob
import json
import math
import os
import shutil
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--out", required=True)
ap.add_argument("--min-mem-gb", type=float, default=16)
a = ap.parse_args()
import numpy as np  # noqa: E402
import ultralytics  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.dataset import YOLODataset  # noqa: E402

files = sorted(glob.glob(os.path.join(a.images, "*.jpg")))
step = max(1, len(files) // a.n)
pick = files[::step][:a.n]
est = 0
for p in files:
    w, h = Image.open(p).size
    r = 640 / max(w, h)
    est += math.ceil(w * r) * math.ceil(h * r) * 3
tmp = "/tmp/cache_equiv"
shutil.rmtree(tmp, ignore_errors=True)
for sub in ("images/train", "labels/train"):
    os.makedirs(f"{tmp}/{sub}")
for p in pick:
    shutil.copy(p, f"{tmp}/images/train/" + os.path.basename(p))
    lab = os.path.realpath(p).replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    lab2 = p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    for l in (lab2, lab):
        if os.path.exists(l):
            shutil.copy(l, f"{tmp}/labels/train/" + os.path.basename(l))
            break
hyp, data = get_cfg(), {"names": {0: "smoke"}, "channels": 3, "nc": 1}


def make(cache):
    return YOLODataset(img_path=f"{tmp}/images/train", imgsz=640, batch_size=16, augment=True, hyp=hyp, rect=False, cache=cache, single_cls=False,
                       stride=32, pad=0.0, prefix="", task="detect", classes=None, data=data, fraction=1.0)


d0, d1 = make(False), make("ram")
eq = []
for i in range(len(d0.im_files)):
    x, y = d0.load_image(i), d1.load_image(i)
    eq.append(bool(np.array_equal(x[0], y[0])) and tuple(x[1]) == tuple(y[1]) and tuple(x[2]) == tuple(y[2]))
mem_gb = float(os.environ.get("RUNPOD_MEM_GB", "0") or 0)
rec = {"deviation": "cache=ram", "scope": "speed only: identical resized pixels, same augmentation/optimizer/epochs/seeds/splits",
       "v1_protocol_had_cache": False, "ultralytics": ultralytics.__version__, "checked_images": len(eq), "all_identical": all(eq),
       "ram_estimate_train_GB": round(est / 1e9, 3), "RUNPOD_MEM_GB": mem_gb, "min_required_GB": a.min_mem_gb,
       "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
rec["ok"] = rec["all_identical"] and mem_gb >= a.min_mem_gb
json.dump(rec, open(a.out, "w"), indent=1)
print("CACHE_EQUIVALENCE", json.dumps(rec))
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(0 if rec["ok"] else 3)
