#!/usr/bin/env python3
"""Materialise the frozen Boreal v2 split as a symlink farm on the RunPod volume (runs ON the pod; stdlib only).

    python3 boreal_v2_materialize.py --src /workspace/boreal/boreal_yolo --dest /workspace/splits/boreal_v2 \
        --filelists filelists.json --scan /workspace/integrity/boreal_scan.json.gz --scan-sha256 8a4db032... \
        --expect train=<sha> --expect val=<sha> --expect test=<sha>

Builds `<dest>.partial` and renames it to `<dest>` only after every check passed; refuses to run if `<dest>` exists (a frozen
split is never rewritten); never writes into `--src`. Checks per file: source image exists, size and MD5 equal the scan record,
label file presence equals the scan, label line count equals the recorded box count. Checks per partition: file-list SHA-256
(sha256 of "\\n".join(sorted names)) equals the frozen value. Exit codes: 0 ok, 3 any mismatch, 4 destination exists.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import shutil
import sys

PARTS = ("train", "val", "test")


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def list_hash(names):
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def label_stats(path):
    """(n_boxes, [area fractions]) from a YOLO label file; raises on malformed lines."""
    n, areas = 0, []
    with open(path) as f:
        for ln in f:
            if not ln.strip():
                continue
            c, cx, cy, w, h = ln.split()[:5]
            if int(c) != 0 or not (0 <= float(cx) <= 1 and 0 <= float(cy) <= 1 and 0 < float(w) <= 1 and 0 < float(h) <= 1):
                raise ValueError(f"malformed label line in {path}: {ln.strip()}")
            n += 1
            areas.append(float(w) * float(h))
    return n, areas


def materialize(src, dest, filelists, scan, expect, verify_md5=True):
    errs = []
    if os.path.exists(dest):
        return 4, ["destination exists: " + dest]
    part_dir = dest + ".partial"
    shutil.rmtree(part_dir, ignore_errors=True)
    manifest = {"name": "boreal_v2", "source": src, "partitions": {}}
    for p in PARTS:
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(part_dir, sub, p))
        names, boxes, areas, nlab, sites = [], 0, [], 0, collections.Counter()
        for ent in filelists[p]:
            n, sp = ent["name"], ent["source_split"]
            img = os.path.join(src, "images", sp, n)
            stem = os.path.splitext(n)[0]
            lab = os.path.join(src, "labels", sp, stem + ".txt")
            rec = scan[sp].get(n)
            if rec is None or not os.path.isfile(img):
                errs.append(f"missing source image or scan record: {sp}/{n}")
                continue
            if os.path.getsize(img) != rec[1] or (verify_md5 and md5_file(img) != rec[0]):
                errs.append(f"source image differs from scan record: {sp}/{n}")
                continue
            has = os.path.isfile(lab)
            if has != ent["has_label_file"] or has != (rec[4] >= 0):
                errs.append(f"label presence mismatch: {sp}/{n}")
                continue
            if has:
                nb, ar = label_stats(lab)
                if nb != ent["n_boxes"] or nb != rec[4]:
                    errs.append(f"box count mismatch: {sp}/{n} ({nb} vs {ent['n_boxes']})")
                    continue
                boxes += nb
                areas += ar
                nlab += 1
                os.symlink(lab, os.path.join(part_dir, "labels", p, stem + ".txt"))
            os.symlink(img, os.path.join(part_dir, "images", p, n))
            names.append(n)
            sites[n.split("__")[0]] += 1
        digest = list_hash(names)
        if expect.get(p) and digest != expect[p]:
            errs.append(f"{p}: file-list sha256 {digest} != frozen {expect[p]}")
        areas.sort()
        manifest["partitions"][p] = {"images": len(names), "images_with_label_file": nlab, "boxes": boxes, "filelist_sha256": digest,
                                     "per_site_prefix": dict(sites),
                                     "median_box_area_fraction": (areas[len(areas) // 2] if areas else None)}
        with open(os.path.join(part_dir, p + ".txt"), "w") as f:
            f.write("\n".join(os.path.join(dest, "images", p, n) for n in sorted(names)) + "\n")
    if errs:
        return 3, errs
    with open(os.path.join(part_dir, "data.yaml"), "w") as f:
        f.write(f"path: {dest}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: smoke\n")
    with open(os.path.join(part_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.rename(part_dir, dest)
    return 0, []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--filelists", required=True)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--scan-sha256", required=True)
    ap.add_argument("--expect", action="append", default=[], help="PART=FILELIST_SHA256 (frozen)")
    ap.add_argument("--skip-md5", action="store_true")
    a = ap.parse_args(argv)
    h = hashlib.sha256(open(a.scan, "rb").read()).hexdigest()
    if h != a.scan_sha256:
        print(f"ABORT: scan sha256 {h} != {a.scan_sha256}", file=sys.stderr)
        return 3
    scan = json.load(gzip.open(a.scan))
    rc, errs = materialize(a.src, a.dest, json.load(open(a.filelists)), scan, dict(e.split("=", 1) for e in a.expect),
                           verify_md5=not a.skip_md5)
    for e in errs[:50]:
        print("ERROR:", e, file=sys.stderr)
    if rc == 0:
        print("MATERIALISED", a.dest, json.dumps({p: v["images"] for p, v in json.load(open(os.path.join(a.dest, "manifest.json")))["partitions"].items()}))
    return rc


if __name__ == "__main__":
    sys.exit(main())
