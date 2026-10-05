#!/usr/bin/env python3
"""
eval/validate_dataset.py — one validator for any YOLO-format detection dataset (synthetic or Boreal).

It enforces the unified annotation format (same box format, same class names) so that one loader and
one evaluation script can serve every cell of the 2x2 matrix, and it prints the label statistics that
the synthetic generator is meant to approximate.

    python3 -m eval.validate_dataset --data ~/synth_poc/data.yaml \\
        --reference <path to boreal_label_stats.json> --report ~/synth_poc/validation.json

Hard checks (exit code 1 on failure):
  * every split directory exists; every image has a label file and vice versa
    (--allow-missing-labels treats a missing file as a background image, as in Boreal)
  * every image decodes; all images have the same size
  * every label line is `cls cx cy w h`; cls is in names; values are in [0, 1]; w, h > 0; box inside the image
  * no byte-identical image appears in two splits
  * if a manifest.json is present: file-list and label hashes match
Informational: per-split statistics, and a side-by-side with the Boreal train statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from PIL import Image

PCT = [5, 25, 50, 75, 95]


def _pct(a) -> Dict[str, float]:
    a = np.asarray(a, float)
    return {f"P{p}": round(float(np.percentile(a, p)), 4) for p in PCT} if len(a) else {}


def _sha(items) -> str:
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def _read(path: str, mode: str = "r"):
    with open(path, mode) as f:
        return f.read()


def validate(data_yaml: str, allow_missing_labels: bool = False, check_manifest: bool = True) -> Dict[str, Any]:
    cfg = yaml.safe_load(_read(data_yaml))
    root = cfg.get("path") or os.path.dirname(os.path.abspath(data_yaml))
    names = {int(k): v for k, v in cfg["names"].items()}
    errors: List[str] = []
    report: Dict[str, Any] = {"data": os.path.abspath(data_yaml), "names": names, "splits": {}}
    sizes = Counter()
    md5_seen: Dict[str, str] = {}

    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        img_dir = os.path.join(root, cfg[split])
        lbl_dir = os.path.join(root, cfg[split].replace("images", "labels", 1))
        if not os.path.isdir(img_dir):
            errors.append(f"[{split}] image dir missing: {img_dir}")
            continue
        imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        lbls = set(os.listdir(lbl_dir)) if os.path.isdir(lbl_dir) else set()
        stems = {os.path.splitext(f)[0] for f in imgs}
        orphan = [l for l in lbls if os.path.splitext(l)[0] not in stems]
        if orphan:
            errors.append(f"[{split}] {len(orphan)} label files without an image, e.g. {orphan[:2]}")
        area, w, h, cx, cy, asp = [], [], [], [], [], []
        nbox = Counter()
        edge = bg = bad = 0
        total_boxes = 0
        for f in imgs:
            stem = os.path.splitext(f)[0]
            path = os.path.join(img_dir, f)
            try:
                with Image.open(path) as im:
                    W, H = im.size
                    im.verify()
            except Exception as e:                      # noqa: BLE001
                errors.append(f"[{split}] cannot decode {f}: {e}")
                continue
            sizes[(W, H)] += 1
            digest = hashlib.md5(_read(path, "rb")).hexdigest()
            if digest in md5_seen and md5_seen[digest] != split:
                errors.append(f"image {f} in {split} is byte-identical to one in {md5_seen[digest]}")
            md5_seen.setdefault(digest, split)
            lp = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(lp):
                if allow_missing_labels:
                    bg += 1
                    nbox[0] += 1
                    continue
                errors.append(f"[{split}] missing label file for {f}")
                continue
            rows = [l.split() for l in _read(lp).splitlines() if l.strip()]
            if not rows:
                bg += 1
                nbox[0] += 1
                continue
            nbox[len(rows)] += 1
            for r in rows:
                total_boxes += 1
                try:
                    if len(r) != 5:
                        raise ValueError("expected 5 fields")
                    c = int(r[0])
                    x, y, bw, bh = (float(v) for v in r[1:])
                    if c not in names:
                        raise ValueError(f"class {c} not in names")
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                        raise ValueError("value out of [0,1] or non-positive size")
                    if x - bw / 2 < -1e-4 or x + bw / 2 > 1 + 1e-4 or y - bh / 2 < -1e-4 or y + bh / 2 > 1 + 1e-4:
                        raise ValueError("box extends outside the image")
                except Exception as e:                  # noqa: BLE001
                    bad += 1
                    errors.append(f"[{split}] {os.path.basename(lp)}: bad label {' '.join(r)} ({e})")
                    continue
                area.append(bw * bh); w.append(bw); h.append(bh); cx.append(x); cy.append(y)
                asp.append((bw * W) / (bh * H))
                if x - bw / 2 <= 0.002 or x + bw / 2 >= 0.998 or y - bh / 2 <= 0.002 or y + bh / 2 >= 0.998:
                    edge += 1
        n = len(imgs)
        report["splits"][split] = {
            "images": n, "background_images": bg, "background_frac": round(bg / n, 4) if n else None,
            "boxes": total_boxes, "boxes_per_image_hist": dict(sorted(nbox.items())), "bad_label_lines": bad,
            "box_area_frac": _pct(area), "box_w_frac": _pct(w), "box_h_frac": _pct(h),
            "box_cx": _pct(cx), "box_cy": _pct(cy), "box_pixel_aspect_w_over_h": _pct(asp),
            "frac_boxes_touching_image_edge": round(edge / max(1, total_boxes), 4),
            "filelist_sha256": _sha(imgs),
        }
    if len(sizes) > 1:
        errors.append(f"images have more than one size: {dict(sizes)}")
    report["image_sizes"] = {f"{w}x{h}": c for (w, h), c in sizes.items()}

    mpath = os.path.join(root, "manifest.json")
    if check_manifest and os.path.exists(mpath):
        man = json.loads(_read(mpath))
        for split, m in man.get("splits", {}).items():
            got = report["splits"].get(split)
            if got is None:
                errors.append(f"manifest lists split {split} not found in data.yaml")
                continue
            if got["filelist_sha256"] != m["filelist_sha256"]:
                errors.append(f"[{split}] file-list hash differs from manifest")
            lbl_dir = os.path.join(root, cfg[split].replace("images", "labels", 1))
            texts = [_read(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt"))
                     for f in sorted(f for f in os.listdir(os.path.join(root, cfg[split])) if f.lower().endswith(".jpg"))
                     if os.path.exists(os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt"))]
            if "labels_sha256" in m and _sha(texts) != m["labels_sha256"]:
                errors.append(f"[{split}] label hash differs from manifest")
        report["manifest_checked"] = True
    report["errors"] = errors
    report["ok"] = not errors
    return report


def compare_with_reference(report: Dict[str, Any], ref_path: str, split: str = "train", ref_split: str = "train") -> List[str]:
    """Plain-text side-by-side of the generator's statistics against the Boreal reference (informational)."""
    ref = json.loads(_read(ref_path))["splits"][ref_split]
    got = report["splits"][split]
    lines = [f"{'statistic':34s} {'synthetic ' + split:>16s} {'Boreal ' + ref_split:>16s}"]
    lines.append(f"{'background fraction':34s} {got['background_frac']:>16.3f} {ref['background_frac']:>16.3f}")
    lines.append(f"{'frac boxes touching image edge':34s} {got['frac_boxes_touching_image_edge']:>16.3f} {ref['frac_boxes_touching_image_edge']:>16.3f}")
    for key, label in (("box_area_frac", "box area / image"), ("box_w_frac", "box width frac"), ("box_h_frac", "box height frac"),
                       ("box_cx", "box centre x"), ("box_cy", "box centre y"), ("box_pixel_aspect_w_over_h", "box pixel aspect w/h")):
        for p in (5, 50, 95):
            g = got[key].get(f"P{p}")
            r = ref[key].get(str(p))
            lines.append(f"{label + ' P' + str(p):34s} {g if g is not None else float('nan'):>16.3f} {r if r is not None else float('nan'):>16.3f}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a YOLO-format detection dataset.")
    ap.add_argument("--data", required=True, help="path to data.yaml")
    ap.add_argument("--allow-missing-labels", action="store_true", help="a missing label file means a background image")
    ap.add_argument("--reference", default=None, help="Boreal label stats JSON to compare against")
    ap.add_argument("--report", default=None, help="write the full JSON report here")
    args = ap.parse_args()
    rep = validate(args.data, allow_missing_labels=args.allow_missing_labels)
    for split, s in rep["splits"].items():
        print(f"[{split}] images={s['images']} background={s['background_images']} boxes={s['boxes']} "
              f"area P50={s['box_area_frac'].get('P50')} edge-touch={s['frac_boxes_touching_image_edge']}")
    print("image sizes:", rep["image_sizes"])
    if args.reference and "train" in rep["splits"]:
        print()
        print("\n".join(compare_with_reference(rep, args.reference)))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(rep, f, indent=2)
    print()
    if rep["errors"]:
        print(f"VALIDATION FAILED ({len(rep['errors'])} problems):")
        for e in rep["errors"][:30]:
            print("  -", e)
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
