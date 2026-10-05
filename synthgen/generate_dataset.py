#!/usr/bin/env python3
"""
synthgen/generate_dataset.py — write a synthetic smoke dataset in the same YOLO
layout as the frozen Boreal split, plus per-image metadata and a manifest.

    python3 -m synthgen.generate_dataset --out ~/synth_poc \\
        --counts train=120,val=30,test=30 --seed 0 --contact-sheet 24

Output (YOLO, class 0 = smoke):
    <out>/images/{train,val,test}/<id>.jpg
    <out>/labels/{train,val,test}/<id>.txt      (empty file = background image)
    <out>/data.yaml
    <out>/metadata.jsonl                         one JSON line per image: the full SceneParams,
                                                 target vs achieved box, attempts, render time
    <out>/manifest.json                          config hash, seeds, counts, file-list hashes, versions

Reproducible: a scene is determined by (config, --seed, split, index, attempt). Splits use
disjoint seed spaces and disjoint ground-texture banks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from typing import Any, Dict, List

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

from synthgen.annotate import box_area_frac, box_from_alpha, box_variants, to_yolo_line
from synthgen.render import Renderer
from synthgen.scene_params import SPLIT_IDS, load_config, sample_scene

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "domain_rand.yaml")


def parse_counts(s: str) -> Dict[str, Any]:
    """`train=3219:260,val=774:0,test=701:350` (positives:negatives) or `train=120` (negatives from the config fraction)."""
    out: Dict[str, Any] = {}
    for part in s.split(","):
        k, v = part.split("=")
        k = k.strip()
        if k not in SPLIT_IDS:
            raise ValueError(f"unknown split {k!r}")
        out[k] = tuple(int(x) for x in v.split(":")) if ":" in v else int(v)
    return out


def resolve_counts(cfg, counts: Dict[str, Any]) -> Dict[str, tuple]:
    """-> {split: (positives, negatives)}."""
    res = {}
    for split, v in counts.items():
        if isinstance(v, tuple):
            res[split] = (v[0], v[1])
        else:
            neg = int(round(float(cfg["negatives"]["fraction"][split]) * v))
            res[split] = (v - neg, neg)
    return res


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _sha256_text(items) -> str:
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def _git_info() -> Dict[str, Any]:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", root, "status", "--porcelain", "synthgen", "sim"],
                                    capture_output=True, text=True).stdout.strip())
        return {"commit": head, "synthgen_sim_dirty": dirty}
    except Exception:
        return {"commit": None}


def negative_indices(base_seed: int, split: str, n: int, frac: float = None, count: int = None) -> set:
    """Exactly `count` (or round(frac*n)) background images, spread over the split by a seeded permutation."""
    rng = np.random.default_rng(np.random.SeedSequence([int(base_seed), SPLIT_IDS[split], 424242]))
    k = int(count) if count is not None else int(round(frac * n))
    return set(int(i) for i in rng.permutation(n)[:k])


def generate_split(cfg, renderer: Renderer, split: str, n_pos: int, n_neg: int, base_seed: int, out: str,
                   max_attempts: int, meta_f, log_every: int = 10) -> List[Dict[str, Any]]:
    bx = cfg["box"]
    W, H = cfg["image"]["width"], cfg["image"]["height"]
    os.makedirs(f"{out}/images/{split}", exist_ok=True)
    os.makedirs(f"{out}/labels/{split}", exist_ok=True)
    n = n_pos + n_neg
    neg = negative_indices(base_seed, split, n, count=n_neg)
    records: List[Dict[str, Any]] = []
    t_split = time.time()
    for idx in range(n):
        positive = idx not in neg
        for attempt in range(max_attempts):
            t0 = time.time()
            sp = sample_scene(cfg, split, idx, base_seed, attempt, positive)
            res = renderer.render(sp)
            t_render = time.time() - t0
            box = None
            if positive:
                box = box_from_alpha(res.smoke_alpha, bx["alpha_threshold"], bx["pad_frac"], bx["min_alpha_pixels"])
                if box is None:
                    continue
                af = box_area_frac(box, W, H)
                if not (bx["min_area_frac"] <= af <= bx["max_area_frac"]):
                    continue
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                dense = float((res.smoke_alpha[y1:y2, x1:x2] >= bx["dense_alpha"]).mean())
                if dense < bx["min_dense_frac"]:        # smoke too faint to be a meaningful label
                    continue
            break
        else:
            raise RuntimeError(f"{sp.image_id}: no valid scene after {max_attempts} attempts")

        name = sp.image_id
        q = int(sp.jpeg_quality)
        cv2.imwrite(f"{out}/images/{split}/{name}.jpg", cv2.cvtColor(res.image, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), q])
        with open(f"{out}/labels/{split}/{name}.txt", "w") as f:
            f.write((to_yolo_line(box, W, H) + "\n") if box else "")
        rec = sp.to_dict()
        variants = box_variants(res.smoke_alpha, bx["variants"], bx["min_alpha_pixels"]) if positive else {}
        rec.update({"attempts_used": attempt + 1, "render_seconds": round(t_render, 3),
                    "label_convention": "default",
                    "box_variants": {k: ([round(float(x), 2) for x in v] if v else None) for k, v in variants.items()},
                    "box_xyxy": [round(v, 2) for v in box] if box else None,
                    "box_area_frac": round(box_area_frac(box, W, H), 5) if box else None,
                    "target_box_xyxy": [round(v, 2) for v in sp.smoke.target_box_xyxy] if sp.smoke else None,
                    "bank_slot": renderer.bank_slot(sp), "smoke_calibration": res.info})
        meta_f.write(json.dumps(rec) + "\n")
        records.append(rec)
        if log_every and (idx + 1) % log_every == 0:
            el = time.time() - t_split
            print(f"[{split}] {idx + 1}/{n}  {el / (idx + 1):.2f} s/img", flush=True)
    return records


def write_contact_sheet(out: str, records: List[Dict[str, Any]], k: int, cols: int = 4, thumb=(480, 270)) -> str:
    """Grid of the first k images (positives and negatives) with their label boxes drawn."""
    picks = records[:k]
    rows = (len(picks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb[0], rows * thumb[1]), (20, 20, 20))
    for n, rec in enumerate(picks):
        im = Image.open(f"{out}/images/{rec['split']}/{rec['image_id']}.jpg").convert("RGB")
        sx, sy = thumb[0] / im.width, thumb[1] / im.height
        im = im.resize(thumb, Image.LANCZOS)
        d = ImageDraw.Draw(im)
        if rec["box_xyxy"]:
            x1, y1, x2, y2 = rec["box_xyxy"]
            d.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=(255, 0, 0), width=2)
        d.text((6, 4), f"{rec['image_id']} {rec['archetype']}" + ("" if rec["is_positive"] else " NEG"), fill=(255, 255, 0))
        sheet.paste(im, ((n % cols) * thumb[0], (n // cols) * thumb[1]))
    path = f"{out}/contact_sheet.jpg"
    sheet.save(path, quality=90)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a synthetic smoke dataset (YOLO format).")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", required=True)
    ap.add_argument("--counts", required=True,
                    help="positives:negatives per split, e.g. train=3219:260,val=774:0,test=701:350 "
                         "(a plain number uses the config negatives.fraction)")
    ap.add_argument("--dataset-id", default="synth_smoke_poc", help="name recorded in the manifest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=8)
    ap.add_argument("--contact-sheet", type=int, default=0, help="write a contact sheet of the first K images")
    args = ap.parse_args()

    out = os.path.abspath(os.path.expanduser(args.out))
    if os.path.exists(os.path.join(out, "manifest.json")):
        print(f"refusing to overwrite an existing dataset at {out}", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)
    cfg = load_config(args.config)
    counts = resolve_counts(cfg, parse_counts(args.counts))
    with open(args.config) as f:
        cfg_text = f.read()
    renderer = Renderer(cfg, args.seed)

    t0 = time.time()
    all_records: List[Dict[str, Any]] = []
    with open(f"{out}/metadata.jsonl", "w") as meta_f:
        for split, (n_pos, n_neg) in counts.items():
            if n_pos + n_neg <= 0:
                continue
            all_records += generate_split(cfg, renderer, split, n_pos, n_neg, args.seed, out, args.max_attempts, meta_f)
    elapsed = time.time() - t0

    names_map = {int(k): v for k, v in cfg["classes"].items()}
    with open(f"{out}/data.yaml", "w") as f:
        yaml.safe_dump({"path": out, "train": "images/train", "val": "images/val", "test": "images/test",
                        "names": names_map}, f, sort_keys=False)

    # Two explicitly separate synthetic test views (reported separately, never merged):
    #   test_pos = smoke-positive images only (the primary comparison with the Boreal test composition)
    #   test_all = test_pos plus the background images (false-positive analysis)
    views: Dict[str, Any] = {}
    test_recs = sorted((r for r in all_records if r["split"] == "test"), key=lambda r: r["image_id"])
    if test_recs:
        os.makedirs(f"{out}/lists", exist_ok=True)
        for view, sel in (("test_pos", [r for r in test_recs if r["is_positive"]]), ("test_all", test_recs)):
            paths = [f"{out}/images/test/{r['image_id']}.jpg" for r in sel]
            with open(f"{out}/lists/{view}.txt", "w") as f:
                f.write("\n".join(paths) + "\n")
            with open(f"{out}/data_{view}.yaml", "w") as f:
                yaml.safe_dump({"path": out, "train": "images/train", "val": "images/val",
                                "test": f"lists/{view}.txt", "names": names_map}, f, sort_keys=False)
            views[view] = {"images": len(sel), "positives": sum(r["is_positive"] for r in sel),
                           "background_images": sum(not r["is_positive"] for r in sel),
                           "list_sha256": _sha256_text(sorted(os.path.basename(x) for x in paths))}

    manifest: Dict[str, Any] = {
        "generator": "synthgen.generate_dataset", "config_path": os.path.abspath(args.config),
        "dataset_id": args.dataset_id, "requested_counts": {k: {"positives": v[0], "negatives": v[1]} for k, v in counts.items()},
        "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(), "base_seed": args.seed,
        "seed_streams": {"scene": "[base_seed, split_id, index, attempt]", "split_ids": SPLIT_IDS,
                         "texture_bank": "[base_seed, split_id, 10000+slot]", "puff_bank": "[base_seed, split_id, 20000]",
                         "negative_selection": "[base_seed, split_id, 424242]"},
        "bank_sizes": {"textures": cfg["background"]["bank_size"], "puff_sprites": cfg["smoke"]["puff_bank_size"]},
        "test_views": views, "label_convention": "default (box.variants.default); alternatives in metadata.jsonl",
        "classes": cfg["classes"], "image_size": [cfg["image"]["width"], cfg["image"]["height"]],
        "splits": {}, "generation_seconds": round(elapsed, 2),
        "images_per_second": round(len(all_records) / max(elapsed, 1e-9), 4),
        "attempts_histogram": dict(sorted(Counter(r["attempts_used"] for r in all_records).items())),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
                     "pillow": Image.__version__, "git": _git_info()},
        "note": "synthetic data; no real imagery used. Smoke is rendered as flat billboards (no true volume).",
    }
    for split in counts:
        recs = [r for r in all_records if r["split"] == split]
        if not recs:
            continue
        names = sorted(f"{r['image_id']}.jpg" for r in recs)
        label_text = [_read(f"{out}/labels/{split}/{r['image_id']}.txt") for r in sorted(recs, key=lambda r: r["image_id"])]
        manifest["splits"][split] = {
            "images": len(recs), "positives": sum(r["is_positive"] for r in recs),
            "background_images": sum(not r["is_positive"] for r in recs),
            "filelist_sha256": _sha256_text(names), "labels_sha256": _sha256_text(label_text),
            "archetypes": dict(Counter(r["archetype"] for r in recs if r["is_positive"])),
            "mean_render_seconds": round(float(np.mean([r["render_seconds"] for r in recs])), 3),
        }
    with open(f"{out}/metadata.jsonl", "rb") as f:
        manifest["metadata_sha256"] = hashlib.sha256(f.read()).hexdigest()
    with open(f"{out}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if args.contact_sheet:
        # interleave so negatives appear: take every record in order across splits
        print("contact sheet:", write_contact_sheet(out, all_records, args.contact_sheet))
    print(f"done: {len(all_records)} images in {elapsed:.1f} s ({manifest['images_per_second']:.2f} img/s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
