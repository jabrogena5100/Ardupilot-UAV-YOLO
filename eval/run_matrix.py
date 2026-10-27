"""
eval/run_matrix.py — config-driven Level 2 evaluation: every cell goes through the same path.

    (checkpoint, dataset) -> predictions (one predictor) -> eval.metrics.evaluate -> JSON

    python3 -m eval.run_matrix --config eval/datasets.yaml --out runs/level2_<tag> \
        --var BOREAL_SPLIT=/workspace/splits/boreal_v1 --var SYNTH_ROOT=/workspace/synth_smoke_v1 \
        --var RUNS=/workspace/runs [--cells model_b:boreal_test,...] [--dry-run] [--preds-from DIR]

Every cell JSON records the checkpoint sha256, dataset pin, settings, git commit, library versions and
the command line. Predictions are saved (gzip JSON) so any number can be recomputed without a GPU.
Aborts when a dataset's image count or file-list hash differs from its pin, or when a pinned checkpoint
hash differs. An unpinned checkpoint (sha256: null) is allowed but flagged "UNPINNED" in the output.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import yaml

from . import metrics as M

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------------------------- config / datasets

def expand(obj: Any, env: Dict[str, str]) -> Any:
    if isinstance(obj, str):
        out = obj
        for k, v in env.items():
            out = out.replace("${" + k + "}", v)
        if "${" in out:
            raise KeyError(f"unresolved variable in {obj!r}; pass --var NAME=VALUE")
        return out
    if isinstance(obj, dict):
        return {k: expand(v, env) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v, env) for v in obj]
    return obj


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_path(img: str) -> str:
    """.../images/<split>/<name>.jpg -> .../labels/<split>/<name>.txt"""
    d, n = os.path.split(img)
    head, sep, tail = d.rpartition(os.sep + "images" + os.sep)
    if not sep:
        raise ValueError(f"{img}: expected an images/<split>/ directory layout")
    return os.path.join(head + os.sep + "labels" + os.sep + tail, os.path.splitext(n)[0] + ".txt")


def load_dataset(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve images + GT for a dataset entry and verify its frozen pin."""
    d = cfg["images_dir"]
    names = sorted(n for n in os.listdir(d) if os.path.splitext(n)[1].lower() in IMG_EXT)
    items = []
    for n in names:
        img = os.path.join(d, n)
        lp = label_path(img)
        items.append((n, img, lp))
    if cfg.get("only_positive"):
        items = [t for t in items if os.path.exists(t[2]) and os.path.getsize(t[2]) > 0]
    kept = [t[0] for t in items]
    digest = hashlib.sha256("\n".join(kept).encode()).hexdigest()
    errs = []
    if cfg.get("n_images") is not None and len(kept) != cfg["n_images"]:
        errs.append(f"{name}: {len(kept)} images, pinned {cfg['n_images']}")
    if cfg.get("filelist_sha256") and digest != cfg["filelist_sha256"]:
        errs.append(f"{name}: file-list sha256 {digest}, pinned {cfg['filelist_sha256']}")
    if errs:
        raise SystemExit("DATASET PIN MISMATCH (refusing to evaluate):\n  " + "\n  ".join(errs))
    gts = {n: M.load_gt(lp) for n, _, lp in items}
    bad = {int(c) for g in gts.values() for c in g[:, 0]} - {0}
    if bad:
        raise SystemExit(f"{name}: unexpected class ids {sorted(bad)}; unified format is single class 0 = smoke")
    return {"name": name, "images": {n: img for n, img, _ in items}, "gts": gts,
            "filelist_sha256": digest, "n_images": len(kept)}


# ----------------------------------------------------------------------------- prediction

def ultralytics_predictor(ckpt: str, images: Dict[str, str], s: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """The one inference path. Lazy import so the metric side runs without torch."""
    import torch
    from ultralytics import YOLO

    torch.manual_seed(int(s["seed"]))
    model = YOLO(ckpt)
    names = list(images)
    out: Dict[str, np.ndarray] = {}
    for i in range(0, len(names), int(s["batch"])):
        chunk = names[i:i + int(s["batch"])]
        res = model.predict([images[n] for n in chunk], imgsz=int(s["imgsz"]), conf=float(s["conf"]),
                            iou=float(s["nms_iou"]), max_det=int(s["max_det"]), verbose=False, augment=False)
        for n, r in zip(chunk, res):
            b = r.boxes
            if b is None or len(b) == 0:
                out[n] = np.zeros((0, 6))
            else:
                xyxyn = b.xyxyn.cpu().numpy()
                out[n] = np.concatenate([xyxyn, b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], axis=1)
    return out


def save_preds(path: str, preds: Dict[str, np.ndarray]) -> None:
    with gzip.open(path, "wt") as f:
        json.dump({k: np.round(v, 6).tolist() for k, v in preds.items()}, f)


def load_preds(path: str) -> Dict[str, np.ndarray]:
    with gzip.open(path, "rt") as f:
        return {k: np.asarray(v, float).reshape(-1, 6) for k, v in json.load(f).items()}


# ----------------------------------------------------------------------------- run

def git_info() -> Dict[str, Any]:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        c = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        d = bool(subprocess.run(["git", "-C", here, "status", "--porcelain", "--", "eval", "sim", "synthgen"],
                                capture_output=True, text=True).stdout.strip())
        return {"commit": c, "eval_sim_synthgen_dirty": d}
    except Exception:                                          # noqa: BLE001
        return {"commit": None, "eval_sim_synthgen_dirty": None}


def versions() -> Dict[str, Any]:
    v = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("ultralytics", "torch"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:                                      # noqa: BLE001
            v[mod] = None
    return v


def run_cell(model_name: str, model: Dict[str, Any], ds: Dict[str, Any], settings: Dict[str, Any],
             predictor: Callable, out_dir: str, preds_from: Optional[str], argv: List[str]) -> Dict[str, Any]:
    tag = f"{model_name}__{ds['name']}"
    ckpt = model["checkpoint"]
    pin = model.get("sha256")
    ck_hash, pinned = None, bool(pin)
    pred_file = os.path.join(preds_from or os.path.join(out_dir, "predictions"), f"{tag}.json.gz")
    if preds_from:
        preds = load_preds(pred_file)
    else:
        ck_hash = sha256_file(ckpt)
        if pin and ck_hash != pin:
            raise SystemExit(f"CHECKPOINT HASH MISMATCH for {model_name}: {ck_hash} != pinned {pin}")
        t0 = time.time()
        preds = predictor(ckpt, ds["images"], settings)
        os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
        save_preds(pred_file, preds)
        preds = load_preds(pred_file)   # evaluate exactly what was saved, so a recompute is bit-identical
        print(f"[{tag}] predicted {len(preds)} images in {time.time() - t0:.1f}s")
    res = M.evaluate(preds, ds["gts"], conf_thresholds=settings["report_conf"])
    cell = {"cell": tag, "model": model_name, "train_domain": model.get("train_domain"), "dataset": ds["name"],
            "checkpoint": ckpt, "checkpoint_sha256": ck_hash, "checkpoint_pinned": pinned,
            "dataset_n_images": ds["n_images"], "dataset_filelist_sha256": ds["filelist_sha256"],
            "settings": settings, "git": git_info(), "versions": versions(), "argv": argv,
            "predictions_file": os.path.basename(pred_file), "metrics": res}
    if not pinned:
        cell["warning"] = "UNPINNED checkpoint: sha256 not fixed in the config"
    os.makedirs(os.path.join(out_dir, "cells"), exist_ok=True)
    with open(os.path.join(out_dir, "cells", f"{tag}.json"), "w") as f:
        json.dump(cell, f, indent=2)
    return cell


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def write_summary(cells: List[Dict[str, Any]], cfg: Dict[str, Any], out_dir: str) -> None:
    rows = []
    for c in cells:
        m, op = c["metrics"], c["metrics"]["at_conf"][f"{cfg['settings']['report_conf'][0]:g}"]
        rows.append({"cell": c["cell"], "train": c["train_domain"], "test": c["dataset"], "images": m["n_images"],
                     "mAP50": m["mAP50"], "mAP50-95": m["mAP50-95"], "precision@op": op["precision"], "recall@op": op["recall"],
                     "FNR@op": op["false_negative_rate"], "FP/img@op": op["fp_per_image"],
                     "empty_img_FP_rate@op": op["image_fp_rate_on_empty"], "mean_conf_TP": op["mean_conf_tp"],
                     "mean_IoU_TP": op["loc_mean_iou_tp"], "center_err/diag": m["loc_center_err_diag_mean"],
                     "recall_edge": m["recall_by_edge_touch"]["edge_touching"]["recall"],
                     "recall_nonedge": m["recall_by_edge_touch"]["not_edge_touching"]["recall"],
                     "maxF1_conf": (m["max_f1"] or {}).get("conf_threshold"), "maxF1": (m["max_f1"] or {}).get("f1"),
                     "unpinned": not c["checkpoint_pinned"]})
    with open(os.path.join(out_dir, "matrix.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    op_c = cfg["settings"]["report_conf"][0]
    lines = ["# Level 2 evaluation matrix", "",
             f"conf floor {cfg['settings']['conf']}, NMS IoU {cfg['settings']['nms_iou']}, imgsz {cfg['settings']['imgsz']}; "
             f"@op = IoU 0.5 at conf >= {op_c}. Same code path for every row (`eval/run_matrix.py` -> `eval/metrics.py`).", "",
             "| train -> test | images | mAP50 | mAP50-95 | P@op | R@op | FNR@op | FP/img@op | empty-img FP | mean conf (TP) | mean IoU (TP) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['train']} -> {r['test']}{' (UNPINNED)' if r['unpinned'] else ''} | {r['images']} | {fmt(r['mAP50'])} | "
                     f"{fmt(r['mAP50-95'])} | {fmt(r['precision@op'])} | {fmt(r['recall@op'])} | {fmt(r['FNR@op'])} | "
                     f"{fmt(r['FP/img@op'])} | {fmt(r['empty_img_FP_rate@op'])} | {fmt(r['mean_conf_TP'])} | {fmt(r['mean_IoU_TP'])} |")
    lines += ["", "Edge-touch diagnostic (recall at the same operating point, GT boxes touching the image border vs not):", "",
              "| cell | recall edge-touching | recall not edge-touching |", "|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['cell']} | {fmt(r['recall_edge'])} | {fmt(r['recall_nonedge'])} |")
    chk = []
    by = {c["cell"]: c for c in cells}
    for rc in cfg.get("reference_checks", []):
        c = by.get(f"{rc['model']}__{rc['dataset']}")
        if not c:
            continue
        for k, v in rc["expect"].items():
            got, tol = c["metrics"][k], rc["tolerance"][k]
            chk.append(f"| {rc['model']} on {rc['dataset']} | {k} | {v} | {got:.4f} | {got - v:+.4f} | {'within' if abs(got - v) <= tol else 'OUTSIDE'} {tol} |")
    if chk:
        lines += ["", "Reproduction check against the one-off Ultralytics numbers (reported only):", "",
                  "| cell | metric | reference | harness | diff | tolerance |", "|---|---|---|---|---|---|"] + chk
    lines += ["", "Limitations: `synth_smoke_v1` edge-touch rate ~0.76 vs Boreal ~0.30 (documented, not corrected); the Boreal test set "
              "has no usable negative images; Model A is a baseline, not evidence the synthetic domain is optimal."]
    with open(os.path.join(out_dir, "matrix.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="eval/datasets.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--var", action="append", default=[], help="NAME=VALUE for ${NAME} in the config")
    ap.add_argument("--cells", help="comma list model:dataset (default: all cells in the config)")
    ap.add_argument("--dry-run", action="store_true", help="verify dataset pins and checkpoint hashes only")
    ap.add_argument("--preds-from", help="recompute metrics from saved predictions in this directory (no model needed)")
    a = ap.parse_args(argv)
    env = {**{k: v for k, v in os.environ.items()}, **dict(v.split("=", 1) for v in a.var)}
    cfg = expand(yaml.safe_load(open(a.config)), env)
    want = {tuple(c.split(":")) for c in a.cells.split(",")} if a.cells else None
    cells_cfg = [c for c in cfg["cells"] if want is None or (c["model"], c["dataset"]) in want]
    if not cells_cfg:
        raise SystemExit("no cells selected")
    dsets: Dict[str, Dict] = {}
    for c in cells_cfg:
        if c["dataset"] not in dsets:
            dsets[c["dataset"]] = load_dataset(c["dataset"], cfg["datasets"][c["dataset"]])
            print(f"dataset {c['dataset']}: {dsets[c['dataset']]['n_images']} images, pin OK")
    for c in cells_cfg:
        m = cfg["models"][c["model"]]
        if not a.preds_from and os.path.exists(m["checkpoint"]):
            h = sha256_file(m["checkpoint"])
            ok = (not m.get("sha256")) or h == m["sha256"]
            print(f"model {c['model']}: {h[:12]}... {'pin OK' if m.get('sha256') and ok else 'UNPINNED' if not m.get('sha256') else 'PIN MISMATCH'}")
            if not ok:
                return 3
        elif not a.preds_from:
            print(f"model {c['model']}: checkpoint missing at {m['checkpoint']}")
            if not a.dry_run:
                return 2
    if a.dry_run:
        return 0
    os.makedirs(a.out, exist_ok=True)
    argv_used = sys.argv if argv is None else argv
    cells = [run_cell(c["model"], cfg["models"][c["model"]], dsets[c["dataset"]], cfg["settings"],
                      ultralytics_predictor, a.out, a.preds_from, list(argv_used)) for c in cells_cfg]
    write_summary(cells, cfg, a.out)
    print(open(os.path.join(a.out, "matrix.md")).read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
