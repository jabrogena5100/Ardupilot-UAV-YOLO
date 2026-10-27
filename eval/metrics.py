"""
eval/metrics.py — the single metric code path for every cell of the 2x2 matrix.

Input is detector-agnostic: per image, predictions and ground truth as plain arrays. Nothing here
imports torch or ultralytics, so every number is computed by the same numpy code whatever model or
dataset produced the boxes (and the code is unit-tested without a GPU).

    preds[name] : (m, 6) float  x1, y1, x2, y2 (normalised 0-1), conf, cls
    gts[name]   : (n, 5) float  cls, x1, y1, x2, y2 (normalised 0-1)

Conventions (fixed; changing them changes every reported number):
  * Matching: per image, per IoU threshold, predictions in descending confidence take the best still
    unmatched same-class GT with IoU >= threshold (greedy, one-to-one). Because matching is greedy by
    confidence, the TP/FP flag of a prediction does not depend on any confidence cut-off.
  * AP: precision envelope, 101-point interpolation (Ultralytics/COCO style). mAP50 uses IoU 0.5;
    mAP50-95 averages IoU 0.50:0.05:0.95. Both are averaged over classes that have GT. As in
    Ultralytics, a perfect detector scores 0.995, not 1.0 (the appended end point of the curve).
  * Operating points: `max_f1` (confidence maximising raw F1 at IoU 0.5; Ultralytics smooths F1, so
    P/R can differ by a small amount) and fixed confidence thresholds (default 0.25) at IoU 0.5.
  * False-positive rate on empty scenes = fraction of GT-free images with >= 1 detection above the
    threshold; None when the set has no such images (the Boreal test set has one).
  * Edge-touch: a GT box with any side within 0.002 of the image border (same rule as
    eval/validate_dataset.py).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)
EDGE_EPS = 0.002


# ----------------------------------------------------------------------------- labels

def xywhn_to_xyxy(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4)
    out = np.empty_like(a)
    out[:, 0] = a[:, 0] - a[:, 2] / 2
    out[:, 1] = a[:, 1] - a[:, 3] / 2
    out[:, 2] = a[:, 0] + a[:, 2] / 2
    out[:, 3] = a[:, 1] + a[:, 3] / 2
    return out


def load_gt(label_path: str) -> np.ndarray:
    """YOLO txt (`cls cx cy w h`, normalised) -> (n,5) cls,x1,y1,x2,y2. Missing file = no boxes."""
    if not os.path.exists(label_path):
        return np.zeros((0, 5))
    rows = []
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if len(p) != 5:
                raise ValueError(f"{label_path}: expected 5 fields, got {len(p)}")
            rows.append([float(x) for x in p])
    if not rows:
        return np.zeros((0, 5))
    a = np.asarray(rows)
    return np.concatenate([a[:, :1], xywhn_to_xyxy(a[:, 1:])], axis=1)


def touches_edge(xyxy: np.ndarray, eps: float = EDGE_EPS) -> np.ndarray:
    xyxy = np.asarray(xyxy, float).reshape(-1, 4)
    return (xyxy[:, 0] <= eps) | (xyxy[:, 1] <= eps) | (xyxy[:, 2] >= 1 - eps) | (xyxy[:, 3] >= 1 - eps)


# ----------------------------------------------------------------------------- core maths

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP over the monotone precision envelope."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(np.interp(x, mrec, mpre), x))


def match_image(pred: np.ndarray, gt: np.ndarray, thr: float):
    """-> (tp flags aligned with `pred` rows sorted by conf desc, order, matched gt index or -1, iou)."""
    order = np.argsort(-pred[:, 4], kind="stable") if len(pred) else np.zeros(0, int)
    tp = np.zeros(len(pred), bool)
    gidx = -np.ones(len(pred), int)
    ious = np.zeros(len(pred))
    if len(pred) and len(gt):
        m = iou_matrix(pred[:, :4], gt[:, 1:5])
        m = m * (pred[:, 5:6] == gt[None, :, 0])
        used = np.zeros(len(gt), bool)
        for i in order:
            cand = np.where(~used & (m[i] >= thr))[0]
            if len(cand):
                j = cand[np.argmax(m[i, cand])]
                used[j] = True
                tp[i], gidx[i], ious[i] = True, j, m[i, j]
    return tp, order, gidx, ious


def _flatten(preds, gts, names):
    """Match every image at every IoU threshold; return flat per-prediction arrays."""
    conf, cls, img, gi, iou, flags = [], [], [], [], [], []
    for k, name in enumerate(names):
        p = np.asarray(preds.get(name, np.zeros((0, 6))), float).reshape(-1, 6)
        g = np.asarray(gts[name], float).reshape(-1, 5)
        per_thr = []
        for t in IOU_THRS:
            tp, _, gidx, ious = match_image(p, g, float(t))
            per_thr.append(tp)
            if t == IOU_THRS[0]:
                gi.append(gidx); iou.append(ious)
        conf.append(p[:, 4]); cls.append(p[:, 5]); img.append(np.full(len(p), k))
        flags.append(np.stack(per_thr, axis=1) if len(p) else np.zeros((0, len(IOU_THRS)), bool))
    cat = lambda xs, d: np.concatenate(xs) if xs else np.zeros(0, d)
    return (cat(conf, float), cat(cls, float), cat(img, int), cat(gi, int), cat(iou, float),
            np.concatenate(flags) if flags else np.zeros((0, len(IOU_THRS)), bool))


def _ap_for(conf, cls, flags_col, gt_cls_counts) -> Tuple[float, Dict[int, float]]:
    aps = {}
    for c, n in gt_cls_counts.items():
        sel = cls == c
        if n == 0:
            continue
        if not sel.any():
            aps[c] = 0.0
            continue
        o = np.argsort(-conf[sel], kind="stable")
        tp = flags_col[sel][o].astype(float)
        ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
        aps[c] = compute_ap(ctp / n, ctp / np.maximum(ctp + cfp, 1e-12))
    return (float(np.mean(list(aps.values()))) if aps else 0.0), aps


# ----------------------------------------------------------------------------- public API

def evaluate(preds: Dict[str, np.ndarray], gts: Dict[str, np.ndarray],
             conf_thresholds: Iterable[float] = (0.25,)) -> Dict[str, Any]:
    """Metrics for one (checkpoint, test set). `gts` defines the image universe."""
    names = sorted(gts)
    conf, cls, img, gidx, iou, flags = _flatten(preds, gts, names)
    gt_all = [np.asarray(gts[n], float).reshape(-1, 5) for n in names]
    n_gt = int(sum(len(g) for g in gt_all))
    counts: Dict[int, int] = {}
    for g in gt_all:
        for c in g[:, 0].astype(int):
            counts[int(c)] = counts.get(int(c), 0) + 1
    n_pos_img = sum(1 for g in gt_all if len(g))
    n_neg_img = len(names) - n_pos_img

    ap50, ap_by_cls = _ap_for(conf, cls, flags[:, 0], counts)
    ap_all = [_ap_for(conf, cls, flags[:, i], counts)[0] for i in range(len(IOU_THRS))]
    out: Dict[str, Any] = {
        "n_images": len(names), "n_positive_images": n_pos_img, "n_negative_images": n_neg_img,
        "n_gt_boxes": n_gt, "n_predictions": int(len(conf)),
        "mAP50": ap50, "mAP50-95": float(np.mean(ap_all)), "AP50_by_class": {str(k): v for k, v in ap_by_cls.items()},
    }
    tp50 = flags[:, 0]

    def at(c: float) -> Dict[str, Any]:
        keep = conf >= c
        tp, fp = int((tp50 & keep).sum()), int((~tp50 & keep).sum())
        fn = n_gt - tp
        d: Dict[str, Any] = {
            "conf_threshold": float(c), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / n_gt if n_gt else None,
            "false_negative_rate": fn / n_gt if n_gt else None,
            "false_discovery_rate": fp / (tp + fp) if tp + fp else None,
            "fp_per_image": fp / len(names) if names else None,
            "mean_conf_tp": float(conf[tp50 & keep].mean()) if (tp50 & keep).any() else None,
            "mean_conf_fp": float(conf[~tp50 & keep].mean()) if (~tp50 & keep).any() else None,
        }
        # image level
        has_det = np.zeros(len(names), bool)
        has_tp = np.zeros(len(names), bool)
        if len(conf):
            has_det[np.unique(img[keep])] = True
            has_tp[np.unique(img[tp50 & keep])] = True
        pos = np.array([len(g) > 0 for g in gt_all], bool)
        d["image_fp_rate_on_empty"] = float(has_det[~pos].mean()) if (~pos).any() else None
        d["image_detection_rate_on_positive"] = float(has_tp[pos].mean()) if pos.any() else None
        # localisation of true positives
        sel = tp50 & keep
        if sel.any():
            ious = iou[sel]
            d["loc_mean_iou_tp"] = float(ious.mean())
            d["loc_median_iou_tp"] = float(np.median(ious))
        else:
            d["loc_mean_iou_tp"] = d["loc_median_iou_tp"] = None
        return d

    # max-F1 operating point (IoU 0.5), raw F1
    if len(conf) and n_gt:
        o = np.argsort(-conf, kind="stable")
        ctp = np.cumsum(tp50[o]); cfp = np.cumsum(~tp50[o])
        p = ctp / np.maximum(ctp + cfp, 1e-12); r = ctp / n_gt
        f1 = 2 * p * r / np.maximum(p + r, 1e-12)
        b = int(np.argmax(f1))
        out["max_f1"] = {**at(float(conf[o][b])), "f1": float(f1[b])}
    else:
        out["max_f1"] = None
    out["at_conf"] = {f"{c:g}": at(c) for c in conf_thresholds}

    # localisation: centre error as a fraction of GT box diagonal, over all IoU>=0.5 TPs (any conf)
    cerr = []
    if tp50.any():
        pk = _flat_pred_boxes(preds, names)
        for fi in np.where(tp50)[0]:
            g = gt_all[img[fi]][gidx[fi], 1:5]
            pb = pk[fi]
            cp = np.array([(pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2]); cg = np.array([(g[0] + g[2]) / 2, (g[1] + g[3]) / 2])
            cerr.append(np.linalg.norm(cp - cg) / max(np.hypot(g[2] - g[0], g[3] - g[1]), 1e-12))
    out["loc_center_err_diag_mean"] = float(np.mean(cerr)) if cerr else None

    # pre-declared diagnostic: recall split by GT edge-touching vs not (first conf threshold, IoU 0.5)
    c0 = float(list(conf_thresholds)[0])
    keep = conf >= c0
    groups = {}
    matched = {(int(i), int(j)) for i, j in zip(img[tp50 & keep], gidx[tp50 & keep])}
    for label in ("edge_touching", "not_edge_touching"):
        n = hit = 0
        for k, g in enumerate(gt_all):
            if not len(g):
                continue
            e = touches_edge(g[:, 1:5])
            for j in range(len(g)):
                if e[j] == (label == "edge_touching"):
                    n += 1; hit += (k, j) in matched
        groups[label] = {"n_gt": n, "recall": hit / n if n else None, "conf_threshold": c0}
    out["recall_by_edge_touch"] = groups
    return out


def _flat_pred_boxes(preds, names) -> np.ndarray:
    parts = [np.asarray(preds.get(n, np.zeros((0, 6))), float).reshape(-1, 6)[:, :4] for n in names]
    return np.concatenate(parts) if parts else np.zeros((0, 4))


def _per_image(preds, gts, names):
    """Match every image once (all IoU thresholds) and keep the per-image pieces a resample needs."""
    out = []
    for n in names:
        p = np.asarray(preds.get(n, np.zeros((0, 6))), float).reshape(-1, 6)
        g = np.asarray(gts[n], float).reshape(-1, 5)
        fl = np.stack([match_image(p, g, float(t))[0] for t in IOU_THRS], axis=1) if len(p) else np.zeros((0, len(IOU_THRS)), bool)
        out.append((p[:, 4], p[:, 5], fl, g[:, 0].astype(int)))
    return out


def _headline(items, conf: float) -> Dict[str, Optional[float]]:
    """Complete detection metric (AP50, AP50-95, P/R/FP-per-image at `conf`) for a list of per-image pieces."""
    cf = np.concatenate([i[0] for i in items]) if items else np.zeros(0)
    cl = np.concatenate([i[1] for i in items]) if items else np.zeros(0)
    fl = np.concatenate([i[2] for i in items]) if items else np.zeros((0, len(IOU_THRS)), bool)
    counts: Dict[int, int] = {}
    for i in items:
        for c in i[3]:
            counts[int(c)] = counts.get(int(c), 0) + 1
    n_gt = sum(counts.values())
    ap50 = _ap_for(cf, cl, fl[:, 0], counts)[0]
    ap = float(np.mean([_ap_for(cf, cl, fl[:, k], counts)[0] for k in range(len(IOU_THRS))]))
    keep = cf >= conf
    tp = int((fl[:, 0] & keep).sum()); fp = int((~fl[:, 0] & keep).sum())
    return {"mAP50": ap50, "mAP50-95": ap, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / n_gt if n_gt else None, "fp_per_image": fp / len(items) if items else None}


def bootstrap_ci(preds: Dict[str, np.ndarray], gts: Dict[str, np.ndarray], n: int = 1000, seed: int = 0,
                 conf: float = 0.25, alpha: float = 0.05,
                 clusters: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Percentile bootstrap CIs for the headline metrics. Every resample recomputes the complete detection metric
    (precision-recall curves, AP at 10 IoU thresholds, P/R/FP at `conf`) on the resampled set; only the per-image
    greedy matching is cached, which is exact because matching is within-image and independent of resampling.

    Unit of resampling: images (with replacement), or, when `clusters` maps image name -> cluster id (e.g. source
    video), whole clusters (with replacement), which keeps correlated frames together.
    Assumptions/limits: captures test-set sampling variability only. It says nothing about training-seed variance
    (needs repeat runs), the model/threshold are held fixed, the test set is treated as a sample from a population of
    images (or clusters), the operating `conf` is fixed in advance, and the percentile method is crude for very few
    clusters (Boreal test has 5 videos) or at a metric ceiling (zero-width intervals).
    """
    names = sorted(gts)
    per = _per_image(preds, gts, names)
    rng = np.random.default_rng(seed)
    keys = ["mAP50", "mAP50-95", "precision", "recall", "fp_per_image"]
    vals = {k: [] for k in keys}
    if clusters is None:
        units = [[i] for i in range(len(names))]
    else:
        cid = sorted({clusters[n_] for n_ in names})
        units = [[i for i, n_ in enumerate(names) if clusters[n_] == c] for c in cid]
    for _ in range(n):
        pick = rng.integers(0, len(units), len(units))
        items = [per[i] for u in pick for i in units[u]]
        h = _headline(items, conf)
        for k in keys:
            if h[k] is not None:
                vals[k].append(h[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"n_resamples": n, "seed": seed, "conf_threshold": conf, "level": 1 - alpha,
            "unit": "image" if clusters is None else f"cluster ({len(units)} clusters)",
            "point": _headline(per, conf),
            "ci": {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] if v else None for k, v in vals.items()}}
