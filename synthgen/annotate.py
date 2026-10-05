#!/usr/bin/env python3
"""
synthgen/annotate.py — turn the rendered smoke alpha mask into a YOLO label.

The label is exact by construction (we rendered the smoke): the box is the
bounding box of the pixels where the smoke layer's alpha is at least
`box.alpha_threshold`, grown by `box.pad_frac` of its size on each side and clipped
to the image. This is NOT Boreal's annotation convention: Boreal boxes are loose,
hand-drawn and sometimes include large empty areas (docs/synthetic/
boreal_visual_analysis.md). That difference is a known, reported label-convention
gap, and the two parameters above are the knobs for studying it.

Format: one line per object, `class cx cy w h`, all normalised to [0, 1] (the
same as the Boreal YOLO labels). Negatives get an empty label file.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

SMOKE_CLASS = 0


def box_from_alpha(alpha: np.ndarray, threshold: float, pad_frac: float,
                   min_pixels: int = 1) -> Optional[Tuple[float, float, float, float]]:
    """Pixel box (x1, y1, x2, y2) around the smoke mask, or None if the mask is too small."""
    H, W = alpha.shape
    mask = alpha >= threshold
    if int(mask.sum()) < min_pixels:
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    x1, x2 = float(cols[0]), float(cols[-1] + 1)
    y1, y2 = float(rows[0]), float(rows[-1] + 1)
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - pad_frac * bw, x2 + pad_frac * bw
    y1, y2 = y1 - pad_frac * bh, y2 + pad_frac * bh
    return (max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2))


def box_area_frac(box: Tuple[float, float, float, float], width: int, height: int) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / float(width * height)


def to_yolo_line(box: Tuple[float, float, float, float], width: int, height: int, cls: int = SMOKE_CLASS) -> str:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height
    w, h = (x2 - x1) / width, (y2 - y1) / height
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def yolo_line_to_box(line: str, width: int, height: int) -> Tuple[int, Tuple[float, float, float, float]]:
    cls, cx, cy, w, h = line.split()
    cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
    return int(cls), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def box_variants(alpha: np.ndarray, variants: Dict[str, Dict[str, float]],
                 min_pixels: int = 1) -> Dict[str, Optional[Tuple[float, float, float, float]]]:
    """The same mask boxed under several label conventions (config `box.variants`), so a label-convention
    ablation can rewrite labels from metadata without re-rendering any image."""
    return {name: box_from_alpha(alpha, v["alpha_threshold"], v["pad_frac"], min_pixels) for name, v in variants.items()}
