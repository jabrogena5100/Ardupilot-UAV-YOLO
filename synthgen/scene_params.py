#!/usr/bin/env python3
"""
synthgen/scene_params.py — everything random about one synthetic image, sampled
from synthgen/configs/domain_rand.yaml and recorded in one SceneParams.

A scene is fully determined by (config, base_seed, split, index, attempt): the
renderer takes a SceneParams and does no sampling of its own, so an image can be
re-rendered, and its label re-derived, from its metadata line alone.

Camera geometry goes through sim/camera.py, the same projection the renderer
uses, so where the smoke is "placed" and where it "lands" cannot disagree.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from sim.camera import Camera, CameraIntrinsics, CameraPose

SPLIT_IDS = {"train": 0, "val": 1, "test": 2}


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("version") != 1:
        raise ValueError(f"unsupported config version {cfg.get('version')!r}")
    return cfg


def scene_rng(base_seed: int, split: str, index: int, attempt: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(base_seed), SPLIT_IDS[split], int(index), int(attempt)]))


def bank_rng(base_seed: int, split: str, slot: int) -> np.random.Generator:
    """RNG for texture-bank entry `slot` of a split. A different spawn key than scenes, and
    different per split, so no ground texture is ever shared between splits."""
    return np.random.default_rng(np.random.SeedSequence([int(base_seed), SPLIT_IDS[split], 10_000 + int(slot)]))


# ----------------------------------------------------------------------
# parameter records
# ----------------------------------------------------------------------

@dataclass
class SmokeParams:
    style: str                       # column | plume | curtain
    anchor_n: float                  # ground point where the smoke starts (world metres)
    anchor_e: float
    height_m: float                  # plume height (curtain: height of the billow cores)
    spread_m: float                  # horizontal half-width scale
    wind_az_deg: float               # compass direction the smoke drifts toward
    lean: float                      # drift as a fraction of height
    opacity: float                   # overall density scale 0..1
    puff_alpha: float                # per-puff peak alpha
    n_puffs: int
    color_mode: str                  # white | grey | tan | mixed
    shape_seed: int
    target_box_xyxy: Tuple[float, float, float, float]   # intended box in pixels (before rendering)
    line_len_m: float = 0.0          # curtain: length of the burn line


@dataclass
class SceneParams:
    image_id: str
    split: str
    index: int
    attempt: int
    is_positive: bool
    archetype: str
    width: int
    height: int
    hfov_deg: float
    cam_alt_m: float
    cam_heading_deg: float
    cam_pitch_deg: float
    horizon_row_px: float
    texture_kind: str
    texture_pick: float              # u in [0,1): which bank entry of that kind
    tex_rot_deg: float
    tex_scale: float
    tex_off: Tuple[float, float]
    sun_az_deg: float
    sun_el_deg: float
    exposure: float
    ground_gain: float
    haze_per_m: float
    sky_zenith: Tuple[float, float, float]
    sky_horizon: Tuple[float, float, float]
    cloud_cover: float
    cloud_seed: int
    cloud_scale: float
    grade: Tuple[float, float, float]
    noise_sigma: float
    blur_sigma_px: float
    jpeg_quality: int
    smoke: Optional[SmokeParams] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics(self.width, self.height, self.hfov_deg)

    @property
    def camera(self) -> Camera:
        return Camera(self.intrinsics, CameraPose(0.0, 0.0, self.cam_alt_m, self.cam_heading_deg, self.cam_pitch_deg))


# ----------------------------------------------------------------------
# sampling helpers
# ----------------------------------------------------------------------

def _u(rng, r) -> float:
    return float(rng.uniform(r[0], r[1]))


def _pick(rng, weights: Dict[str, float]) -> str:
    keys = sorted(weights)
    p = np.array([weights[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def _lognormal_clipped(rng, median, sigma, lo=None, hi=None) -> float:
    v = float(median * math.exp(sigma * rng.standard_normal()))
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def _tnormal(rng, mean, sd, lo, hi) -> float:
    return float(np.clip(rng.normal(mean, sd), lo, hi))


# ----------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------

def sample_scene(cfg: Dict[str, Any], split: str, index: int, base_seed: int, attempt: int = 0,
                 is_positive: bool = True) -> SceneParams:
    rng = scene_rng(base_seed, split, index, attempt)
    W, H = int(cfg["image"]["width"]), int(cfg["image"]["height"])
    arch_cfg = cfg["archetypes"]
    arch = _pick(rng, {k: v["weight"] for k, v in arch_cfg.items()})
    a = arch_cfg[arch]

    # ---- camera -------------------------------------------------------
    hfov = _u(rng, cfg["camera"]["hfov_deg"])
    intr = CameraIntrinsics(W, H, hfov)
    alt = _u(rng, a["alt_m"])
    heading = float(rng.uniform(0, 360))
    if a.get("horizon_row") is not None:
        v_h = _u(rng, a["horizon_row"]) * H
        pitch = math.degrees(math.atan((intr.cy - v_h) / intr.fy))      # v_h = cy - fy*tan(pitch)
    else:
        pitch = _u(rng, a["pitch_deg"])
    pitch = float(np.clip(pitch, -40.0, 89.0))
    cam = Camera(intr, CameraPose(0.0, 0.0, alt, heading, pitch))
    horizon_row = cam.horizon_row()

    # ---- environment --------------------------------------------------
    at = cfg["atmosphere"]
    bg = cfg["background"]
    kind = _pick(rng, a["ground_kinds"])
    sun_el = _u(rng, at["sun_elevation_deg"])
    backlit = bool(rng.random() < at["backlit_probability"])
    ground_gain = _u(rng, at["backlit_ground_gain"]) if backlit else 1.0
    cloud_cover = _u(rng, a["cloud_cover"])
    if not is_positive:
        cloud_cover = min(0.9, cloud_cover + bg["clouds_in_negatives_bias"] * float(rng.random()))
    zen = np.array([0.20, 0.42, 0.78]) * rng.uniform(0.85, 1.1)
    hor = np.array([0.72, 0.84, 0.92]) * rng.uniform(0.9, 1.05)
    if backlit:
        hor = np.clip(hor * 1.08, 0, 1)

    p = dict(
        image_id=f"{split}_{index:06d}", split=split, index=int(index), attempt=int(attempt),
        is_positive=bool(is_positive), archetype=arch, width=W, height=H, hfov_deg=hfov,
        cam_alt_m=alt, cam_heading_deg=heading, cam_pitch_deg=pitch, horizon_row_px=float(horizon_row),
        texture_kind=kind, texture_pick=float(rng.random()), tex_rot_deg=float(rng.uniform(0, 360)),
        tex_scale=_u(rng, bg["scale"]), tex_off=(float(rng.uniform(0, 1024)), float(rng.uniform(0, 1024))),
        sun_az_deg=float(rng.uniform(0, 360)), sun_el_deg=sun_el,
        exposure=_u(rng, at["exposure"]), ground_gain=float(ground_gain),
        haze_per_m=_u(rng, at["haze_per_m"]),
        sky_zenith=tuple(float(x) for x in np.clip(zen, 0, 1)), sky_horizon=tuple(float(x) for x in np.clip(hor, 0, 1)),
        cloud_cover=float(cloud_cover), cloud_seed=int(rng.integers(0, 2**31 - 1)), cloud_scale=float(rng.uniform(0.6, 1.6)),
        grade=tuple(float(x) for x in rng.uniform(0.94, 1.06, size=3)),
        noise_sigma=_u(rng, cfg["post"]["noise_sigma"]), blur_sigma_px=_u(rng, cfg["post"]["blur_sigma_px"]),
        jpeg_quality=int(rng.integers(cfg["image"]["jpeg_quality"][0], cfg["image"]["jpeg_quality"][1] + 1)),
        smoke=None,
    )
    if not is_positive:
        return SceneParams(**p)

    # ---- target box (image space) ----------------------------------------
    area = _lognormal_clipped(rng, a["box_area"]["median"], a["box_area"]["sigma"], a["box_area"]["lo"], a["box_area"]["hi"])
    asp = _lognormal_clipped(rng, a["box_aspect_wh"]["median"], a["box_aspect_wh"]["sigma"], 0.4, 4.5)
    w_px = math.sqrt(area * W * H * asp)
    h_px = w_px / asp
    if w_px > 0.98 * W:
        w_px = 0.98 * W
        h_px = area * W * H / w_px
    if h_px > 0.98 * H:
        h_px = 0.98 * H
        w_px = area * W * H / h_px
    cx = _tnormal(rng, a["box_cx"]["mean"], a["box_cx"]["sd"], 0.10, 0.92) * W
    cy = _tnormal(rng, a["box_cy"]["mean"], a["box_cy"]["sd"], 0.15, 0.80) * H

    if a.get("base_below_horizon_px") is not None:        # horizon archetypes: base just under the horizon
        base_v = horizon_row + _u(rng, a["base_below_horizon_px"])
    else:                                                  # steep views: base at the bottom of the box
        base_v = min(cy + h_px / 2.0, H - 6.0)
    base_v = float(np.clip(base_v, max(horizon_row + 6.0, 4.0), H - 4.0))
    base_u = float(np.clip(cx, 0.08 * W, 0.95 * W))

    n0, e0, ok = cam.unproject_ground(np.array(base_u), np.array(base_v))
    if not bool(ok):
        raise RuntimeError("smoke base is above the horizon; check config ranges")
    an, ae = float(n0), float(e0)
    _, depth = cam.project(np.array([an, ae, 0.0]))
    scale = float(depth) / intr.fx                                       # metres per pixel at the smoke

    # plume height: choose it so the projected vertical extent matches the target box height
    ph = h_px * scale
    for _ in range(3):
        uv_top, _ = cam.project(np.array([an, ae, ph]))
        app = base_v - float(uv_top[1]) if np.isfinite(uv_top[1]) else h_px
        if app <= 1.0:
            ph *= 2.0
            continue
        ph *= h_px / app
    ph = float(np.clip(ph, 8.0, 6000.0))

    style = a["style"]
    sm = cfg["smoke"]
    w_world = w_px * scale                                                # target width in metres at the smoke
    if style == "curtain":
        Wd = max(4.0, 0.5 * w_world / 1.4)
        height_m = ph * 0.45
        lean = 0.0
        wind = float(rng.uniform(0, 360))
    else:
        # drift across the view: ~half the target width comes from the plume leaning with the wind
        # (columns lean little, wide plumes lean a lot); wind is mostly perpendicular to the camera heading
        drift_w = w_world * (0.15 if style == "column" else 0.55)
        lean = float(np.clip(drift_w / max(ph, 1.0), 0.0, 3.0)) * float(rng.uniform(0.8, 1.2))
        Wd = max(3.0, 0.5 * (w_world - drift_w) * 0.7)
        height_m = ph
        side = 1.0 if rng.random() < 0.5 else -1.0
        wind = float((heading + side * rng.uniform(50.0, 130.0)) % 360.0)
    smoke = SmokeParams(
        style=style, anchor_n=an, anchor_e=ae,
        height_m=height_m,
        spread_m=Wd,
        wind_az_deg=wind, lean=lean,
        opacity=_u(rng, a["opacity"]), puff_alpha=_u(rng, sm["puff_alpha"]),
        n_puffs=int(rng.integers(sm["puffs"][style][0], sm["puffs"][style][1] + 1)),
        color_mode=_pick(rng, sm["color_modes"]), shape_seed=int(rng.integers(0, 2**31 - 1)),
        target_box_xyxy=(float(cx - w_px / 2), float(base_v - h_px if style != "curtain" else cy - h_px / 2),
                         float(cx + w_px / 2), float(base_v if style != "curtain" else cy + h_px / 2)),
        line_len_m=0.0,
    )
    p["smoke"] = smoke
    return SceneParams(**p)
