#!/usr/bin/env python3
"""
sim/frame_synth.py — the virtual UAV camera: (pose, smoke sources) -> a BGR frame.

This is the rendering half of the Level 3 loop. It sits on the SIMULATION side of the truth boundary:

    fog (owns the fire) --SmokeSource list--> FrameSynth.render(pose) --pixels--> (service) --> agent --> YOLO

Nothing here imports fog, FireSim or an agent, and nothing it returns carries truth: a frame is an image and
nothing else. `sources_from_fire_cells()` is the one adapter from fire cells to sources and is called only by
the fog-side camera service (sim/camera_service.py).

It reuses the 2.5D renderer in synthgen/ UNCHANGED (frozen with synth_smoke_v1): a subclass overrides the
renderer's `_smoke` hook so that smoke is a set of puff plumes placed at WORLD ground positions and projected
through the same sim/camera.py pinhole the dataset generator and the detector's box->cell step use. Ground
texture, sky, haze, exposure and grading come from the same sampler and the TEST-split texture/sprite banks
(never seen in training), seeded by the scenario seed, so two models in the same scenario see identical pixels
for identical poses.

Limitations, to be repeated in any write-up: smoke is a stack of flat billboard puffs, not a volume; flat
ground; one pinhole, no lens distortion; per-source plume geometry (height, spread, count) is a simulation
parameter with no empirical basis; a nadir camera is outside the synth_smoke_v1 training pose distribution
(training pitch is 28-89 deg with most mass oblique; see docs/handoffs/ardupilot_integration_plan.md).
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mission.geo import latlon_to_ne
from sim.camera import Camera, CameraIntrinsics, CameraPose, GridSpec
from synthgen.render import Renderer
from synthgen.scene_params import SmokeParams, load_config, sample_scene

DEFAULT_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "synthgen", "configs", "domain_rand.yaml")


@dataclass(frozen=True)
class CameraSpec:
    """Documented fixed camera. Default: oblique, 60 deg below horizontal, 78 deg HORIZONTAL fov, 1280x720 (the dataset frame).
    Look direction = vehicle heading (the paired configs set perception.camera.yaw_to_track so heading follows travel)."""
    width: int = 1280
    height: int = 720
    hfov_deg: float = 78.0
    pitch_deg: float = 60.0          # degrees BELOW the horizontal: 0 = horizon, 90 = nadir. Level 3 default: oblique 60.


@dataclass(frozen=True)
class SmokeSource:
    """A smoke plume rooted at a ground point, in world metres relative to the experiment origin (n, e)."""
    n: float
    e: float
    height_m: float = 80.0
    spread_m: float = 20.0           # cross-section half-width scale (synthgen's `spread_m`)
    opacity: float = 0.8
    seed: int = 0


@dataclass(frozen=True)
class PlumeModel:
    """Simulation parameters for what a burning cell emits. These are NOT measured or fitted to anything."""
    height_m: float = 80.0
    spread_m: float = 20.0
    n_puffs: int = 260
    puff_alpha: float = 0.22
    lean: float = 0.4                # along-wind drift of the plume top / height
    wind_az_deg: float = 45.0
    color_mode: str = "white"
    jitter_cell_frac: float = 0.25   # deterministic placement jitter inside a cell, as a fraction of cell size


def _stable_seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def sources_from_fire_cells(cells: Iterable[Tuple[int, int, float]], grid: GridSpec, model: PlumeModel,
                            scenario_seed: int) -> List[SmokeSource]:
    """One plume per burning cell, (i, j, heat) -> SmokeSource. i = east index, j = north index (fog/state.py).
    Placement jitter and shape seed depend only on (scenario seed, i, j), so a cell's plume is the same in every
    frame and for every model."""
    out: List[SmokeSource] = []
    for i, j, heat in cells:
        n, e = grid.center_ne(int(i), int(j))
        rng = np.random.default_rng(_stable_seed("src", scenario_seed, i, j))
        jn, je = (rng.uniform(-1, 1, 2) * model.jitter_cell_frac * grid.cell_m)
        out.append(SmokeSource(n=n + float(jn), e=e + float(je), height_m=model.height_m, spread_m=model.spread_m,
                               opacity=float(np.clip(0.45 + 0.55 * float(heat), 0.3, 1.0)),
                               seed=_stable_seed("shape", scenario_seed, i, j)))
    return out


class _WorldRenderer(Renderer):
    """synthgen's Renderer with the single-plume `_smoke` hook replaced by world-placed plumes."""

    def __init__(self, cfg: Dict[str, Any], base_seed: int, plume: PlumeModel):
        super().__init__(cfg, base_seed)
        self.plume = plume
        self._sources: Sequence[SmokeSource] = ()
        self._uav_ne: Tuple[float, float] = (0.0, 0.0)
        self.last_visible_sources = 0

    def set_frame(self, sources: Sequence[SmokeSource], uav_ne: Tuple[float, float]) -> None:
        self._sources, self._uav_ne = sources, uav_ne

    def _smoke(self, sp, hor, sun_el, exposure, gate=None):
        W, H = sp.width, sp.height
        rs = float(self.cfg["smoke"].get("render_scale", 0.5))
        Ws, Hs = max(8, int(round(W * rs))), max(8, int(round(H * rs)))
        cam = Camera(CameraIntrinsics(Ws, Hs, sp.hfov_deg),
                     CameraPose(0.0, 0.0, sp.cam_alt_m, sp.cam_heading_deg, sp.cam_pitch_deg))
        C = np.zeros((Hs, Ws, 3), np.float32)
        A = np.zeros((Hs, Ws), np.float32)
        un, ue = self._uav_ne
        m = self.plume
        layers = []
        for s in self._sources:
            rel_n, rel_e = s.n - un, s.e - ue
            smk = SmokeParams(style="plume", anchor_n=rel_n, anchor_e=rel_e, height_m=s.height_m,
                              spread_m=s.spread_m, wind_az_deg=m.wind_az_deg, lean=m.lean, opacity=s.opacity,
                              puff_alpha=m.puff_alpha, n_puffs=m.n_puffs, color_mode=m.color_mode,
                              shape_seed=int(s.seed), target_box_xyxy=(0.0, 0.0, 1.0, 1.0))
            u = self._plume_unit(smk, sp)
            Hp = s.height_m
            pos, rad = self._plume_world(u, smk, s.spread_m, Hp, smk.lean * Hp)
            uv, depth = cam.project(pos)
            px_r = np.where(depth > 1.0, cam.intr.fx * rad / np.maximum(depth, 1.0), 0.0)
            vis = (np.isfinite(uv[:, 0]) & (px_r > 1.2) & (uv[:, 0] > -px_r) & (uv[:, 0] < Ws + px_r)
                   & (uv[:, 1] > -px_r) & (uv[:, 1] < Hs + px_r))
            if not vis.any():
                continue
            layers.append((float(np.min(depth[vis])), smk, u, pos, rad))
        self.last_visible_sources = len(layers)
        for _, smk, u, pos, rad in sorted(layers, key=lambda x: -x[0]):      # far plumes first
            Ci, Ai = self._composite(sp, smk, u, pos, rad, hor, sun_el, exposure, rs)
            C = C * (1.0 - Ai[..., None]) + Ci
            A = A * (1.0 - Ai) + Ai
        if C.shape[:2] != (H, W):
            C = cv2.resize(C, (W, H), interpolation=cv2.INTER_LINEAR)
            A = cv2.resize(A, (W, H), interpolation=cv2.INTER_LINEAR)
        return C, A, {"visible_sources": len(layers)}


class FrameSynth:
    """Renders what a camera at `pose` sees. Deterministic: (scenario_seed, pose, sources, frame_key) -> pixels."""

    def __init__(self, scenario_seed: int, lat0: float, lon0: float, camera: CameraSpec = CameraSpec(),
                 plume: PlumeModel = PlumeModel(), cfg_path: str = DEFAULT_CFG_PATH, split: str = "test",
                 base_seed: int = 0):
        self.scenario_seed = int(scenario_seed)
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self.camera = camera
        cfg = load_config(cfg_path)
        cfg["image"] = dict(cfg["image"], width=camera.width, height=camera.height)
        self.cfg = cfg
        self.renderer = _WorldRenderer(cfg, base_seed, plume)
        # scenario-wide environment (texture, sun, haze, exposure, grade, sky): the environment fields of a negative
        # scene drawn from the scenario seed; every pose in the scenario shares it, whichever model is flying.
        self._env = sample_scene(cfg, split, self.scenario_seed, base_seed, is_positive=False)

    @property
    def last_visible_sources(self) -> int:
        return self.renderer.last_visible_sources

    def scene_for(self, n: float, e: float, alt: float, heading_deg: float) -> Any:
        """SceneParams for a camera at (n, e, alt): the shared environment with a pose-specific ground-texture
        offset so the forest is one continuous world rather than a new texture per frame."""
        env = self._env
        r = math.radians(env.tex_rot_deg)
        cr, sr = math.cos(r), math.sin(r)
        texel = float(self.cfg["background"]["texel_m"]) * env.tex_scale
        off = (env.tex_off[0] + (e * cr - n * sr) / texel, env.tex_off[1] + (e * sr + n * cr) / texel)
        return replace(env, width=self.camera.width, height=self.camera.height, hfov_deg=self.camera.hfov_deg,
                       cam_alt_m=float(alt), cam_heading_deg=float(heading_deg) % 360.0,
                       cam_pitch_deg=self.camera.pitch_deg, horizon_row_px=0.0, tex_off=off,
                       noise_sigma=0.0,                       # per-frame noise is added below
                       smoke=SmokeParams("plume", 0, 0, 1, 1, 0, 0, 0, 0, 0, "white", 0, (0, 0, 1, 1)))

    def camera_for(self, n: float, e: float, alt: float, heading_deg: float) -> Camera:
        return Camera(CameraIntrinsics(self.camera.width, self.camera.height, self.camera.hfov_deg),
                      CameraPose(n, e, float(alt), float(heading_deg) % 360.0, self.camera.pitch_deg))

    def render(self, lat: float, lon: float, alt: float, heading_deg: Optional[float],
               sources: Sequence[SmokeSource], frame_key: Any = 0) -> np.ndarray:
        """uint8 H x W x 3, BGR (Ultralytics' convention for numpy input)."""
        return self.render_frame(lat, lon, alt, heading_deg, sources, frame_key)[0]

    def render_frame(self, lat: float, lon: float, alt: float, heading_deg: Optional[float],
                     sources: Sequence[SmokeSource], frame_key: Any = 0) -> Tuple[np.ndarray, np.ndarray]:
        """(BGR frame, smoke alpha float32 HxW). The alpha is SIMULATION-SIDE truth for calibration (visibility of the
        smoke in the frame); the camera service never returns it."""
        n, e = latlon_to_ne(lat, lon, self.lat0, self.lon0)
        hdg = 0.0 if heading_deg is None else float(heading_deg)
        sp = self.scene_for(n, e, alt, hdg)
        self.renderer.set_frame(sources, (n, e))
        res = self.renderer.render(sp)
        rgb = res.image
        sigma = float(self._env.noise_sigma)
        if sigma > 1e-4:
            rng = np.random.default_rng(_stable_seed("noise", self.scenario_seed, frame_key))
            rgb = np.clip(rgb.astype(np.float32) + rng.normal(0.0, sigma * 255.0, rgb.shape), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb[..., ::-1]), res.smoke_alpha
