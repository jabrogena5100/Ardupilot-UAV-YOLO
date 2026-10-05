#!/usr/bin/env python3
"""
synthgen/render.py — 2.5D renderer: procedural ground plane + sky + billboard smoke.

Everything geometric goes through sim/camera.py:

  * the ground is rendered by inverse mapping: for each pixel, Camera.ground_grid
    says which ground point is under it, and the ground texture is sampled there;
  * the smoke is a cloud of 3D "puff" centres projected with Camera.project, each
    drawn as a camera-facing sprite whose pixel radius is the pinhole radius
    fx * r / depth.

Known limitations (state them in any write-up): smoke is a set of flat billboards,
not a true volume; there is no 3D occlusion between smoke and terrain (the ground
is a flat plane; trees are texture, not geometry); no lens distortion; no
physically based light transport. The renderer is deliberately simple and cheap,
and its weakness is a threat to the Synth->Real result that must be reported.

No real imagery is read anywhere in this module; every pixel is procedural.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from synthgen.scene_params import SceneParams, bank_rng

# ----------------------------------------------------------------------
# small numeric helpers
# ----------------------------------------------------------------------

def smoothstep(a: float, b: float, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fft_noise(rng: np.random.Generator, size: int, beta: float) -> np.ndarray:
    """Tileable noise with a 1/f^beta amplitude spectrum, normalised to zero mean / unit std."""
    white = rng.standard_normal((size, size)).astype(np.float32)
    spec = np.fft.rfft2(white)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    out = np.fft.irfft2(spec * (f ** (-beta)), s=(size, size)).astype(np.float32)
    out -= out.mean()
    out /= (out.std() + 1e-8)
    return out


def _conv_fft(field: np.ndarray, sigma: float) -> np.ndarray:
    """Circular (tileable) Gaussian blur via FFT."""
    size = field.shape[0]
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    g = np.exp(-2.0 * (math.pi ** 2) * (sigma ** 2) * (fx * fx + fy * fy))
    return np.fft.irfft2(np.fft.rfft2(field) * g, s=(size, size)).astype(np.float32)


def _lerp(a, b, t):
    return a * (1.0 - t) + b * t


# ----------------------------------------------------------------------
# ground texture bank (procedural, tileable, built once per split)
# ----------------------------------------------------------------------

BANK_PATTERN = ["canopy", "mixed", "clearcut", "canopy", "canopy", "mixed", "canopy", "clearcut"]


def bank_kinds(size: int) -> List[str]:
    return [BANK_PATTERN[i % len(BANK_PATTERN)] for i in range(size)]


def build_ground_texture(rng: np.random.Generator, kind: str, size: int, lake_fraction: float) -> List[np.ndarray]:
    """RGBA uint8 mip pyramid. RGB = sun-lit albedo with baked relief shading, A = water mask."""
    forest_frac = {"canopy": 1.0, "mixed": 0.6, "clearcut": 0.22}[kind]

    # --- conifer crowns: sparse impulses blurred at two scales (tileable by FFT convolution)
    imp1 = (rng.random((size, size)) < 0.014).astype(np.float32) * rng.uniform(0.6, 1.0, (size, size)).astype(np.float32)
    imp2 = (rng.random((size, size)) < 0.0035).astype(np.float32) * rng.uniform(0.7, 1.0, (size, size)).astype(np.float32)
    height = _conv_fft(imp1, 2.0) * 1.0 + _conv_fft(imp2, 4.2) * 4.5 + 0.18 * fft_noise(rng, size, 1.0)
    height = (height - height.mean()) / (height.std() + 1e-8)
    gx = np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)
    gy = np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)
    shade = np.clip(1.0 + 0.55 * (gx + gy) / (np.std(gx + gy) + 1e-8), 0.30, 1.9)
    ao = np.clip(0.30 + 0.55 * smoothstep(-1.0, 0.9, height), 0.18, 1.2)        # dark gaps between crowns

    patch = smoothstep(-0.4, 0.6, fft_noise(rng, size, 1.3))               # light (deciduous) patches
    dark = np.array([0.030, 0.085, 0.040], np.float32)
    mid = np.array([0.075, 0.170, 0.070], np.float32)
    light = np.array([0.170, 0.300, 0.110], np.float32)
    canopy = _lerp(dark[None, None, :], mid[None, None, :], np.clip(0.5 + 0.25 * height, 0, 1)[..., None])
    canopy = _lerp(canopy, light[None, None, :], (0.55 * patch * np.clip(0.4 + 0.3 * height, 0, 1))[..., None])
    canopy = canopy * (shade * ao)[..., None]

    # --- open ground (clearcut / mossy): brown-olive with mottling and sparse debris
    mott = fft_noise(rng, size, 1.2)
    brown = np.array([0.20, 0.175, 0.105], np.float32)
    moss = np.array([0.15, 0.215, 0.095], np.float32)
    ground = _lerp(brown[None, None, :], moss[None, None, :], smoothstep(-0.3, 0.7, mott)[..., None])
    speck = _conv_fft((rng.random((size, size)) < 0.01).astype(np.float32), 0.9) * 6.0
    ground = ground * (0.85 + 0.10 * fft_noise(rng, size, 0.3)[..., None]) * (1.0 - 0.35 * np.clip(speck, 0, 1))[..., None]

    # forest vs open mask: forest_frac of the area is forest
    fm = fft_noise(rng, size, 1.6)
    thr = float(np.quantile(fm, 1.0 - forest_frac)) if forest_frac < 1.0 else -1e9
    forest = smoothstep(thr - 0.15, thr + 0.15, fm) if forest_frac < 1.0 else np.ones_like(fm)
    rgb = _lerp(ground, canopy, forest[..., None])

    # --- lakes
    water = np.zeros((size, size), np.float32)
    if lake_fraction > 0.003:
        lm = fft_noise(rng, size, 1.9)
        lthr = float(np.quantile(lm, 1.0 - lake_fraction))
        water = smoothstep(lthr, lthr + 0.12, lm)
        wcol = np.array([0.040, 0.100, 0.150], np.float32) * (1.0 + 0.15 * fft_noise(rng, size, 0.8)[..., None])
        rgb = _lerp(rgb, wcol, water[..., None])

    rgba = np.concatenate([np.clip(rgb, 0, 1), water[..., None]], axis=-1)
    level = (rgba * 255.0 + 0.5).astype(np.uint8)
    pyr = [level]
    for _ in range(4):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _sample_texture(pyr: List[np.ndarray], gn: np.ndarray, ge: np.ndarray, texel_m: float, scale: float,
                    rot_deg: float, off: Tuple[float, float], footprint_m: np.ndarray) -> np.ndarray:
    """Trilinear-ish sampling of the mip pyramid at ground points (gn, ge); returns float32 (H, W, 4)."""
    r = math.radians(rot_deg)
    cr, sr = math.cos(r), math.sin(r)
    x = ((ge * cr - gn * sr) / scale) / texel_m + off[0]
    y = ((ge * sr + gn * cr) / scale) / texel_m + off[1]
    tpp = footprint_m / (texel_m * scale)                                 # texels per pixel
    lod = np.clip(np.log2(np.maximum(tpp, 1.0)), 0.0, len(pyr) - 1.0)
    out = np.zeros(gn.shape + (4,), np.float32)
    for L, img in enumerate(pyr):
        w = np.clip(1.0 - np.abs(lod - L), 0.0, 1.0)
        if not w.any():
            continue
        s = img.shape[0]
        pad = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode="wrap")
        mx = ((x / (2 ** L)) % s + 2.0).astype(np.float32)
        my = ((y / (2 ** L)) % s + 2.0).astype(np.float32)
        smp = cv2.remap(pad, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255.0
        out += smp * w[..., None]
    return out


# ----------------------------------------------------------------------
# smoke puff sprites
# ----------------------------------------------------------------------

def make_puff_bank(rng: np.random.Generator, k: int = 24, size: int = 96) -> Tuple[np.ndarray, np.ndarray]:
    """K billowy puff sprites: alpha (k, s, s) and luminance (k, s, s). Higher contrast than a
    plain radial falloff so overlapping puffs read as cauliflower-like billows."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    xx = (xx + 0.5) / size * 2 - 1
    yy = (yy + 0.5) / size * 2 - 1
    r = np.sqrt(xx * xx + yy * yy)
    alphas = np.zeros((k, size, size), np.float32)
    lums = np.zeros((k, size, size), np.float32)
    for i in range(k):
        n = fft_noise(rng, size, 1.45)
        n2 = fft_noise(rng, size, 1.1)
        base = np.clip(1.0 - r ** 2.2, 0.0, 1.0)
        a = np.clip((base * 1.25 + 0.50 * n - 0.38) / 0.45, 0.0, 1.0)
        a = a * a * (3.0 - 2.0 * a)                                   # crisper billow edges
        a *= np.clip(1.0 - r ** 6, 0.0, 1.0)
        alphas[i] = a
        lit = 0.5 - yy * 0.5                                           # top-lit
        lums[i] = np.clip(0.80 + 0.40 * lit * (0.6 + 0.4 * np.clip(base, 0, 1)) + 0.32 * n2 - 0.18 * np.clip(base, 0, 1) ** 2, 0.45, 1.20)
    return alphas, lums


SMOKE_COLORS = {"white": (0.90, 0.91, 0.93), "grey": (0.62, 0.64, 0.68), "tan": (0.78, 0.70, 0.58)}


@dataclass
class RenderResult:
    image: np.ndarray               # uint8 RGB (H, W, 3)
    smoke_alpha: np.ndarray         # float32 (H, W) in [0, 1]; zeros for negatives
    info: Dict[str, Any] = None     # smoke calibration record (None for negatives)


class Renderer:
    def __init__(self, cfg: Dict[str, Any], base_seed: int):
        self.cfg = cfg
        self.base_seed = int(base_seed)
        self._banks: Dict[str, List[Tuple[str, List[np.ndarray]]]] = {}
        self._puffs = make_puff_bank(np.random.default_rng(np.random.SeedSequence([self.base_seed, 777])))

    # ---------------- texture bank ----------------

    def _bank(self, split: str):
        if split not in self._banks:
            bgc = self.cfg["background"]
            n = int(bgc["bank_size"][split])
            items = []
            for slot, kind in enumerate(bank_kinds(n)):
                rng = bank_rng(self.base_seed, split, slot)
                lake = float(rng.uniform(*bgc["lake_fraction"]))
                items.append((kind, build_ground_texture(rng, kind, int(bgc["texture_px"]), lake)))
            self._banks[split] = items
        return self._banks[split]

    def bank_slot(self, sp: SceneParams) -> int:
        items = self._bank(sp.split)
        same = [i for i, (k, _) in enumerate(items) if k == sp.texture_kind] or list(range(len(items)))
        return same[min(len(same) - 1, int(sp.texture_pick * len(same)))]

    # ---------------- main entry ----------------

    def render(self, sp: SceneParams) -> RenderResult:
        W, H = sp.width, sp.height
        cam = sp.camera
        bgc = self.cfg["background"]
        texel_m = float(bgc["texel_m"])

        gn, ge, ok = cam.ground_grid()
        gn = gn.astype(np.float32)
        ge = ge.astype(np.float32)
        okf = ok & np.isfinite(gn)

        # --- per-pixel ray geometry (sky + distance)
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        rays = cam.pixel_rays(uu, vv)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        elev = np.arcsin(np.clip(rays[..., 2], -1, 1))
        dist = np.where(okf, np.hypot(np.hypot(gn, ge), sp.cam_alt_m), 1e9).astype(np.float32)

        exposure = sp.exposure
        zen = np.array(sp.sky_zenith, np.float32)
        hor = np.array(sp.sky_horizon, np.float32)

        # --- sky
        t = np.clip(np.maximum(elev, 0.0) / (math.pi / 2), 0, 1) ** 0.45
        sky = _lerp(hor[None, None, :], zen[None, None, :], t[..., None])
        sun_el = math.radians(sp.sun_el_deg)
        sun_az = math.radians(sp.sun_az_deg)
        sun_dir = np.array([math.cos(sun_el) * math.cos(sun_az), math.cos(sun_el) * math.sin(sun_az), math.sin(sun_el)], np.float32)
        glow = np.clip(rays @ sun_dir, 0, 1) ** 48
        sky = sky + 0.30 * glow[..., None]
        sky = self._clouds(sky, rays, elev, sp, sun_dir, hor)

        # --- ground
        footprint = self._footprint(gn, ge, okf)
        items = self._bank(sp.split)
        slot = self.bank_slot(sp)
        _, pyr = items[slot]
        tex = _sample_texture(pyr, np.nan_to_num(gn), np.nan_to_num(ge), texel_m, sp.tex_scale, sp.tex_rot_deg,
                              sp.tex_off, footprint)
        sun_gain = 0.62 + 0.55 * math.sin(sun_el)
        ground = tex[..., :3] * (sun_gain * sp.ground_gain)
        water = tex[..., 3:4]
        if water.max() > 0.01:        # lakes mirror the sky a little
            ground = _lerp(ground, sky * 0.55 + 0.02, 0.38 * water)
        hz = (1.0 - np.exp(-sp.haze_per_m * dist))
        hz = np.where(dist > bgc["max_ground_range_m"], 1.0, hz)[..., None].astype(np.float32)
        hazecol = (hor * 0.97)[None, None, :]
        ground = _lerp(ground, hazecol, hz)

        bg = np.where(okf[..., None], ground, sky) * exposure
        # soften the horizon seam (ground ends at the horizon row)
        bg = bg.astype(np.float32)

        # --- smoke
        smoke_alpha = np.zeros((H, W), np.float32)
        info = None
        if sp.smoke is not None:
            C, A, info = self._smoke(sp, hor, sun_el, exposure)
            bg = bg * (1.0 - A[..., None]) + C
            smoke_alpha = A

        # --- camera / sensor
        img = np.clip(bg * np.array(sp.grade, np.float32)[None, None, :], 0.0, 1.0)
        if sp.blur_sigma_px > 0.05:
            img = cv2.GaussianBlur(img, (0, 0), sp.blur_sigma_px)
            smoke_alpha = cv2.GaussianBlur(smoke_alpha, (0, 0), sp.blur_sigma_px)
        if sp.noise_sigma > 1e-4:
            nrng = np.random.default_rng(np.random.SeedSequence([self.base_seed, sp.cloud_seed, 99]))
            img = np.clip(img + nrng.normal(0.0, sp.noise_sigma, img.shape).astype(np.float32), 0.0, 1.0)
        return RenderResult((img * 255.0 + 0.5).astype(np.uint8), smoke_alpha.astype(np.float32), info)

    # ---------------- pieces ----------------

    @staticmethod
    def _footprint(gn: np.ndarray, ge: np.ndarray, ok: np.ndarray) -> np.ndarray:
        """Metres on the ground per pixel step (larger of the two axes), for texture LOD."""
        n = np.where(ok, gn, 0.0)
        e = np.where(ok, ge, 0.0)
        dn_du, dn_dv = np.gradient(n, axis=1), np.gradient(n, axis=0)
        de_du, de_dv = np.gradient(e, axis=1), np.gradient(e, axis=0)
        fp = np.sqrt(np.maximum(dn_du ** 2 + de_du ** 2, dn_dv ** 2 + de_dv ** 2))
        return np.where(ok, np.clip(fp, 0.05, 5e4), 5e4).astype(np.float32)

    def _clouds(self, sky: np.ndarray, rays: np.ndarray, elev: np.ndarray, sp: SceneParams,
                sun_dir: np.ndarray, hor: np.ndarray) -> np.ndarray:
        if sp.cloud_cover < 0.03:
            return sky
        rng = np.random.default_rng(np.random.SeedSequence([self.base_seed, sp.cloud_seed]))
        size = 512
        nz = fft_noise(rng, size, 1.45)
        up = np.maximum(rays[..., 2], 0.03)
        k = 140.0 * sp.cloud_scale
        px = rays[..., 0] / up * k
        py = rays[..., 1] / up * k
        mx = (px % size).astype(np.float32)
        my = (py % size).astype(np.float32)
        pad = np.pad(nz, 2, mode="wrap")
        val = cv2.remap(pad, mx + 2.0, my + 2.0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sh = cv2.remap(pad, ((px + 6.0 * sun_dir[0]) % size).astype(np.float32) + 2.0,
                       ((py + 6.0 * sun_dir[1]) % size).astype(np.float32) + 2.0, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
        thr = float(np.quantile(nz, 1.0 - float(np.clip(sp.cloud_cover, 0.02, 0.95))))
        dens = smoothstep(thr, thr + 0.7, val) * smoothstep(0.015, 0.18, elev)
        shade = np.clip(0.93 + 0.55 * (val - sh), 0.62, 1.08)
        white = np.array([0.97, 0.97, 0.98], np.float32)[None, None, :] * shade[..., None]
        grey = np.array([0.60, 0.64, 0.70], np.float32)[None, None, :]
        col = _lerp(white, grey, (np.clip(dens, 0, 1) ** 2 * 0.28)[..., None])
        return _lerp(sky, col, np.clip(dens * 0.92, 0, 1)[..., None])

    # ---------------- smoke ----------------

    def _plume_unit(self, smk, sp: SceneParams) -> Dict[str, np.ndarray]:
        """Scale-free random plume description: drawn once, so recalibrating size does not reshuffle the puffs.
        Lengths are in units of: cross-section half-width Wd (lu, au for the curtain), puff size Rp (ru, zj, au for
        plumes), height Hp (t), drift (along-wind drift of the plume top)."""
        rng = np.random.default_rng(smk.shape_seed)
        N = smk.n_puffs
        u: Dict[str, np.ndarray] = {}
        if smk.style in ("column", "plume"):
            t = rng.random(N) ** 1.15
            rb = 0.45 if smk.style == "column" else 0.55
            u["t"] = t
            u["ru"] = (rb + (1.0 - rb) * t ** 0.8) * np.exp(0.25 * rng.standard_normal(N))
            spread = (0.35 + 0.65 * t ** 0.7) if smk.style == "column" else (0.45 + 0.55 * t ** 0.6)
            u["lu"] = np.clip(rng.normal(0, 0.42, N), -1.3, 1.3) * spread + 0.15 * np.sin(2.1 * math.pi * t * rng.uniform(0.6, 1.4) + rng.uniform(0, 6.28))
            u["au"] = rng.normal(0, 0.8, N)
            u["zj"] = rng.normal(0, 0.55, N)
            u["fade"] = 1.0 - 0.40 * t
            u["hf"] = t
        else:
            front = rng.random(N) < 0.22
            d = np.where(front, rng.random(N) ** 2.0 * 0.35, rng.random(N) ** 1.3)
            u["front"] = front
            u["t"] = np.where(front, rng.random(N) * 1.6, rng.beta(1.0, 2.6, N) * 1.1)
            u["ru"] = np.where(front, rng.uniform(0.9, 1.5, N), rng.uniform(0.6, 1.3, N)) * (0.7 + 0.5 * d)
            u["lu"] = rng.uniform(-1.0, 1.0, N) * (1.0 + 0.4 * d)
            u["au"] = d * 1.9
            u["zj"] = rng.normal(0, 0.3, N)
            u["fade"] = np.where(front, 1.0, rng.uniform(0.40, 0.85, N))
            u["hf"] = np.clip(u["t"] / 1.6, 0, 1)
        u["kb"] = rng.integers(0, self._puffs[0].shape[0], N)
        u["flip"] = rng.random(N) < 0.5
        u["pa"] = 0.6 + 0.8 * rng.random(N)
        mode = smk.color_mode
        cols = np.zeros((N, 3), np.float32)
        jit = rng.random(N)
        nz = rng.standard_normal(N)
        for i in range(N):
            if mode == "mixed":
                m = np.clip(0.2 + 0.7 * (1 - u["hf"][i]) + 0.25 * nz[i], 0, 1)
                cols[i] = _lerp(np.array(SMOKE_COLORS["white"]), np.array(SMOKE_COLORS["tan"]), m)
            elif mode == "tan":
                cols[i] = np.array(SMOKE_COLORS["tan"]) * (0.9 + 0.12 * jit[i])
            elif mode == "grey":
                cols[i] = np.array(SMOKE_COLORS["grey"]) * (0.9 + 0.2 * jit[i])
            else:
                cols[i] = np.array(SMOKE_COLORS["white"]) * (0.92 + 0.1 * jit[i])
        u["cols"] = cols
        return u

    @staticmethod
    def _plume_world(u, smk, Wd: float, Hp: float, drift: float):
        """Puff centres and radii in world metres from the scale-free description.
        Wd cross-section half-width, Hp height, drift along-wind displacement of the plume top."""
        wa = math.radians(smk.wind_az_deg)
        wind = np.array([math.cos(wa), math.sin(wa)])
        perp = np.array([-wind[1], wind[0]])
        if smk.style in ("column", "plume"):
            Rp = 0.34 * min(Wd, 1.2 * Hp)
            t = u["t"]
            along = drift * t ** 1.4 + Rp * u["au"]
            lat = Wd * u["lu"]
            z = Hp * t + Rp * u["zj"]
        else:
            Rp = 0.30 * Wd
            along = Wd * u["au"]
            lat = Wd * u["lu"]
            z = Hp * u["t"] + Rp * u["zj"]
        rad = Rp * u["ru"]
        pos = np.stack([smk.anchor_n + along * wind[0] + lat * perp[0],
                        smk.anchor_e + along * wind[1] + lat * perp[1], np.maximum(z, 0.0)], axis=-1)
        return pos, rad

    def _composite(self, sp: SceneParams, smk, u, pos, rad, hor, sun_el, exposure, scale: float):
        """Depth-sorted puff compositing at `scale` x the output resolution. Returns (premult RGB, alpha)."""
        from sim.camera import Camera, CameraIntrinsics, CameraPose
        Ws, Hs = max(8, int(round(sp.width * scale))), max(8, int(round(sp.height * scale)))
        cam = Camera(CameraIntrinsics(Ws, Hs, sp.hfov_deg),
                     CameraPose(0.0, 0.0, sp.cam_alt_m, sp.cam_heading_deg, sp.cam_pitch_deg))
        fx = cam.intr.fx
        uv, depth = cam.project(pos)
        px_r = np.where(depth > 1.0, fx * rad / np.maximum(depth, 1.0), 0.0)
        keep = np.isfinite(uv[:, 0]) & (px_r > 1.2)
        order = np.argsort(-depth)
        max_r = float(self.cfg["smoke"]["max_puff_radius_px"]) * scale

        light = (0.58 + 0.42 * math.sin(sun_el)) * (0.82 + 0.28 * u["hf"])
        dist = np.hypot(np.hypot(pos[:, 0], pos[:, 1]), sp.cam_alt_m - pos[:, 2])
        hz = (1.0 - np.exp(-sp.haze_per_m * dist))[:, None].astype(np.float32)
        cols = _lerp(u["cols"] * light[:, None], hor[None, :] * 0.98, hz * 0.8) * exposure
        gain = float(self.cfg["smoke"].get("style_alpha_gain", {}).get(smk.style, 1.0))
        pa_all = np.clip(smk.puff_alpha * smk.opacity * u["fade"] * u["pa"] * 1.7 * gain, 0.0, 0.95)

        alpha_bank, lum_bank = self._puffs
        ps = alpha_bank.shape[1]
        C = np.zeros((Hs, Ws, 3), np.float32)
        A = np.zeros((Hs, Ws), np.float32)
        for i in order:
            if not keep[i]:
                continue
            R = float(min(px_r[i], max_r))
            d = max(3, int(round(2 * R)))
            x0, y0 = int(round(float(uv[i, 0]) - R)), int(round(float(uv[i, 1]) - R))
            cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + d, Ws), min(y0 + d, Hs)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            a_t, l_t = alpha_bank[u["kb"][i]], lum_bank[u["kb"][i]]
            if u["flip"][i]:
                a_t, l_t = a_t[:, ::-1], l_t[:, ::-1]
            interp = cv2.INTER_AREA if d < ps else cv2.INTER_LINEAR
            sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
            a = cv2.resize(np.ascontiguousarray(a_t), (d, d), interpolation=interp)[sl] * float(pa_all[i])
            lum = cv2.resize(np.ascontiguousarray(l_t), (d, d), interpolation=interp)[sl]
            roiC = C[cy0:cy1, cx0:cx1]
            roiA = A[cy0:cy1, cx0:cx1]
            roiC *= (1.0 - a)[..., None]
            roiC += (cols[i][None, None, :] * (lum * a)[..., None])
            roiA *= (1.0 - a)
            roiA += a
        return C, A

    def _smoke(self, sp: SceneParams, hor: np.ndarray, sun_el: float, exposure: float):
        """Calibrate the plume size at 1/4 resolution so the rendered box matches the sampled target
        box, then render it at `smoke.render_scale` and upsample. Returns (premult RGB, alpha, info)."""
        smk = sp.smoke
        scfg = self.cfg["smoke"]
        W, H = sp.width, sp.height
        u = self._plume_unit(smk, sp)
        thr = float(self.cfg["box"]["alpha_threshold"])
        tx1, ty1, tx2, ty2 = smk.target_box_xyxy
        tw = max(8.0, min(tx2, W) - max(tx1, 0.0))
        th = max(8.0, min(ty2, H) - max(ty1, 0.0))
        S, Hp = smk.spread_m, smk.height_m
        drift = smk.lean * Hp
        cs = float(scfg.get("calib_scale", 0.25))
        history = []
        for it in range(int(scfg.get("calib_iters", 5))):
            pos, rad = self._plume_world(u, smk, S, Hp, drift)
            _, A = self._composite(sp, smk, u, pos, rad, hor, sun_el, exposure, cs)
            m = A >= thr
            if m.sum() < 6:
                S *= 1.5
                Hp *= 1.5
                drift *= 1.5
                history.append({"iter": it, "empty": True})
                continue
            rows, cols_ = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
            mw = (cols_[-1] + 1 - cols_[0]) / cs
            mh = (rows[-1] + 1 - rows[0]) / cs
            history.append({"iter": it, "mw": round(float(mw), 1), "mh": round(float(mh), 1)})
            rw, rh = tw / max(mw, 1.0), th / max(mh, 1.0)
            if abs(rw - 1) < 0.07 and abs(rh - 1) < 0.07:
                break
            fw = float(np.clip(rw, 0.4, 2.5)) ** 0.9
            S *= fw
            drift *= fw
            Hp *= float(np.clip(rh, 0.4, 2.5)) ** 0.9
        pos, rad = self._plume_world(u, smk, S, Hp, drift)
        rs = float(scfg.get("render_scale", 0.5))
        C, A = self._composite(sp, smk, u, pos, rad, hor, sun_el, exposure, rs)
        if C.shape[:2] != (H, W):
            C = cv2.resize(C, (W, H), interpolation=cv2.INTER_LINEAR)
            A = cv2.resize(A, (W, H), interpolation=cv2.INTER_LINEAR)
        return C, A, {"spread_m": round(float(S), 2), "height_m": round(float(Hp), 2), "drift_m": round(float(drift), 2), "calibration": history}
