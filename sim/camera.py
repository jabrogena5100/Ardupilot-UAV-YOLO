#!/usr/bin/env python3
"""
sim/camera.py — the one pinhole projection, used twice.

    synthgen/render.py   places a smoke sprite at a ground coordinate and renders a
                         ground plane by asking "what ground point is under this pixel?"
    perception/detector  turns a YOLO pixel box back into a ground point / grid cell.

Both go through the Camera below. If the two ever disagree about where a pixel
lands, a synthetic label and a UAV detection stop describing the same place, so
do not add a second projection anywhere else (see CLAUDE.md, "2.5D").

Frames and conventions
----------------------
World: metres, (n, e, u) = (north, east, up), relative to the experiment origin,
       matching mission/geo.py. Flat ground at u = 0. Grid cells are exactly
       fog/state.py's: i = east index, j = north index, (0, 0) = south-west corner,
       the square grid is centred on the origin.
Image: continuous pixel coordinates, origin at the top-left corner of the image,
       x to the right, y down. The image spans [0, W] x [0, H]; pixel (r, c)
       has its centre at (c + 0.5, r + 0.5).
Pose:  position (n, e, alt) with alt = height above the flat ground.
       heading_deg: compass bearing of the camera's horizontal look direction
                    (0 = north, 90 = east).
       pitch_deg:   how far the optical axis points BELOW the horizon
                    (0 = horizontal, 90 = straight down / nadir; slightly negative
                    values look above the horizon). Roll is not modelled.
       This is an explicit convention because the repo's `camera: {tilt_deg: 30}`
       does not say whether 30 is measured from nadir or from the horizon. Convert
       before use, do not assume.
FOV:   `hfov_deg` is the HORIZONTAL field of view; square pixels, so the vertical
       FOV follows from the aspect ratio. `CameraIntrinsics.from_fov` also accepts a
       diagonal FOV because `fov_deg: 78` in the repo does not say which it is.
Flat-earth pinhole only: no lens distortion, no terrain relief, no earth curvature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from fog.state import cell_index          # grid indexing is owned by fog/state.py
from mission.geo import M_PER_MILE, latlon_to_ne

_EPS = 1e-9


# ----------------------------------------------------------------------
# Intrinsics / pose / grid
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class CameraIntrinsics:
    width: int = 1280
    height: int = 720
    hfov_deg: float = 78.0          # horizontal field of view

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def fy(self) -> float:           # square pixels
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.fy))

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float,
                 fov_kind: str = "horizontal") -> "CameraIntrinsics":
        """Build intrinsics from a single FOV figure. fov_kind: "horizontal" | "diagonal"."""
        if fov_kind == "horizontal":
            return cls(width, height, fov_deg)
        if fov_kind == "diagonal":
            diag = math.hypot(width, height)
            f = (diag / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
            hfov = math.degrees(2.0 * math.atan((width / 2.0) / f))
            return cls(width, height, hfov)
        raise ValueError(f"fov_kind must be 'horizontal' or 'diagonal', got {fov_kind!r}")


@dataclass(frozen=True)
class CameraPose:
    n: float
    e: float
    alt: float                       # metres above the flat ground plane (must be > 0)
    heading_deg: float = 0.0         # compass bearing of the look direction
    pitch_deg: float = 30.0          # optical axis below the horizon (0 horizon, 90 nadir, <0 above)


@dataclass(frozen=True)
class GridSpec:
    """The experiment grid, matching fog/state.py (square, centred on the origin)."""
    n_cells: int = 10
    cell_m: float = 160.9344         # 1 mile / 10 cells, the default experiment grid

    @classmethod
    def from_miles(cls, miles: float, n_cells: int) -> "GridSpec":
        return cls(n_cells=n_cells, cell_m=miles * M_PER_MILE / n_cells)

    @property
    def half_side_m(self) -> float:
        return self.n_cells * self.cell_m / 2.0

    def cell_of(self, n: float, e: float) -> Optional[Tuple[int, int]]:
        """(i, j) containing the point, or None if outside the grid (fog/state.cell_index)."""
        return cell_index(n, e, self.half_side_m, self.cell_m, self.n_cells)

    def center_ne(self, i: int, j: int) -> Tuple[float, float]:
        """(n, e) of a cell's centre; the inverse of `cell_of`."""
        half = self.half_side_m
        return (-half + (j + 0.5) * self.cell_m, -half + (i + 0.5) * self.cell_m)


# ----------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------

class Camera:
    def __init__(self, intrinsics: CameraIntrinsics, pose: CameraPose):
        if not pose.alt > 0:
            raise ValueError(f"camera altitude must be > 0 (got {pose.alt})")
        if not -45.0 <= pose.pitch_deg <= 90.0:
            raise ValueError(f"pitch_deg must be in [-45, 90] (got {pose.pitch_deg})")
        self.intr = intrinsics
        self.pose = pose
        h = math.radians(pose.heading_deg)
        p = math.radians(pose.pitch_deg)
        # orthonormal camera basis in (n, e, u)
        self.forward = np.array([math.cos(p) * math.cos(h), math.cos(p) * math.sin(h), -math.sin(p)])
        self.right = np.array([-math.sin(h), math.cos(h), 0.0])
        self.up = np.array([math.sin(p) * math.cos(h), math.sin(p) * math.sin(h), math.cos(p)])
        self.position = np.array([pose.n, pose.e, pose.alt])

    @classmethod
    def from_pose(cls, pose, intrinsics: CameraIntrinsics, lat0: float, lon0: float,
                  pitch_deg: float, default_heading_deg: float = 0.0) -> "Camera":
        """Build from perception.base.Pose (lat/lon/alt/heading) and the experiment origin."""
        if pose.lat is None or pose.lon is None or pose.alt is None:
            raise ValueError("Pose needs lat, lon and alt to build a camera")
        n, e = latlon_to_ne(pose.lat, pose.lon, lat0, lon0)
        hdg = pose.heading_deg if pose.heading_deg is not None else default_heading_deg
        return cls(intrinsics, CameraPose(n, e, float(pose.alt), float(hdg), pitch_deg))

    # ---------------- world -> image ----------------

    def project(self, points_neu) -> Tuple[np.ndarray, np.ndarray]:
        """(..., 3) world points (n, e, u) -> (uv (..., 2) pixels, depth (...)).

        depth is the distance along the optical axis; points with depth <= 0 are
        behind the camera and their uv is NaN. Points may lie outside the image."""
        pts = np.asarray(points_neu, dtype=float)
        d = pts - self.position
        xc = d @ self.right
        yc = d @ self.up
        zc = d @ self.forward
        ok = zc > _EPS
        safe = np.where(ok, zc, 1.0)
        u = self.intr.cx + self.intr.fx * xc / safe
        v = self.intr.cy - self.intr.fy * yc / safe
        uv = np.stack([np.where(ok, u, np.nan), np.where(ok, v, np.nan)], axis=-1)
        return uv, zc

    def project_ground(self, n, e, ground_u: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Project ground-plane points given as separate n, e arrays."""
        n = np.asarray(n, dtype=float)
        e = np.asarray(e, dtype=float)
        pts = np.stack([n, e, np.full_like(n, ground_u)], axis=-1)
        return self.project(pts)

    # ---------------- image -> world ----------------

    def pixel_rays(self, u, v) -> np.ndarray:
        """Unnormalised ray directions (..., 3) in (n, e, u) for pixel coordinates."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        x = (u - self.intr.cx) / self.intr.fx
        y = (v - self.intr.cy) / self.intr.fy
        return (self.forward + x[..., None] * self.right - y[..., None] * self.up)

    def unproject_ground(self, u, v, ground_u: float = 0.0):
        """Pixels -> ground points. Returns (n, e, valid).

        valid is False for rays that do not hit the ground plane (at or above the
        horizon); n and e are NaN there."""
        d = self.pixel_rays(u, v)
        du = d[..., 2]
        valid = du < -_EPS
        t = np.where(valid, (self.pose.alt - ground_u) / np.where(valid, -du, 1.0), np.nan)
        n = self.position[0] + t * d[..., 0]
        e = self.position[1] + t * d[..., 1]
        return n, e, valid

    def ground_grid(self, width: Optional[int] = None, height: Optional[int] = None):
        """Ground (n, e) under every pixel centre, shape (H, W), plus a validity mask.

        This is what the renderer samples its ground texture with. Pixels at or above
        the horizon are invalid (the renderer fills them with sky)."""
        w = int(width or self.intr.width)
        h = int(height or self.intr.height)
        if (w, h) != (self.intr.width, self.intr.height):
            raise ValueError("ground_grid size must match the intrinsics; build a Camera for that size")
        uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        return self.unproject_ground(uu, vv)

    def horizon_row(self) -> float:
        """Image row of the horizon line (may lie outside the image; -inf at nadir)."""
        if self.pose.pitch_deg >= 90.0 - 1e-9:
            return -math.inf
        return self.intr.cy - self.intr.fy * math.tan(math.radians(self.pose.pitch_deg))

    # ---------------- footprint / cells ----------------

    def ground_or_far(self, u, v, max_range_m: float = 2000.0):
        """Like unproject_ground, but rays above the horizon, or hitting the ground
        farther than max_range_m, are placed at max_range_m horizontally along the
        ray. Returns (n, e, far)."""
        d = self.pixel_rays(u, v)
        n, e, valid = self.unproject_ground(u, v)
        horiz = np.hypot(d[..., 0], d[..., 1])
        horiz = np.where(horiz < _EPS, 1.0, horiz)
        fn = self.position[0] + max_range_m * d[..., 0] / horiz
        fe = self.position[1] + max_range_m * d[..., 1] / horiz
        dist = np.hypot(n - self.position[0], e - self.position[1])
        far = (~valid) | (np.nan_to_num(dist, nan=np.inf) > max_range_m)
        return np.where(far, fn, n), np.where(far, fe, e), far

    def footprint(self, max_range_m: float = 2000.0, samples_per_edge: int = 8):
        """Ground polygon seen by the image, as an (K, 2) array of (n, e), perimeter order.

        Parts of the image above the horizon are cut off at max_range_m. Returns
        (polygon, clipped) where clipped says some of the image looked past the range."""
        W, H = self.intr.width, self.intr.height
        s = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
        top = np.stack([s * W, np.zeros_like(s)], axis=-1)
        right = np.stack([np.full_like(s, W), s * H], axis=-1)
        bottom = np.stack([W - s * W, np.full_like(s, H)], axis=-1)
        left = np.stack([np.zeros_like(s), H - s * H], axis=-1)
        edge = np.concatenate([top, right, bottom, left], axis=0)
        n, e, far = self.ground_or_far(edge[:, 0], edge[:, 1], max_range_m)
        return np.stack([n, e], axis=-1), bool(far.any())

    def cells_in_footprint(self, grid: GridSpec, max_range_m: float = 2000.0) -> List[Tuple[int, int]]:
        """Grid cells whose centre lies inside the image footprint, sorted by (i, j)."""
        poly, _ = self.footprint(max_range_m)
        n_lo, e_lo = poly.min(axis=0)
        n_hi, e_hi = poly.max(axis=0)
        out: List[Tuple[int, int]] = []
        for j in range(grid.n_cells):
            cn = -grid.half_side_m + (j + 0.5) * grid.cell_m
            if cn < n_lo - grid.cell_m or cn > n_hi + grid.cell_m:
                continue
            for i in range(grid.n_cells):
                ce = -grid.half_side_m + (i + 0.5) * grid.cell_m
                if ce < e_lo - grid.cell_m or ce > e_hi + grid.cell_m:
                    continue
                if _point_in_polygon(cn, ce, poly):
                    out.append((i, j))
        return out

    def bbox_ground_anchor(self, bbox_xyxy: Sequence[float]):
        """Ground point (n, e) under the bottom-centre of a pixel box, or None.

        For a smoke plume the base of the box is where it meets the ground, so this
        is the point the planner should treat as the smoke's location."""
        x1, y1, x2, y2 = bbox_xyxy
        n, e, valid = self.unproject_ground(np.array(0.5 * (x1 + x2)), np.array(float(y2)))
        if not bool(valid):
            return None
        return float(n), float(e)

    def bbox_to_cell(self, bbox_xyxy: Sequence[float], grid: GridSpec) -> Optional[Tuple[int, int]]:
        pt = self.bbox_ground_anchor(bbox_xyxy)
        return None if pt is None else grid.cell_of(*pt)


def _point_in_polygon(n: float, e: float, poly: np.ndarray) -> bool:
    """Even-odd rule in the (e, n) plane; poly is (K, 2) of (n, e)."""
    inside = False
    k = len(poly)
    for a in range(k):
        n1, e1 = poly[a]
        n2, e2 = poly[(a + 1) % k]
        if (n1 > n) != (n2 > n):
            e_cross = e1 + (n - n1) * (e2 - e1) / (n2 - n1)
            if e < e_cross:
                inside = not inside
    return inside
