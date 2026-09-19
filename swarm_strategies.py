#!/usr/bin/env python3
"""
swarm_strategies.py — pluggable velocity policies for your experiment stack

✅ Works with your current swarm_agent.py
   • All policies accept **kwargs and ignore unknown keys (lat0/lon0/etc.).
   • Includes Random / Expanding / Lawnmower / Hold.
   • Optional FrontierGridPolicy for non-mission demos (your agent's GridMission is still the main one).

Return signature: (vn, ve, vz, mode_str)
  - vn, ve, vz are NED velocities (m/s) in **local ENU** sign for vn/ve (north/east), vz up is negative in NED
  - mode_str is a short HUD label (e.g., "PAT:lawnmower").

Usage from swarm_agent.py
  policy = strat.make_policy(policy_name, alt_target=..., vxy_max=..., vz_max=..., lat0=..., lon0=...)
  vn, ve, vz, mode = policy.step(mine_state_dict, neighbors)

Note: When --mission grid_frontier is used, your Agent uses GridMission; this module's
FrontierGridPolicy is only for non-grid mission demos.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math, random, time

# --- small utils ---
R_EARTH = 6378137.0

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def latlon_to_ne(lat_deg: float, lon_deg: float, lat0_deg: float, lon0_deg: float) -> Tuple[float, float]:
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg); lon0 = math.radians(lon0_deg)
    dn = (lat - lat0) * R_EARTH
    de = (lon - lon0) * R_EARTH * math.cos(lat0)
    return dn, de

# --- base class ---
class BasePolicy:
    name = "base"
    def step(self, mine: Dict, neigh: List[Dict]) -> Tuple[float,float,float,str]:
        raise NotImplementedError

# --- simple policies ---
class RandomPolicy(BasePolicy):
    name = "random"
    def __init__(self, vxy_max: float = 6.0, vz_max: float = 1.5, alt_target: float = 20.0, **kwargs):
        self.vxy_max = vxy_max; self.vz_max = vz_max; self.alt_target = alt_target
        self._t = time.time()
    def step(self, mine, neigh):
        # gentle random walk with slight smoothing
        rng = 1.5
        vn = random.uniform(-rng, rng)
        ve = random.uniform(-rng, rng)
        alt = (mine.get("alt") or self.alt_target)
        vz = clamp(-0.5 * (self.alt_target - alt), -self.vz_max, self.vz_max)
        return vn, ve, vz, "SEARCH:random"

class ExpandingPolicy(BasePolicy):
    name = "expanding"
    def __init__(self, speed: float = 6.0, leg0: float = 20.0, **kwargs):
        self.speed = speed; self.leg = leg0; self.t0 = time.time(); self.k = 0
    def step(self, mine, neigh):
        t = time.time() - self.t0
        leg_time = max(1e-3, self.leg / self.speed)
        which = int(t / leg_time)
        if which != self.k:
            self.k = which
            if which % 2 == 1:
                self.leg += 10.0  # increase every second segment
        dir4 = which % 4  # 0:E,1:N,2:W,3:S in EN frame (vn,ve)
        if dir4 == 0:   vn, ve = 0.0, +self.speed
        elif dir4 == 1: vn, ve = +self.speed, 0.0
        elif dir4 == 2: vn, ve = 0.0, -self.speed
        else:           vn, ve = -self.speed, 0.0
        vz = 0.0
        return vn, ve, vz, "PAT:expanding"

class LawnmowerPolicy(BasePolicy):
    name = "lawnmower"
    def __init__(self, speed: float = 6.0, lane_len: float = 60.0, lane_sep: float = 20.0, **kwargs):
        self.speed = speed; self.L = lane_len; self.W = lane_sep; self.t0 = time.time()
    def step(self, mine, neigh):
        t = time.time() - self.t0
        seg_time = max(1e-3, self.L / self.speed)
        k = int(t / seg_time)
        dir_north = 1 if (k % 2 == 0) else -1
        phase = (t % seg_time) / seg_time
        if phase < 0.9:
            vn, ve = dir_north * self.speed, 0.0
        else:
            # quick sidestep to next lane over the last 10% of the segment
            vn, ve = 0.0, self.W / (0.1 * seg_time)
        vz = 0.0
        return vn, ve, vz, "PAT:lawnmower"

class HoldPolicy(BasePolicy):
    name = "hold"
    def __init__(self, **kwargs):
        pass
    def step(self, mine, neigh):
        return 0.0, 0.0, 0.0, "HOLD"

# --- lightweight local frontier (optional demo only) ---
@dataclass
class GridCfg:
    lat0: float
    lon0: float
    cells: int = 100
    cell_m: float = 10.0

class FrontierGridPolicy(BasePolicy):
    """Minimal local grid to bias toward unseen cells (demo policy).
    Your main experiment should still use Agent's GridMission when --mission grid_frontier.
    """
    name = "frontier_grid"
    def __init__(self, cfg: GridCfg, vxy_max: float = 6.0, vz_max: float = 1.5, alt_target: float = 20.0, **kwargs):
        self.cfg = cfg
        N = cfg.cells
        self.seen = [[False]*N for _ in range(N)]
        self.vxy_max = vxy_max; self.vz_max = vz_max; self.alt_target = alt_target
        self._target: Optional[Tuple[int,int]] = None
    def _pos_to_cell(self, lat, lon) -> Tuple[int,int]:
        dn, de = latlon_to_ne(lat, lon, self.cfg.lat0, self.cfg.lon0)
        half = self.cfg.cells * self.cfg.cell_m / 2.0
        gx = int((de + half) / self.cfg.cell_m)
        gy = int((dn + half) / self.cfg.cell_m)
        return gx, gy
    def _cell_center_ne(self, i, j) -> Tuple[float,float]:
        half = self.cfg.cells * self.cfg.cell_m / 2.0
        en = -half + j*self.cfg.cell_m + 0.5*self.cfg.cell_m
        ee = -half + i*self.cfg.cell_m + 0.5*self.cfg.cell_m
        return en, ee
    def _nearest_frontier(self, ci, cj) -> Optional[Tuple[int,int]]:
        N = self.cfg.cells
        for r in range(N):
            i0 = max(0, ci-r); i1 = min(N-1, ci+r)
            j0 = max(0, cj-r); j1 = min(N-1, cj+r)
            for i in range(i0, i1+1):
                for j in (j0, j1):
                    if 0 <= i < N and 0 <= j < N and not self.seen[i][j]:
                        return (i,j)
            for j in range(j0+1, j1):
                for i in (i0, i1):
                    if 0 <= i < N and 0 <= j < N and not self.seen[i][j]:
                        return (i,j)
        return None
    def step(self, mine, neigh):
        lat, lon = mine.get("lat"), mine.get("lon")
        if lat is None or lon is None:
            # GPS not ready yet → gentle drift
            rng = 1.0
            return random.uniform(-rng,rng), random.uniform(-rng,rng), 0.0, "SEARCH:frontier(init)"
        ci, cj = self._pos_to_cell(lat, lon)
        N = self.cfg.cells
        if 0 <= ci < N and 0 <= cj < N:
            self.seen[ci][cj] = True
        if self._target is None or self.seen[self._target[0]][self._target[1]]:
            self._target = self._nearest_frontier(ci, cj)
        if self._target is None:
            return 0.0, 0.0, 0.0, "SEARCH:frontier(done)"
        tn, te = self._cell_center_ne(*self._target)
        me_n, me_e = latlon_to_ne(lat, lon, self.cfg.lat0, self.cfg.lon0)
        dn, de = tn - me_n, te - me_e
        d = math.hypot(dn, de) + 1e-6
        vn = (dn / d) * min(self.vxy_max, d)
        ve = (de / d) * min(self.vxy_max, d)
        alt = (mine.get("alt") or self.alt_target)
        vz = clamp(-0.4 * (self.alt_target - alt), -self.vz_max, self.vz_max)
        return vn, ve, vz, "SEARCH:frontier_grid"

# --- factory ---
def make_policy(name: str, **kwargs) -> BasePolicy:
    name = (name or "random").lower()
    if name in ("hold",):
        return HoldPolicy(**kwargs)
    if name in ("random", "rw"):
        return RandomPolicy(**kwargs)
    if name in ("expanding", "spiral"):
        return ExpandingPolicy(**kwargs)
    if name in ("lawnmower", "lawn"):
        return LawnmowerPolicy(**kwargs)
    if name in ("frontier", "frontier_grid", "grid"):
        # build a minimal local grid from kwargs if provided, else defaults
        cfg = kwargs.get("grid_cfg") or GridCfg(
            lat0=kwargs.get("lat0", 0.0), lon0=kwargs.get("lon0", 0.0),
            cells=int(kwargs.get("cells", 100)), cell_m=float(kwargs.get("cell_m", 10.0))
        )
        return FrontierGridPolicy(cfg, vxy_max=kwargs.get("vxy_max",6.0),
                                  vz_max=kwargs.get("vz_max",1.5), alt_target=kwargs.get("alt_target",20.0))
    # default
    return RandomPolicy(**kwargs)
