"""
Probabilistic wildfire spread on the Fog grid.

Fire states: 0 = unburned, 1 = burning, 2 = burnt
Spreading:   stochastic per-neighbor ignition (4- or 8-connected)
Burn time:   sampled per cell (fixed / uniform / normal) in steps of dt_seconds
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

UNBURNED, BURNING, BURNT = 0, 1, 2


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class FireCell:
    state: int = UNBURNED   # 0 = unburned, 1 = burning, 2 = burnt
    heat: float = 0.0       # [0,1] visual/intensity cue
    burn_time_left: float = 0.0  # seconds remaining in burning state
    ignited_at: Optional[float] = None


class FireSim:
    """
    Lightweight wildfire model for an N x N grid.

    Core rules (per tick):
      • Each burning cell loses `dt_seconds` of its burn timer; once ≤ 0 it becomes BURNT.
      • Each burning cell attempts to ignite neighbors with probability `p_ignite`
        (4-way von Neumann or 8-way Moore). Optional wind biases the chance.
      • Burn durations are sampled per-cell (fixed/uniform/normal) when they ignite.
    """

    def __init__(
        self,
        N: int,
        *,
        p_ignite: float = 0.30,
        neighborhood: str = "von_neumann",   # "von_neumann" (4) or "moore" (8)
        burn_steps: int = 5,                 # baseline steps a cell keeps burning
        dt_seconds: float = 1.0,             # length of one cycle
        burn_time_dist: str = "fixed",       # "fixed", "uniform", or "normal"
        burn_steps_min: Optional[int] = None,
        burn_steps_max: Optional[int] = None,
        burn_steps_sigma: float = 1.5,       # used only when burn_time_dist == "normal"
        wind_direction_deg: float = 0.0,     # 0 = north
        wind_strength: float = 0.0,          # 0–1 multiplier on downwind ignition boost
        random_seed: Optional[int] = None,
    ):
        self.N = int(N)
        self.cells: List[List[FireCell]] = [
            [FireCell() for _ in range(self.N)] for __ in range(self.N)
        ]

        self.p_ignite = float(p_ignite)
        self.neighborhood = neighborhood
        self.burn_steps = max(1, int(burn_steps))
        self.dt_seconds = max(1e-6, float(dt_seconds))
        self.burn_time_dist = burn_time_dist
        self.burn_steps_min = burn_steps_min
        self.burn_steps_max = burn_steps_max
        self.burn_steps_sigma = burn_steps_sigma
        self.wind_direction_deg = wind_direction_deg
        self.wind_strength = clamp(wind_strength, 0.0, 1.0)
        self.rng = random.Random(random_seed)

        self.time_s = 0.0

    # ---- construction helpers ----

    @classmethod
    def from_config(cls, cfg: Dict, grid_size: int) -> "FireSim":
        """
        Build a FireSim from a YAML/JSON dict with a `fire_model:` section.
        Missing fields fall back to sensible defaults.
        """
        fm = cfg.get("fire_model", {}) if isinstance(cfg, dict) else {}
        return cls(
            grid_size,
            p_ignite=fm.get("p_ignite", 0.30),
            neighborhood=fm.get("neighborhood", "von_neumann"),
            burn_steps=fm.get("burn_steps", 5),
            dt_seconds=fm.get("dt_seconds", 1.0),
            burn_time_dist=fm.get("burn_time_dist", "fixed"),
            burn_steps_min=fm.get("burn_steps_min"),
            burn_steps_max=fm.get("burn_steps_max"),
            burn_steps_sigma=fm.get("burn_steps_sigma", 1.5),
            wind_direction_deg=fm.get("wind_direction_deg", 0.0),
            wind_strength=fm.get("wind_strength", 0.0),
            random_seed=fm.get("random_seed"),
        )

    # ---- public API ----

    def reset(self):
        self.time_s = 0.0
        for i in range(self.N):
            for j in range(self.N):
                self.cells[i][j] = FireCell()

    def ignite(self, i: int, j: int):
        """Force a cell into burning state and sample a burn duration for it."""
        if not (0 <= i < self.N and 0 <= j < self.N):
            return
        c = self.cells[i][j]
        c.state = BURNING
        c.ignited_at = self.time_s
        c.burn_time_left = self._sample_burn_seconds()
        c.heat = max(c.heat, 0.5)

    def ignite_many(self, coords: Iterable[Sequence[int]]):
        for ij in coords:
            try:
                i, j = int(ij[0]), int(ij[1])
                self.ignite(i, j)
            except Exception:
                continue

    def snapshot(self) -> List[List[int]]:
        return [[self.cells[i][j].state for j in range(self.N)] for i in range(self.N)]

    def hazard_messages(self, sysid: Optional[int] = None) -> List[Dict]:
        """Emit lightweight HAZARD_UPDATE payloads for all burning cells."""
        msgs = []
        for i in range(self.N):
            for j in range(self.N):
                if self.cells[i][j].state == BURNING:
                    msgs.append(
                        {
                            "type": "HAZARD_UPDATE",
                            "sys": sysid,
                            "i": i,
                            "j": j,
                            "hazard": "fire",
                            "t": self.time_s,
                            "heat": round(self.cells[i][j].heat, 2),
                        }
                    )
        return msgs

    # ---- simulation ----

    def step(self) -> List[Tuple[int, int, int, float]]:
        """
        Advance the fire simulation by one cycle of dt_seconds.
        Returns cells that changed state or cooled/heated noticeably.
        """
        dt = self.dt_seconds
        N = self.N
        prev_state = [[self.cells[i][j].state for j in range(N)] for i in range(N)]

        to_ignite: List[Tuple[int, int]] = []

        # 1) Decide ignitions from burning neighbors (using previous state)
        for i in range(N):
            for j in range(N):
                if prev_state[i][j] != BURNING:
                    continue
                for di, dj in self._neighbor_offsets():
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < N and 0 <= nj < N):
                        continue
                    if prev_state[ni][nj] != UNBURNED:
                        continue

                    p = self.p_ignite
                    p *= 1.0 + self.wind_strength * self._wind_alignment(di, dj)
                    p_eff = clamp(p, 0.0, 1.0)
                    if self.rng.random() < p_eff:
                        to_ignite.append((ni, nj))

        changed: List[Tuple[int, int, int, float]] = []

        # 2) Update timers, apply state transitions, and newly ignite cells
        for i in range(N):
            for j in range(N):
                c = self.cells[i][j]
                before_state, before_heat = c.state, c.heat

                if c.state == BURNING:
                    c.burn_time_left -= dt
                    c.heat = clamp(c.heat + 0.1, 0.0, 1.0)
                    if c.burn_time_left <= 0.0:
                        c.state = BURNT
                        c.heat = 0.0

                if (i, j) in to_ignite and c.state == UNBURNED:
                    self.ignite(i, j)

                if c.state != before_state or abs(c.heat - before_heat) > 0.05:
                    changed.append((i, j, c.state, c.heat))

        self.time_s += dt
        return changed

    # ---- internals ----

    def _neighbor_offsets(self) -> List[Tuple[int, int]]:
        if self.neighborhood.lower().startswith("von"):
            return [(1, 0), (-1, 0), (0, 1), (0, -1)]
        return [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

    def _wind_alignment(self, di: int, dj: int) -> float:
        """
        Downwind boost only: returns [0,1] where 1 means perfectly aligned with wind.
        """
        if self.wind_strength <= 0.0:
            return 0.0
        # Convert wind bearing (deg, 0=north) into grid delta (i east, j north)
        theta = math.radians(self.wind_direction_deg)
        wi = math.sin(theta)
        wj = math.cos(theta)
        dot = di * wi + dj * wj
        mag_n = max(1e-6, math.sqrt(di * di + dj * dj))
        mag_w = max(1e-6, math.sqrt(wi * wi + wj * wj))
        return clamp(dot / (mag_n * mag_w), 0.0, 1.0)

    def _sample_burn_seconds(self) -> float:
        """Sample how long a cell should burn, in seconds."""
        base_steps = self.burn_steps
        if self.burn_time_dist == "uniform":
            lo = self.burn_steps_min or base_steps
            hi = self.burn_steps_max or max(base_steps, lo)
            steps = self.rng.randint(max(1, int(lo)), max(1, int(hi)))
        elif self.burn_time_dist == "normal":
            steps = int(round(self.rng.gauss(base_steps, self.burn_steps_sigma)))
            steps = max(1, steps)
            if self.burn_steps_min:
                steps = max(int(self.burn_steps_min), steps)
            if self.burn_steps_max:
                steps = min(int(self.burn_steps_max), steps)
        else:
            steps = max(1, base_steps)
        return steps * self.dt_seconds
