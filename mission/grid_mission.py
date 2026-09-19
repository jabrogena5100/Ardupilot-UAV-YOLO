#!/usr/bin/env python3
"""
mission/grid_mission.py — the TRAVERSE coverage model.

Lifted out of swarm_agent.py unchanged in behaviour. This is the part of the
system that decides *where to fly*, and it is the part you will most want to
swap or subclass when the probability map starts driving targets, so it should
not be buried 400 lines inside a class that also owns sockets and a MAVLink
connection.

The TRAVERSE model
------------------
A cell is not "covered" because a drone clipped its corner. It is covered when
the drone's track inside that cell spans at least `traverse_frac` of the cell
width on one axis — an actual sweep, edge to edge.

Mechanically: for each cell, track min/max of the local east and north
coordinates of every position sample inside it. Once
max(span_e, span_n) >= traverse_frac * cell_m, the cell is done.

To produce that span the mission latches a "run": pick the axis with less
coverage so far, aim at a point 10% in from one edge, and when the drone gets
there, flip to the 90% point. Back and forth until the span threshold trips.
The flip has both a distance gate (flip_dist_m) and a time gate
(flip_min_sec) so a drone hovering near the goal cannot chatter between the
two targets every control tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .geo import M_PER_MILE, clamp, latlon_to_ne

Verbose = Optional[Callable[[str], None]]


@dataclass
class GridCfg:
    lat0: float
    lon0: float
    miles: float = 1.0            # side length of the square, in miles
    cells: int = 10               # N, giving an N x N grid
    traverse_frac: float = 0.80   # edge-to-edge threshold, as a fraction of cell width
    # Legacy fields. The TRAVERSE model ignores both; they are kept so the
    # launcher's --map-radius-m / --cell-complete-frac flags still parse.
    map_radius_m: float = 20.0
    complete_frac: float = 0.2

    @property
    def side_m(self) -> float:
        return self.miles * M_PER_MILE

    @property
    def cell_m(self) -> float:
        return self.side_m / self.cells


class GridMission:
    """Frontier selection plus edge-to-edge traversal of the chosen cell."""

    def __init__(self, cfg: GridCfg):
        self.cfg = cfg
        N = cfg.cells

        # per-cell extent of this drone's own track, in cell-local metres
        self.min_e = [[float("inf")] * N for _ in range(N)]
        self.max_e = [[float("-inf")] * N for _ in range(N)]
        self.min_n = [[float("inf")] * N for _ in range(N)]
        self.max_n = [[float("-inf")] * N for _ in range(N)]
        self.done = [[False] * N for _ in range(N)]

        self.claimed: Dict[Tuple[int, int], int] = {}   # (i,j) -> sysid that called it
        self.prev_pos: Optional[Tuple[float, float]] = None
        self.target: Optional[Tuple[int, int]] = None

        # latched traverse run: {'i','j','axis','side','goal':(n,e),'last_flip_t'}
        self.run: Optional[Dict] = None

        # tunables
        self.flip_dist_m = 2.0     # how close counts as "reached the edge goal"
        self.flip_min_sec = 1.0    # minimum dwell before flipping, anti-chatter
        self.edge_eps_m = 0.5      # boundary snapping tolerance

    # ---------------- cell geometry ----------------

    def _pos_to_cell_inclusive(self, dn: float, de: float) -> Tuple[int, int]:
        """NE metres -> (i,j), clamped, with an epsilon so a drone sitting
        exactly on a boundary lands in a cell rather than falling off the grid."""
        half = self.cfg.side_m / 2.0
        cell = self.cfg.cell_m
        dn = clamp(dn, -half - self.edge_eps_m, half + self.edge_eps_m)
        de = clamp(de, -half - self.edge_eps_m, half + self.edge_eps_m)
        u = (de + half) / cell
        v = (dn + half) / cell
        i, j = int(u), int(v)
        if abs(u - self.cfg.cells) < 1e-9:
            i = self.cfg.cells - 1
        if abs(v - self.cfg.cells) < 1e-9:
            j = self.cfg.cells - 1
        i = max(0, min(self.cfg.cells - 1, i))
        j = max(0, min(self.cfg.cells - 1, j))
        return i, j

    def pos_to_cell(self, lat: float, lon: float) -> Tuple[int, int]:
        dn, de = latlon_to_ne(lat, lon, self.cfg.lat0, self.cfg.lon0)
        return self._pos_to_cell_inclusive(dn, de)

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.cfg.cells and 0 <= j < self.cfg.cells

    def cell_center_ne(self, i: int, j: int) -> Tuple[float, float]:
        half = self.cfg.side_m / 2.0
        cn = -half + j * self.cfg.cell_m + 0.5 * self.cfg.cell_m
        ce = -half + i * self.cfg.cell_m + 0.5 * self.cfg.cell_m
        return cn, ce

    def local_east_north_in_cell(self, dn: float, de: float, i: int, j: int) -> Tuple[float, float]:
        half = self.cfg.side_m / 2.0
        e0 = -half + i * self.cfg.cell_m
        n0 = -half + j * self.cfg.cell_m
        return (clamp(de - e0, 0.0, self.cfg.cell_m),
                clamp(dn - n0, 0.0, self.cfg.cell_m))

    def cell_progress(self, i: int, j: int) -> float:
        """0..1, how far through the traverse this cell is. HUD only."""
        if not self.in_bounds(i, j):
            return 0.0
        span = max(0.0,
                   self.max_e[i][j] - self.min_e[i][j],
                   self.max_n[i][j] - self.min_n[i][j])
        return clamp(span / self.cfg.cell_m, 0.0, 1.0)

    # ---------------- progress ----------------

    def update_progress(self, lat: float, lon: float) -> None:
        dn, de = latlon_to_ne(lat, lon, self.cfg.lat0, self.cfg.lon0)
        i, j = self._pos_to_cell_inclusive(dn, de)
        if not self.in_bounds(i, j):
            self.prev_pos = (dn, de)
            return
        le, ln = self.local_east_north_in_cell(dn, de, i, j)
        self.min_e[i][j] = min(self.min_e[i][j], le)
        self.max_e[i][j] = max(self.max_e[i][j], le)
        self.min_n[i][j] = min(self.min_n[i][j], ln)
        self.max_n[i][j] = max(self.max_n[i][j], ln)
        if not self.done[i][j]:
            span = max(self.max_e[i][j] - self.min_e[i][j],
                       self.max_n[i][j] - self.min_n[i][j])
            if span >= self.cfg.traverse_frac * self.cfg.cell_m:
                self.done[i][j] = True
        self.prev_pos = (dn, de)

    # ---------------- frontier selection ----------------

    def nearest_frontier(
        self,
        i0: int, j0: int,
        allowed: Optional[Iterable[Tuple[int, int]]] = None,
        avoid: Optional[Iterable[Tuple[int, int]]] = None,
    ) -> Optional[Tuple[int, int]]:
        """Nearest not-done, not-claimed cell, searched as expanding square
        rings around (i0, j0). `allowed` is the partition mask; `avoid` is the
        union of neighbours' recent visits, merged done-cells, and hazards."""
        N = self.cfg.cells
        allowed_set = set(allowed) if allowed is not None else None
        avoid_set = set(avoid) if avoid is not None else set()

        def ok(i: int, j: int) -> bool:
            if not self.in_bounds(i, j) or self.done[i][j]:
                return False
            if allowed_set is not None and (i, j) not in allowed_set:
                return False
            return (i, j) not in avoid_set and (i, j) not in self.claimed

        for r in range(N):
            i_min, i_max = max(0, i0 - r), min(N - 1, i0 + r)
            j_min, j_max = max(0, j0 - r), min(N - 1, j0 + r)
            for i in range(i_min, i_max + 1):          # top and bottom edges
                for j in (j_min, j_max):
                    if ok(i, j):
                        return (i, j)
            for j in range(j_min + 1, j_max):          # left and right edges
                for i in (i_min, i_max):
                    if ok(i, j):
                        return (i, j)
        return None

    # ---------------- latched traverse runs ----------------

    def _edge_goal_from_latch(self, i: int, j: int, axis: str, side: int) -> Tuple[float, float]:
        half = self.cfg.side_m / 2.0
        cell = self.cfg.cell_m
        k = 0.10 if side == 0 else 0.90
        if axis == "E":
            goal_e = -half + i * cell + k * cell
            goal_n = -half + j * cell + 0.5 * cell
        else:
            goal_n = -half + j * cell + k * cell
            goal_e = -half + i * cell + 0.5 * cell
        return goal_n, goal_e

    def _make_run(self, ti: int, tj: int, dn: float, de: float,
                  now: float, vp: Verbose = None) -> None:
        cell = self.cfg.cell_m
        le, ln = self.local_east_north_in_cell(dn, de, ti, tj)
        span_e = self.max_e[ti][tj] - self.min_e[ti][tj]
        span_n = self.max_n[ti][tj] - self.min_n[ti][tj]
        axis = "E" if span_e <= span_n else "N"   # sweep the weaker axis
        side = 1 if (le <= 0.5 * cell if axis == "E" else ln <= 0.5 * cell) else 0
        self.run = {
            "i": ti, "j": tj, "axis": axis, "side": side,
            "goal": self._edge_goal_from_latch(ti, tj, axis, side),
            "last_flip_t": now,
        }
        if vp:
            vp(f"TRAVERSE: latch axis={'E-W' if axis == 'E' else 'N-S'} "
               f"side={'high' if side == 1 else 'low'}")

    def ensure_run_and_goal(self, ti: int, tj: int, dn: float, de: float,
                            now: float, vp: Verbose = None) -> Tuple[float, float]:
        if self.run is None or self.run.get("i") != ti or self.run.get("j") != tj:
            self._make_run(ti, tj, dn, de, now, vp)
        goal_n, goal_e = self.run["goal"]
        dist = math.hypot(goal_n - dn, goal_e - de)
        if dist <= self.flip_dist_m and not self.done[ti][tj]:
            if (now - self.run["last_flip_t"]) >= self.flip_min_sec:
                self.run["side"] = 1 - int(self.run["side"])
                goal_n, goal_e = self._edge_goal_from_latch(ti, tj, self.run["axis"], self.run["side"])
                self.run["goal"] = (goal_n, goal_e)
                self.run["last_flip_t"] = now
                if vp:
                    vp(f"TRAVERSE: flip side (dist={dist:.1f} m)")
        return goal_n, goal_e

    def clear_run(self) -> None:
        self.run = None

    # ---------------- completion ----------------

    def all_done(self) -> bool:
        return all(self.done[i][j]
                   for i in range(self.cfg.cells)
                   for j in range(self.cfg.cells))

    def done_count(self) -> int:
        return sum(1 for i in range(self.cfg.cells)
                   for j in range(self.cfg.cells) if self.done[i][j])