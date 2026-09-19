#!/usr/bin/env python3
"""
fog/state.py — the ground station's model of the world.

Pure state and geometry. No sockets, no CSV, no HTTP, no argparse. That
separation is what lets you unit-test coverage accounting without opening a
multicast socket, and it is why the dashboard could be split out.

What fog knows
--------------
1. Coverage    — which cells have been visited, and which have been traversed
                 edge-to-edge (the TRAVERSE model).
2. Fire truth  — the authoritative FireSim state. Fog owns this. Agents must
                 never run their own FireSim; two RNGs means two different
                 fires and no ground truth at all.
3. Detections  — what the neural network reported, scored against (2).

Point 3 is the whole of Layer 3. Fog is the only process that can hold both
the belief and the truth, which makes it the only place the comparison can
honestly be made.

Grid convention: i = east index, j = north index, (0,0) = south-west corner.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from wildfire_sim import FireSim

R_EARTH = 6378137.0
M_PER_MILE = 1609.34

UNBURNED, BURNING, BURNT = 0, 1, 2

# How long a detection stays "live" on the dashboard, in seconds of run time.
DET_FRESH_S = 5.0

# Probability below which a cell is not worth putting on the wire.
BELIEF_FLOOR = 0.02


# ---------------- geometry ----------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def ll_to_ne_m(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Equirectangular projection about the origin. Good to well under a metre
    at the ~1 mile scales this project uses."""
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    n = R_EARTH * dlat
    e = R_EARTH * dlon * math.cos(math.radians((lat + lat0) / 2.0))
    return n, e


def cell_index(n: float, e: float, half_side_m: float, cell_m: float, N: int) -> Optional[Tuple[int, int]]:
    """Metres-from-origin -> (i, j), or None if outside the grid."""
    x = e + half_side_m
    y = n + half_side_m
    if x < 0.0 or y < 0.0 or x >= 2 * half_side_m or y >= 2 * half_side_m:
        return None
    i = int(x // cell_m)
    j = int(y // cell_m)
    if 0 <= i < N and 0 <= j < N:
        return (i, j)
    return None


def local_xy_in_cell(n: float, e: float, i: int, j: int,
                     half_side_m: float, cell_m: float) -> Tuple[float, float]:
    """Position within a cell, measured from its south-west corner, in [0, cell_m]."""
    x = e + half_side_m
    y = n + half_side_m
    le = clamp(x - i * cell_m, 0.0, cell_m)
    ln = clamp(y - j * cell_m, 0.0, cell_m)
    return le, ln


# ---------------- per-cell record ----------------

class Cell:
    __slots__ = (
        "visited", "visited_first_sys", "visited_first_strat", "visited_first_t",
        "visits", "last_t",
        "done", "min_e", "max_e", "min_n", "max_n",
    )

    def __init__(self) -> None:
        # first touch
        self.visited = False
        self.visited_first_sys: Optional[int] = None
        self.visited_first_strat: Optional[str] = None
        self.visited_first_t = float("inf")
        self.visits = 0
        self.last_t = 0.0
        # TRAVERSE: a cell is "done" once a drone's track spans traverse_frac
        # of the cell width on either axis.
        self.done = False
        self.min_e = float("inf")
        self.max_e = float("-inf")
        self.min_n = float("inf")
        self.max_n = float("-inf")


# ---------------- fog ----------------

class Fog:
    def __init__(self, lat0: float, lon0: float, miles: float, cells: int,
                 traverse_frac: float, detection_truth_radius: int = 1):
        self.lat0, self.lon0 = lat0, lon0
        self.side_m = miles * M_PER_MILE
        self.half = self.side_m / 2.0
        self.N = int(cells)
        self.cell_m = self.side_m / self.N
        self.traverse_frac = float(traverse_frac)

        self.grid: List[List[Cell]] = [[Cell() for _ in range(self.N)] for __ in range(self.N)]

        # who is who
        self.strat_by_sys: Dict[int, str] = {}
        self.last_cell_by_sys: Dict[int, Tuple[int, int]] = {}
        self.agent_info: Dict[int, Dict[str, Any]] = {}

        # time series (kept in memory, flushed to CSV at the end)
        self.coverage_ts: List[Tuple[float, float, float]] = []
        self.fire_ts: List[Tuple[float, int, int, float, float, int]] = []

        # ---- fire ground truth ----
        self.fire_state: List[List[int]] = [[UNBURNED] * self.N for _ in range(self.N)]
        self.fire_heat: List[List[float]] = [[0.0] * self.N for _ in range(self.N)]
        self.fire_sim: Optional[FireSim] = None

        # ---- Layer 3: perception bookkeeping ----
        # A detection counts as correct if a cell within Chebyshev distance
        # `detection_truth_radius` is burning. Radius 1 is the honest default:
        # a UAV camera footprint at 25 m covers more than one 100 m cell, so
        # demanding an exact cell hit would score a good detection as a miss.
        self.detection_truth_radius = int(detection_truth_radius)
        self.detections: List[Dict[str, Any]] = []
        self.det_by_cell: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.det_true = 0
        self.det_false = 0
        self.first_detection_t: Optional[float] = None   # first correct detection
        self.first_ignition_t: Optional[float] = None    # first cell to start burning

        # latest probability map per agent (dashboard + belief-vs-truth metric)
        self.belief_by_sys: Dict[int, List[List[float]]] = {}

    # ================= coverage =================

    def _mark_visited(self, i: int, j: int, sysid: Optional[int], t: float, strat: Optional[str]) -> None:
        c = self.grid[i][j]
        c.visits += 1
        c.last_t = max(c.last_t, t)
        if not c.visited:
            c.visited = True
            c.visited_first_sys = sysid
            c.visited_first_strat = strat
            c.visited_first_t = t

    def ingest_pose(self, sysid: Optional[int], t: float, lat: float, lon: float,
                    cell_hint: Optional[Sequence[int]], strat: Optional[str],
                    extra: Optional[Dict[str, Any]] = None) -> None:
        """POSE does three jobs: trust the agent's own cell index, update the
        TRAVERSE spans from raw lat/lon, and cache telemetry for the dashboard."""
        sid = int(sysid) if sysid is not None else None
        if sid is not None and strat:
            self.strat_by_sys[sid] = strat

        # 1) agent-supplied cell index
        if isinstance(cell_hint, (list, tuple)) and len(cell_hint) == 2:
            try:
                ih, jh = int(cell_hint[0]), int(cell_hint[1])
                if 0 <= ih < self.N and 0 <= jh < self.N:
                    if sid is not None:
                        self.last_cell_by_sys[sid] = (ih, jh)
                    self._mark_visited(ih, jh, sid, t, self.strat_by_sys.get(sid or -1, strat))
            except Exception:
                pass

        # 2) raw position -> spans
        n, e = ll_to_ne_m(lat, lon, self.lat0, self.lon0)
        idx = cell_index(n, e, self.half, self.cell_m, self.N)

        # 3) telemetry cache (kept even when the drone is outside the grid,
        #    otherwise a drone that drifts out vanishes from the dashboard)
        if sid is not None:
            info = self.agent_info.get(sid, {})
            info.update({"last_t": t, "lat": lat, "lon": lon})
            if idx is not None:
                info["cell"] = [idx[0], idx[1]]
            if extra:
                info.update({k: v for k, v in extra.items() if v is not None})
            self.agent_info[sid] = info

        if idx is None:
            return
        i, j = idx
        if sid is not None:
            self.last_cell_by_sys[sid] = (i, j)

        self._mark_visited(i, j, sid, t, self.strat_by_sys.get(sid or -1, strat))

        le, ln = local_xy_in_cell(n, e, i, j, self.half, self.cell_m)
        c = self.grid[i][j]
        c.min_e = min(c.min_e, le)
        c.max_e = max(c.max_e, le)
        c.min_n = min(c.min_n, ln)
        c.max_n = max(c.max_n, ln)
        if not c.done:
            span = max(c.max_e - c.min_e, c.max_n - c.min_n)
            if span >= self.traverse_frac * self.cell_m:
                c.done = True

    def ingest_visit(self, sysid: Optional[int], t: float, i: int, j: int, strat: Optional[str]) -> None:
        if not (0 <= i < self.N and 0 <= j < self.N):
            return
        sid = int(sysid) if sysid is not None else None
        if sid is not None and strat:
            self.strat_by_sys[sid] = strat
        if sid is not None:
            self.last_cell_by_sys[sid] = (i, j)
            info = self.agent_info.get(sid, {})
            info.update({"cell": [i, j], "last_t": t})
            self.agent_info[sid] = info
        self._mark_visited(i, j, sid, t, self.strat_by_sys.get(sid or -1, strat))

    def snapshot(self, t_now: float) -> Tuple[float, float, int, int]:
        """Append one coverage sample and return (visited_frac, traverse_frac,
        visited_cnt, traverse_cnt)."""
        visited_cnt = traverse_cnt = 0
        for row in self.grid:
            for c in row:
                if c.visited or c.done:
                    visited_cnt += 1
                if c.done:
                    traverse_cnt += 1
        total = self.N * self.N
        vf, tf = visited_cnt / total, traverse_cnt / total
        self.coverage_ts.append((t_now, vf, tf))
        return vf, tf, visited_cnt, traverse_cnt

    def all_visited(self) -> bool:
        return all(c.visited for row in self.grid for c in row)

    def all_traverse_done(self) -> bool:
        return all(c.done for row in self.grid for c in row)

    # ================= fire truth =================

    def sync_fire_from_sim(self, sim: FireSim) -> None:
        snap = sim.snapshot()
        for i in range(self.N):
            for j in range(self.N):
                self.fire_state[i][j] = snap[i][j]
                self.fire_heat[i][j] = clamp(sim.cells[i][j].heat, 0.0, 1.0)
                if self.first_ignition_t is None and snap[i][j] == BURNING:
                    self.first_ignition_t = sim.time_s

    def ingest_fire(self, i: int, j: int, state: Optional[int], heat: Optional[float], t: float) -> None:
        """Only used if some other process owns the fire. Normally fog owns it."""
        if not (0 <= i < self.N and 0 <= j < self.N):
            return
        self.fire_state[i][j] = max(0, min(2, 0 if state is None else int(state)))
        self.fire_heat[i][j] = clamp(0.0 if heat is None else float(heat), 0.0, 1.0)

    def burning_cells(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(self.N) for j in range(self.N)
                if self.fire_state[i][j] == BURNING]

    def burnt_cells(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(self.N) for j in range(self.N)
                if self.fire_state[i][j] == BURNT]

    def proximity_detections(self) -> int:
        """LEGACY metric, kept so old runs stay comparable: how many drones are
        within Chebyshev 1 of a burning cell.

        This is proximity, not perception — it says a drone *could* have seen
        the fire, not that the network *did*. Once Layer 3 is wired up, report
        det_true / det_false instead. Kept in the CSV as `proximity_cnt`."""
        burning = self.burning_cells()
        if not burning or not self.last_cell_by_sys:
            return 0
        hits = 0
        for ci, cj in self.last_cell_by_sys.values():
            if any(max(abs(ci - bi), abs(cj - bj)) <= 1 for bi, bj in burning):
                hits += 1
        return hits

    def fire_snapshot(self, t_now: float) -> Tuple[int, int, float, float, int]:
        burn = len(self.burning_cells())
        burnt = len(self.burnt_cells())
        total = self.N * self.N
        prox = self.proximity_detections()
        row = (t_now, burn, burnt, burn / total, burnt / total, prox)
        self.fire_ts.append(row)
        return burn, burnt, burn / total, burnt / total, prox

    # ================= perception (Layer 3) =================

    def _truth_at(self, i: int, j: int) -> int:
        if 0 <= i < self.N and 0 <= j < self.N:
            return self.fire_state[i][j]
        return UNBURNED

    def _is_correct(self, i: int, j: int) -> bool:
        r = self.detection_truth_radius
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if self._truth_at(i + di, j + dj) == BURNING:
                    return True
        return False

    def ingest_detection(self, msg: Dict[str, Any], t_rel: float) -> Dict[str, Any]:
        """Score one DETECTION against ground truth and file it.

        Returns the scored record so the caller can stream it to CSV. The
        score is computed *now*, at ingest, because fire state changes: a
        detection that was correct at t=40 s may sit on a burnt cell by t=90 s,
        and grading it later would be revisionist."""
        i, j = int(msg.get("i", -1)), int(msg.get("j", -1))
        correct = self._is_correct(i, j)
        rec = {
            "t": round(t_rel, 2),
            "sys": int(msg.get("sys", 0)),
            "i": i, "j": j,
            "cls": str(msg.get("cls", "")),
            "conf": float(msg.get("conf", 0.0)),
            "model": str(msg.get("model", "")),
            "truth_state": self._truth_at(i, j),
            "correct": int(correct),
        }
        self.detections.append(rec)
        if correct:
            self.det_true += 1
            if self.first_detection_t is None:
                self.first_detection_t = t_rel
        else:
            self.det_false += 1

        if 0 <= i < self.N and 0 <= j < self.N:
            prev = self.det_by_cell.get((i, j))
            if prev is None or rec["conf"] > prev["conf"]:
                self.det_by_cell[(i, j)] = rec
        return rec

    def ingest_belief(self, sysid: Optional[int], probs: Any, n_cells: int) -> None:
        if sysid is None or int(n_cells) != self.N:
            return
        if not isinstance(probs, list) or len(probs) != self.N:
            return
        self.belief_by_sys[int(sysid)] = probs

    def merged_belief(self) -> List[List[float]]:
        """Per-cell max across agents. Simple and legible on a dashboard; if
        you want the swarm's *joint* belief, do the log-odds merge in
        mapping/probability_map.py and broadcast that instead."""
        out = [[0.0] * self.N for _ in range(self.N)]
        for grid in self.belief_by_sys.values():
            for i in range(min(self.N, len(grid))):
                row = grid[i]
                for j in range(min(self.N, len(row))):
                    try:
                        v = float(row[j])
                    except (TypeError, ValueError):
                        continue
                    if v > out[i][j]:
                        out[i][j] = v
        return out

    def detection_scores(self) -> Dict[str, Any]:
        """Precision and lag. Note there is no recall here: a false negative is
        'a burning cell no drone ever photographed', which needs the camera
        footprint log from sim/camera.py to compute honestly. Compute it
        offline in eval/ rather than guessing at runtime."""
        total = self.det_true + self.det_false
        return {
            "det_true": self.det_true,
            "det_false": self.det_false,
            "det_total": total,
            "precision": (self.det_true / total) if total else 0.0,
            "first_ignition_t": self.first_ignition_t,
            "first_detection_t": self.first_detection_t,
            "detection_lag_s": (
                round(self.first_detection_t - self.first_ignition_t, 2)
                if (self.first_detection_t is not None and self.first_ignition_t is not None)
                else None
            ),
        }

    # ================= snapshots for the wire =================

    def _vis_bitmap_hex(self) -> Tuple[str, str]:
        vis = done = 0
        for i in range(self.N):
            for j in range(self.N):
                bit = 1 << (i * self.N + j)
                c = self.grid[i][j]
                if c.visited or c.done:
                    vis |= bit
                if c.done:
                    done |= bit
        return format(vis, "x"), format(done, "x")

    def compact_state(self, t_rel: float, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Everything the dashboard needs, in one small dict.

        Coverage travels as two hex bitmaps and fire as one N*N digit string,
        which keeps a 10x10 world at a few hundred bytes. Sending the visited
        cells as a list of [i,j] pairs — as the old inline viz did — is about
        6x bigger and was also, as it happened, never populated: the old
        current_stats() emitted visited_cnt but no `visited` list, so the
        canvas drew coverage nowhere. Fixed here."""
        vis_hex, done_hex = self._vis_bitmap_hex()
        fire_str = "".join(
            str(self.fire_state[i][j]) for i in range(self.N) for j in range(self.N)
        )
        total = self.N * self.N
        visited_cnt = bin(int(vis_hex, 16)).count("1") if vis_hex else 0
        traverse_cnt = bin(int(done_hex, 16)).count("1") if done_hex else 0
        burn = len(self.burning_cells())
        burnt = len(self.burnt_cells())

        now = time.time()
        agents: Dict[str, Any] = {}
        for sid, info in self.agent_info.items():
            # Show a detection badge only if this drone's current cell has a
            # recent one. Stale badges make a dashboard lie convincingly.
            det = None
            cell = info.get("cell")
            if cell:
                d = self.det_by_cell.get((int(cell[0]), int(cell[1])))
                if d is not None and (t_rel - d["t"]) <= DET_FRESH_S:
                    det = {"cls": d["cls"], "conf": d["conf"], "correct": d["correct"]}
            agents[str(sid)] = {
                "cell": cell,
                "lat": info.get("lat"), "lon": info.get("lon"),
                "alt": info.get("alt"), "hdg": info.get("hdg"),
                "batt": info.get("batt"), "mode": info.get("mode"),
                "mission": info.get("mission"), "claim": info.get("claim"),
                "strat": self.strat_by_sys.get(sid),
                "age_s": round(now - float(info.get("last_t", now)), 1),
                "det": det,
            }

        out: Dict[str, Any] = {
            "N": self.N,
            "t": round(t_rel, 2),
            "vis": vis_hex,
            "done": done_hex,
            "fire": fire_str,
            "visited_cnt": visited_cnt,
            "traverse_cnt": traverse_cnt,
            "visited_frac": visited_cnt / total,
            "traverse_frac": traverse_cnt / total,
            "burning_cnt": burn,
            "burnt_cnt": burnt,
            "burning_frac": burn / total,
            "burnt_frac": burnt / total,
            "proximity_cnt": self.proximity_detections(),
            "agents": agents,
            "recent_det": self.detections[-12:],
        }
        out.update(self.detection_scores())
        if self.belief_by_sys:
            # Sparse: [[i, j, p], ...] for cells above the floor. A dense 10x10
            # grid of "0.xx" costs ~500 B and pushed this packet past one MTU;
            # in practice only a handful of cells carry probability, so the
            # sparse form is an order of magnitude smaller and stays in one
            # datagram at N=40.
            out["belief"] = [
                [i, j, round(p, 2)]
                for i, row in enumerate(self.merged_belief())
                for j, p in enumerate(row)
                if p >= BELIEF_FLOOR
            ]
        if meta:
            out.update(meta)
        return out