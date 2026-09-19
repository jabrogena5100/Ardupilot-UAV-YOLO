#!/usr/bin/env python3
"""
planning/priority.py — turning evidence of fire into somewhere to fly.

Extracted from the block inside step_grid() that used to chase HAZARD_UPDATE
cells. That block worked, but it took ground-truth fire as input, so it was
really an oracle follower. Same control logic here, fed from a source that can
be either:

    • DETECTION messages — this drone's own network output, or a neighbour's
    • HAZARD_UPDATE — ground truth, only when fire_model.oracle_hazards is on

which means you can run identical flight code in oracle mode and in perception
mode and compare the two fairly. That comparison is the Level 3 result.

Evidence decays
---------------
Each cell's evidence carries a TTL. A detection is a claim about a moment, not
a standing fact: smoke seen 90 seconds ago on a cell nobody has revisited is
weak evidence, and without decay a single false positive would pin a drone to
an empty cell for the rest of the run.

Upgrade path to mapping/probability_map.py
------------------------------------------
This class scores a cell as `max(conf)` over live detections, with a small
bonus for repeat sightings. That is a deliberate placeholder. The principled
version is a log-odds Bayesian grid:

    logit += log(sensitivity / (1 - specificity))       on a positive look
    logit += log((1 - sensitivity) / specificity)       on a negative look

which needs your Level 2 confusion matrices to supply sensitivity and
specificity per model — so it is worth writing *after* the eval matrix is run,
not before. When you do, swap the body of score() and leave choose() alone.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

Cell = Tuple[int, int]

BURNING, BURNT = 1, 2


class Evidence:
    __slots__ = ("conf", "expires_at", "hits", "cls", "source", "last_t")

    def __init__(self, conf: float, expires_at: float, cls: str, source: str, t: float):
        self.conf = float(conf)
        self.expires_at = float(expires_at)
        self.hits = 1
        self.cls = cls
        self.source = source
        self.last_t = t


class PriorityPlanner:
    def __init__(
        self,
        detection_ttl_s: float = 20.0,
        hazard_ttl_s: float = 8.0,
        burnt_ttl_s: float = 20.0,
        conf_threshold: float = 0.45,
        arrive_radius_m: float = 1.0,
        repeat_bonus: float = 0.05,
    ):
        self.detection_ttl_s = detection_ttl_s
        self.hazard_ttl_s = hazard_ttl_s
        self.burnt_ttl_s = burnt_ttl_s
        self.conf_threshold = conf_threshold
        self.arrive_radius_m = arrive_radius_m
        self.repeat_bonus = repeat_bonus

        self.evidence: Dict[Cell, Evidence] = {}
        self.target: Optional[Cell] = None
        self.investigated: Set[Cell] = set()

    # ---------------- inputs ----------------

    def on_detection(self, i: int, j: int, cls: str, conf: float, now: float,
                     source: str = "nn") -> bool:
        """Record a neural-network detection. Returns True if it was strong
        enough to keep. Sub-threshold detections are dropped here rather than
        at the detector, so the DETECTION message still goes on the bus and
        fog still scores it — you want your false positives in the CSV, not
        quietly filtered out before anyone counts them."""
        if conf < self.conf_threshold:
            return False
        key = (int(i), int(j))
        ev = self.evidence.get(key)
        if ev is None:
            self.evidence[key] = Evidence(conf, now + self.detection_ttl_s, cls, source, now)
        else:
            ev.conf = max(ev.conf, float(conf))
            ev.expires_at = now + self.detection_ttl_s
            ev.hits += 1
            ev.last_t = now
        return True

    def on_hazard(self, i: int, j: int, state: int, now: float) -> None:
        """Ground-truth fire, oracle mode only."""
        key = (int(i), int(j))
        ttl = self.burnt_ttl_s if int(state) == BURNT else self.hazard_ttl_s
        conf = 0.5 if int(state) == BURNT else 1.0
        ev = self.evidence.get(key)
        if ev is None:
            self.evidence[key] = Evidence(conf, now + ttl, "fire", "oracle", now)
        else:
            ev.conf = max(ev.conf, conf)
            ev.expires_at = now + ttl
            ev.last_t = now

    # ---------------- maintenance ----------------

    def prune(self, now: float) -> None:
        for key in [k for k, ev in self.evidence.items() if ev.expires_at <= now]:
            self.evidence.pop(key, None)
        if self.target is not None and self.target not in self.evidence:
            self.target = None

    def score(self, cell: Cell) -> float:
        ev = self.evidence.get(cell)
        if ev is None:
            return 0.0
        return min(1.0, ev.conf + self.repeat_bonus * (ev.hits - 1))

    def clear(self, cell: Cell) -> None:
        """Called on arrival: the cell has been looked at, drop it and move on."""
        self.evidence.pop(cell, None)
        self.investigated.add(cell)
        if self.target == cell:
            self.target = None

    # ---------------- output ----------------

    def choose(self, cur_cell: Optional[Cell],
               avoid: Optional[Iterable[Cell]] = None) -> Optional[Cell]:
        """Pick the cell worth diverting to, or None to carry on with coverage.

        Ranking is score first, then distance. Score-first is what makes this a
        *priority* planner rather than a nearest-hazard follower: a confident
        detection across the map outranks a marginal one next door, which is
        the behaviour you want to be able to point at in a demo.

        Sticky targets: once committed, the drone keeps its target while the
        evidence lives, even if a slightly better cell appears. Re-ranking every
        tick makes drones oscillate between two similar candidates and arrive
        at neither."""
        avoid_set = set(avoid or ())
        if self.target is not None and self.target in self.evidence and self.target not in avoid_set:
            return self.target

        candidates = [c for c in self.evidence if c not in avoid_set]
        if not candidates:
            self.target = None
            return None

        def rank(c: Cell) -> Tuple[float, int]:
            dist = 0 if cur_cell is None else max(abs(c[0] - cur_cell[0]), abs(c[1] - cur_cell[1]))
            return (-self.score(c), dist)

        self.target = min(candidates, key=rank)
        return self.target

    # ---------------- reporting ----------------

    def belief_grid(self, n_cells: int) -> List[List[float]]:
        """Current evidence as an N x N grid, for the BELIEF message. This is
        the placeholder the dashboard's probability layer draws; replace with
        the real posterior once probability_map.py lands."""
        grid = [[0.0] * n_cells for _ in range(n_cells)]
        for (i, j) in self.evidence:
            if 0 <= i < n_cells and 0 <= j < n_cells:
                grid[i][j] = self.score((i, j))
        return grid

    def live_cells(self) -> List[Cell]:
        return list(self.evidence.keys())