#!/usr/bin/env python3
"""
swarm_agent.py — one drone's brain.

Split out of the old 1086-line build. What lives where now:

    mission/geo.py           projection and Chebyshev distance
    mission/grid_mission.py  TRAVERSE coverage: frontier choice + edge sweeps
    mission/partitions.py    static area assignment
    mapping/bitmap.py        visited/done bitmaps and neighbour merging
    planning/priority.py     detections -> a cell worth investigating
    perception/base.py       the neural-network seam (NullPerception default)
    comms/gossip.py          the multicast bus
    comms/schema.py          message shapes
    swarm_strategies.py      fallback velocity policies (unchanged)

What is left in this file is the control loop: read state, talk, decide,
clamp, command. That is genuinely one job.

Changes in behaviour from the previous build
--------------------------------------------
1. No FireSim here. Every agent used to be able to run its own copy of the
   wildfire model from --fire-config-json. N agents plus fog meant N+1
   independent fires with N+1 random seeds, and therefore no ground truth.
   Fog owns the fire. The flag is gone.

2. Perception hook. Each tick (rate-limited) the agent asks its Perception for
   detections, broadcasts each as a DETECTION message, and feeds them to the
   planner. With NullPerception — the default — this costs one function call
   returning [] and the agent behaves exactly as before.

3. Fire-chasing moved to PriorityPlanner. Same control law, but the evidence
   can now come from the network instead of only from the oracle.

4. POSE now carries heading, battery, and mode, which the dashboard's fleet
   panel displays.

5. MAP bit order is i*N+j, matching fog and the dashboard. See mapping/bitmap.py.

CLI is unchanged apart from --fire-config-json being removed, so
launch_experiment.py needs no edits beyond the core/ path fix.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import swarm_strategies as strat
from comms import schema
from comms.gossip import GossipBus
from custom_vehicle import CustomVehicle
from mapping.bitmap import CoverageBitmap
from mission.geo import M_PER_MILE, clamp, cheby_dist, latlon_to_ne
from mission.grid_mission import GridCfg, GridMission
from mission.partitions import allowed_cells
from perception import base as perception_base
from planning.priority import PriorityPlanner


@dataclass
class Params:
    alt_target: float = 20.0
    vxy_max: float = 30.0
    vz_max: float = 2.0
    axy_max: float = 3.0
    az_max: float = 1.5
    rate_hz: float = 10.0
    gossip_group: str = "239.255.0.1"
    gossip_port: int = 5005
    comm_grid_radius: int = 2
    comm_mode: str = "full"     # full | tx_only | off
    end_when: str = "never"     # never | all_done | time
    max_seconds: float = 0.0
    verbose: bool = False


class Agent:
    def __init__(
        self,
        name: str,
        conn: str,
        origin_lat: float,
        origin_lon: float,
        geofence_miles: float,
        params: Params,
        mission: str = "grid_frontier",
        grid_cfg: Optional[GridCfg] = None,
        policy_name: str = "random",
        experiment_tag: str = "",
        strategy_tag: str = "",
        seed: Optional[int] = None,
        partition_scheme: str = "none",
        partition_id: int = 0,
        partition_n: int = 1,
        perception_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.params = params
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.half_side_m = (geofence_miles * M_PER_MILE) / 2.0
        self.experiment_tag = experiment_tag
        self.strategy_tag = strategy_tag or policy_name
        self.stop = False
        self.sysid: Optional[int] = None
        self.t0 = time.time()

        self._vp = ((lambda s: print(f"[{self.name}|sys{self.sysid}] {s}"))
                    if params.verbose else (lambda s: None))

        # ---- vehicle and bus ----
        self.v = CustomVehicle(conn, autorequest_rates=True)
        self.gossip = GossipBus(params.gossip_group, params.gossip_port, name=name)

        # ---- policy ----
        if seed is not None:
            random.seed(seed)
        self.policy = strat.make_policy(
            policy_name, alt_target=params.alt_target, vxy_max=params.vxy_max,
            vz_max=params.vz_max, lat0=origin_lat, lon0=origin_lon,
        )

        # ---- mission ----
        self.mission = mission.lower()
        self.grid: Optional[GridMission] = None
        if self.mission == "grid_frontier":
            self.grid = GridMission(grid_cfg or GridCfg(lat0=origin_lat, lon0=origin_lon))
        self.N = self.grid.cfg.cells if self.grid else 10

        self.partition_scheme = partition_scheme
        self.partition_id = max(0, partition_id)
        self.partition_n = max(1, partition_n)
        self._allowed = allowed_cells(partition_scheme, self.N,
                                      self.partition_id, self.partition_n) if self.grid else None

        # ---- shared map ----
        self.bitmap = CoverageBitmap(self.N)

        # ---- motion state ----
        self.vx_prev = self.vy_prev = self.vz_prev = 0.0
        self._rtl_requested = False

        # ---- cell tracking and VISIT sharing ----
        self.cur_cell: Optional[Tuple[int, int]] = None
        self.prev_cell: Optional[Tuple[int, int]] = None
        self.last_visit_emit_t = 0.0
        self.visit_emit_period = 0.5
        self.neighbor_visit_ttl_s = 5.0
        self.neighbor_visits: Dict[Tuple[int, int], float] = {}
        self._map_emit_period = 2.0
        self._last_map_emit = 0.0
        self._nudged_once_for_idx: Optional[int] = None
        self._complete_printed = False

        # ---- Layer 3: perception and priority planning ----
        pcfg = perception_cfg or {}
        self.perception = perception_base.from_config(pcfg, grid=self.grid)
        self.model_tag = str(pcfg.get("model_tag", getattr(self.perception, "model_tag", "")))
        self.perception_period = 1.0 / max(0.05, float(pcfg.get("rate_hz", 1.0)))
        self._next_perception_t = 0.0
        self.planner = PriorityPlanner(
            detection_ttl_s=float(pcfg.get("detection_ttl_s", 20.0)),
            conf_threshold=float(pcfg.get("conf_threshold", 0.45)),
        )
        self._belief_emit_period = 2.0
        self._last_belief_emit = 0.0

    # ================= setup / teardown =================

    def ensure_ready(self) -> None:
        self.v.wait_for_healthy(timeout=20.0, min_gps_fix=3)
        self.v.set_mode("GUIDED")
        self.v.wait_for_mode("GUIDED", timeout=10.0)
        self.v.arm(True)
        self.v.wait_for_armed(True, timeout=12.0)
        self.v.takeoff(self.params.alt_target)
        self.v.wait_for(
            lambda s: (s.get("alt_rel_m") or 0.0) >= 0.9 * self.params.alt_target,
            timeout=60.0,
        )
        try:
            self.sysid = int(getattr(self.v, "target_system", 0) or 0)
        except Exception:
            self.sysid = 0
        self._vp("READY: GUIDED / ARMED / altitude reached")

    def _request_rtl(self) -> None:
        """One-shot: stop moving and hand the vehicle back to the autopilot."""
        if self._rtl_requested:
            return
        self._rtl_requested = True
        for action in (lambda: self.v.brake(),
                       lambda: self.v.set_mode("RTL"),
                       lambda: self.v.wait_for_mode("RTL", timeout=5.0),
                       lambda: self.v.rtl()):
            try:
                action()
            except Exception:
                pass

    def close(self) -> None:
        self._request_rtl()
        try:
            self.perception.close()
        except Exception:
            pass
        try:
            self.gossip.close()
        except Exception:
            pass
        self.v.close()

    # ================= outgoing messages =================

    def state_packet(self) -> Dict[str, Any]:
        st = self.v.get_state()
        lat, lon = st.get("lat"), st.get("lon")
        cell = None
        if self.grid and lat is not None and lon is not None:
            i, j = self.grid.pos_to_cell(lat, lon)
            if self.grid.in_bounds(i, j):
                cell = [i, j]
                if self.cur_cell != (i, j):
                    self.prev_cell = self.cur_cell
                self.cur_cell = (i, j)
                self.bitmap.set_visited(i, j)
        claim = list(self.grid.target) if (self.grid and self.grid.target) else None
        return schema.make_pose(
            sysid=int(self.sysid or 0), name=self.name,
            lat=lat, lon=lon, alt=st.get("alt_rel_m"),
            vx=st.get("vx"), vy=st.get("vy"), vz=st.get("vz"),
            heading_deg=st.get("heading_deg"),
            battery_pct=st.get("battery_remaining_pct"),
            mode=st.get("mode"),
            mission=self.mission, strat=self.strategy_tag, exp=self.experiment_tag,
            cell=cell, claim=claim,
        )

    def _emit_visit_now(self, now: float, override_cell: Optional[Tuple[int, int]] = None) -> None:
        cell = override_cell if override_cell is not None else self.cur_cell
        if cell is None:
            return
        # Always transmit, even with comm_mode 'off': 'off' means this drone
        # does not LISTEN to peers, not that it hides from the ground station.
        # Fog needs VISIT to score coverage in every condition.
        self.gossip.send(schema.make_visit(sysid=int(self.sysid or 0),
                                           i=int(cell[0]), j=int(cell[1]),
                                           strat=self.strategy_tag, t=now))
        self.last_visit_emit_t = now

    def emit_visit_if_due(self, now: float) -> None:
        if self.cur_cell and (now - self.last_visit_emit_t) >= self.visit_emit_period:
            self._emit_visit_now(now)

    def emit_map_if_due(self, now: float, cell: Optional[List[int]]) -> None:
        if (now - self._last_map_emit) < self._map_emit_period:
            return
        self.gossip.send(schema.make_map(
            sysid=int(self.sysid or 0), n_cells=self.N,
            vis_hex=self.bitmap.vis_hex(), done_hex=self.bitmap.done_hex(),
            strat=self.strategy_tag, cell=cell, t=now,
        ))
        self._last_map_emit = now

    def emit_belief_if_due(self, now: float) -> None:
        if not self.planner.evidence:
            return
        if (now - self._last_belief_emit) < self._belief_emit_period:
            return
        self.gossip.send(schema.make_belief(
            sysid=int(self.sysid or 0), n_cells=self.N,
            probs=self.planner.belief_grid(self.N),
            strat=self.strategy_tag, t=now,
        ))
        self._last_belief_emit = now

    # ================= perception =================

    def run_perception(self, mine: Dict[str, Any], now: float) -> None:
        """Look through the camera, broadcast what the network saw.

        Rate-limited independently of the control loop: the controller wants
        10 Hz, a YOLO forward pass does not, and coupling them would make
        flight quality depend on inference latency."""
        if not getattr(self.perception, "enabled", False):
            return
        if now < self._next_perception_t:
            return
        self._next_perception_t = now + self.perception_period

        pose = perception_base.Pose(
            lat=mine.get("lat"), lon=mine.get("lon"), alt=mine.get("alt"),
            heading_deg=mine.get("hdg"), cell=self.cur_cell, t=now,
            sysid=int(self.sysid or 0),
        )
        try:
            detections = self.perception.observe(pose)
        except Exception as e:
            self._vp(f"PERCEPTION: observe failed ({e})")
            return

        for d in detections:
            # Broadcast every detection, including weak ones. Fog scores them
            # all; filtering before the bus would hide false positives from
            # the very CSV that is supposed to measure them.
            self.gossip.send(schema.make_detection(
                sysid=int(self.sysid or 0), i=d.i, j=d.j, cls=d.cls, conf=d.conf,
                model_tag=self.model_tag, bbox=d.bbox,
                lat=mine.get("lat"), lon=mine.get("lon"), alt=mine.get("alt"),
                frame_id=d.frame_id, t=now,
            ))
            if self.planner.on_detection(d.i, d.j, d.cls, d.conf, now, source="self"):
                self._vp(f"DETECT: {d.cls} {d.conf:.2f} at ({d.i},{d.j})")

    # ================= incoming messages =================

    def recv_neighbors(self, mine: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Drain the bus. Returns in-range POSE packets; everything else is
        folded into local state here."""
        my_cell = tuple(mine["cell"]) if mine.get("cell") else None
        now = time.time()
        out: List[Dict[str, Any]] = []

        for m in self.gossip.recv_all():
            if not schema.is_valid(m):
                continue
            typ = m["type"]

            if typ == schema.EXPERIMENT_END:
                print(f"[{self.name}|sys{self.sysid}] STOP: EXPERIMENT_END "
                      f"reason={m.get('reason', '')}")
                self.stop = True
                continue

            if typ == schema.FOG_STATE:
                continue  # for the dashboard, not for us

            if m.get("sys") == mine.get("sys"):
                continue  # our own echo

            if typ == schema.VISIT:
                i, j = int(m["i"]), int(m["j"])
                if my_cell and cheby_dist(my_cell, (i, j)) <= self.params.comm_grid_radius:
                    self.neighbor_visits[(i, j)] = now + self.neighbor_visit_ttl_s
                continue

            if typ == schema.MAP:
                if not my_cell or int(m.get("N", -1)) != self.N:
                    continue
                cell = m.get("cell")
                if cell and len(cell) == 2:
                    if cheby_dist(my_cell, (int(cell[0]), int(cell[1]))) > self.params.comm_grid_radius:
                        continue
                self.bitmap.merge_visited(str(m.get("vis", "")))
                for (di, dj) in self.bitmap.merge_done(str(m.get("done", ""))):
                    if self.grid and self.grid.in_bounds(di, dj):
                        self.grid.done[di][dj] = True
                continue

            if typ == schema.HAZARD_UPDATE:
                # Only arrives in oracle mode; fog is silent otherwise.
                if m.get("hazard") == "fire":
                    i, j = int(m["i"]), int(m["j"])
                    if 0 <= i < self.N and 0 <= j < self.N:
                        self.planner.on_hazard(i, j, int(m.get("state", 1)), now)
                continue

            if typ == schema.DETECTION:
                # A neighbour's network saw something. Trusting peer detections
                # is what makes 'coop' differ from 'solo' at the perception
                # layer, not just the coverage layer.
                i, j = int(m["i"]), int(m["j"])
                if my_cell and cheby_dist(my_cell, (i, j)) > self.params.comm_grid_radius:
                    continue
                self.planner.on_detection(i, j, str(m.get("cls", "")),
                                          float(m.get("conf", 0.0)), now, source="peer")
                continue

            if typ == schema.POSE and self.mission == "grid_frontier":
                cell = m.get("cell")
                if my_cell and cell:
                    if cheby_dist(my_cell, (int(cell[0]), int(cell[1]))) <= self.params.comm_grid_radius:
                        out.append(m)
                continue

        self._prune(now)
        return out

    def _prune(self, now: float) -> None:
        for k in [k for k, exp in self.neighbor_visits.items() if exp <= now]:
            self.neighbor_visits.pop(k, None)
        self.planner.prune(now)

    # ================= geofence =================

    def geofence_push(self, vn: float, ve: float,
                      lat: Optional[float], lon: Optional[float]) -> Tuple[float, float]:
        """Outside the box, drive straight back in. Inside but close to an
        edge, taper speed toward it so the drone decelerates rather than
        bouncing off the boundary."""
        if lat is None or lon is None:
            return vn, ve
        en, ee = latlon_to_ne(lat, lon, self.origin_lat, self.origin_lon)
        half = self.half_side_m
        margin = 10.0

        push_n = push_e = 0.0
        if en > half:
            push_n -= (en - half)
        if en < -half:
            push_n += (-half - en)
        if ee > half:
            push_e -= (ee - half)
        if ee < -half:
            push_e += (-half - ee)

        if push_n or push_e:
            d = math.hypot(push_n, push_e) + 1e-6
            if self.params.verbose:
                print(f"[{self.name}|sys{self.sysid}] GEOFENCE: pushback")
            return -3.0 * (push_n / d), -3.0 * (push_e / d)

        def taper(pos: float) -> float:
            return clamp((half - abs(pos)) / margin, 0.1, 1.0) if abs(pos) > (half - margin) else 1.0

        return vn * taper(en), ve * taper(ee)

    def hard_recentering(self, lat: Optional[float],
                         lon: Optional[float]) -> Optional[Tuple[float, float, str]]:
        """Well outside the geofence or the grid: abandon the mission and fly
        at the origin until back inside. Overrides everything else."""
        if lat is None or lon is None:
            return None
        en, ee = latlon_to_ne(lat, lon, self.origin_lat, self.origin_lon)
        margin = 20.0
        outside = (abs(en) > self.half_side_m + margin) or (abs(ee) > self.half_side_m + margin)
        if self.grid:
            half_grid = self.grid.cfg.side_m / 2.0
            outside = outside or (abs(en) > half_grid + margin) or (abs(ee) > half_grid + margin)
        if not outside:
            return None
        d = math.hypot(en, ee) + 1e-6
        speed = max(15.0, self.params.vxy_max)
        return (-en / d * speed, -ee / d * speed, "RECENTER")

    def reset_mission_state(self) -> None:
        if self.grid:
            self.grid.target = None
            self.grid.clear_run()
            self.grid.claimed.clear()
        self.neighbor_visits.clear()

    # ================= mission logic =================

    def _alt_hold_vz(self, mine: Dict[str, Any]) -> float:
        alt = mine.get("alt") or self.params.alt_target
        return clamp(-0.4 * (self.params.alt_target - alt),
                     -self.params.vz_max, self.params.vz_max)

    def _velocity_to(self, goal_n: float, goal_e: float,
                     me_n: float, me_e: float) -> Tuple[float, float, float]:
        """Proportional approach that saturates at vxy_max and eases in over
        the last few metres, so the drone settles instead of overshooting."""
        dn, de = goal_n - me_n, goal_e - me_e
        d = math.hypot(dn, de)
        if d < 1e-3:
            return 0.0, 0.0, d
        vmag = min(self.params.vxy_max, d)
        return (dn / d) * vmag, (de / d) * vmag, d

    def step_grid(self, mine: Dict[str, Any],
                  neigh: List[Dict[str, Any]]) -> Tuple[float, float, float, str]:
        assert self.grid is not None
        lat, lon = mine.get("lat"), mine.get("lon")
        now = time.time()

        # --- 1. progress on the cell we are in ---
        if lat is not None and lon is not None:
            before = self.grid.done[self.cur_cell[0]][self.cur_cell[1]] if self.cur_cell else False
            self.grid.update_progress(lat, lon)
            if self.cur_cell:
                i, j = self.cur_cell
                if self.grid.done[i][j] and not before:
                    self.bitmap.set_done(i, j)
                    self._emit_visit_now(now)

        # --- 2. recentering overrides everything ---
        rc = self.hard_recentering(lat, lon)
        if rc is not None:
            vn, ve, tag = rc
            if lat is not None and lon is not None:
                en, ee = latlon_to_ne(lat, lon, self.origin_lat, self.origin_lon)
                if math.hypot(en, ee) <= 5.0:
                    self.reset_mission_state()
                    ve = max(ve, 0.5)
            return vn, ve, self._alt_hold_vz(mine), tag

        # --- 3. neighbour claims ---
        if self.params.comm_mode == "full":
            for m in neigh:
                c = m.get("claim")
                if c and len(c) == 2:
                    self.grid.claimed[(int(c[0]), int(c[1]))] = int(m.get("sys") or 0)

        # --- 4. drop a target that finished or was taken ---
        if self.grid.target is not None:
            i, j = self.grid.target
            if self.grid.done[i][j]:
                self._vp(f"DONE: cell ({i},{j})")
                self.grid.claimed.pop((i, j), None)
                self.grid.target = None
                self.grid.clear_run()
            elif self.params.comm_mode == "full":
                owner = self.grid.claimed.get((i, j))
                if owner is not None and owner != self.sysid:
                    self._vp(f"TARGET: stolen by sys={owner}")
                    self.grid.target = None
                    self.grid.clear_run()

        # --- 5. investigate fire evidence before continuing coverage ---
        if lat is not None and lon is not None:
            ci, cj = self.grid.pos_to_cell(lat, lon)
            hot = self.planner.choose((ci, cj), avoid=self.bitmap.merged_avoid)
            if hot is not None:
                goal_n, goal_e = self.grid.cell_center_ne(*hot)
                me_n, me_e = latlon_to_ne(lat, lon, self.grid.cfg.lat0, self.grid.cfg.lon0)
                vn, ve, d = self._velocity_to(goal_n, goal_e, me_n, me_e)
                if d < self.planner.arrive_radius_m:
                    self.planner.clear(hot)
                return vn, ve, self._alt_hold_vz(mine), f"INVESTIGATE→{hot[0]},{hot[1]}"

        # --- 6. pick a coverage target ---
        if self.grid.target is None and lat is not None and lon is not None:
            ci, cj = self.grid.pos_to_cell(lat, lon)
            avoid = set(self.neighbor_visits) | self.bitmap.merged_avoid
            self.grid.target = self.grid.nearest_frontier(ci, cj, self._allowed, avoid)
            if self.grid.target is None:
                # Stuck guard: relax the avoid-set, then the partition. Without
                # this, a drone whose partition is finished hovers forever while
                # cells elsewhere go uncovered.
                self.grid.target = (self.grid.nearest_frontier(ci, cj, self._allowed, None)
                                    or self.grid.nearest_frontier(ci, cj, None, None))
                if self.grid.target:
                    self._vp(f"TARGET: forced {self.grid.target} (stuck-guard)")
            else:
                self._vp(f"TARGET: {self.grid.target}")
            if self.grid.target:
                self.grid.claimed[self.grid.target] = int(self.sysid or 0)

        if self.params.end_when == "all_done" and self.grid.all_done():
            return 0.0, 0.0, 0.0, "HOLD:complete"

        if self.grid.target is None or lat is None or lon is None:
            vn, ve, vz, mode = self.policy.step(mine, [])
            return vn, ve, vz, f"SEARCH:fallback({mode})"

        # --- 7. fly to it: approach the centre, then sweep edge to edge ---
        ti, tj = self.grid.target
        me_n, me_e = latlon_to_ne(lat, lon, self.grid.cfg.lat0, self.grid.cfg.lon0)
        ci, cj = self.grid.pos_to_cell(lat, lon)

        if (ci, cj) != (ti, tj):
            goal_n, goal_e = self.grid.cell_center_ne(ti, tj)
            label = f"APPROACH→{ti},{tj}"
        elif self.grid.done[ti][tj]:
            self._vp(f"DONE (race): {self.grid.target} finished by someone else")
            self.grid.claimed.pop((ti, tj), None)
            self.grid.target = None
            self.grid.clear_run()
            vn, ve, vz, _ = self.policy.step(mine, [])
            return vn, ve, vz, "SEARCH:retarget"
        else:
            goal_n, goal_e = self.grid.ensure_run_and_goal(ti, tj, me_n, me_e, now, self._vp)
            label = f"TRAVERSE→{ti},{tj}"

        vn, ve, _ = self._velocity_to(goal_n, goal_e, me_n, me_e)
        return vn, ve, self._alt_hold_vz(mine), label

    def _maybe_nudge_last_cell(self, now: float) -> None:
        """Endgame: with one cell left, every drone avoids it as someone else's
        claim and the run stalls. Force it."""
        rem = self.bitmap.single_remaining()
        if rem is None:
            return
        ri, rj, ridx = rem
        if self.grid and self.grid.target != (ri, rj):
            self.grid.target = (ri, rj)
            self.grid.claimed[(ri, rj)] = int(self.sysid or 0)
            self.neighbor_visits.clear()
            self._vp(f"NUDGE: forcing target to last cell {(ri, rj)}")
        if self.cur_cell and cheby_dist(self.cur_cell, (ri, rj)) <= 1:
            if self._nudged_once_for_idx != ridx:
                self._emit_visit_now(now, override_cell=(ri, rj))
                self._nudged_once_for_idx = ridx

    # ================= main loop =================

    def run(self) -> None:
        self.ensure_ready()
        dt = 1.0 / max(5.0, self.params.rate_hz)
        last_print = 0.0
        try:
            while not self.stop:
                loop_now = time.time()

                if (self.params.end_when == "time" and self.params.max_seconds > 0
                        and (loop_now - self.t0) >= self.params.max_seconds):
                    print(f"[{self.name}|sys{self.sysid}] STOP: max time reached")
                    break

                mine = self.state_packet()
                t_msg = mine["t"]

                # ---- transmit (always, regardless of comm_mode) ----
                if self.cur_cell and self.cur_cell != self.prev_cell:
                    self._emit_visit_now(t_msg)
                self.gossip.send(mine)
                self.emit_visit_if_due(t_msg)
                self.emit_map_if_due(t_msg, mine.get("cell"))

                # ---- look ----
                self.run_perception(mine, t_msg)
                self.emit_belief_if_due(t_msg)

                # ---- listen (only in full comm mode) ----
                neigh = self.recv_neighbors(mine) if self.params.comm_mode == "full" else []
                if self.stop:
                    break

                self._maybe_nudge_last_cell(t_msg)

                # ---- decide ----
                if self.mission == "grid_frontier" and self.grid is not None:
                    vx, vy, vz, mode = self.step_grid(mine, neigh)
                else:
                    vx, vy, vz, mode = self.policy.step(mine, neigh)

                if self.params.end_when == "all_done" and self.grid and self.grid.all_done():
                    if not self._complete_printed:
                        print(f"[{self.name}|sys{self.sysid}] STOP: all cells complete")
                        self._complete_printed = True
                    break

                # ---- constrain ----
                vx, vy = self.geofence_push(vx, vy, mine.get("lat"), mine.get("lon"))
                dvxy = self.params.axy_max * dt
                dvz = self.params.az_max * dt
                vx = clamp(vx, self.vx_prev - dvxy, self.vx_prev + dvxy)
                vy = clamp(vy, self.vy_prev - dvxy, self.vy_prev + dvxy)
                vz = clamp(vz, self.vz_prev - dvz, self.vz_prev + dvz)
                self.vx_prev, self.vy_prev, self.vz_prev = vx, vy, vz

                # ---- command ----
                if not self._rtl_requested:
                    try:
                        self.v.send_ned_velocity(vx, vy, vz, body_frame=False)
                    except Exception as e:
                        if self.params.verbose:
                            print(f"[{self.name}|sys{self.sysid}] WARN velocity: {e}")

                # ---- HUD ----
                if (loop_now - last_print) > 1.0 and not mode.startswith("HOLD:complete"):
                    last_print = loop_now
                    msg = (f"[{self.name}|sys{self.sysid}] v=({vx:+.2f},{vy:+.2f},{vz:+.2f}) "
                           f"mode={mode} exp={self.experiment_tag} strat={self.strategy_tag}")
                    cell = mine.get("cell")
                    if self.grid and cell:
                        i, j = int(cell[0]), int(cell[1])
                        prog = 100 * self.grid.cell_progress(i, j) if self.grid.in_bounds(i, j) else 0.0
                        msg += f" cell={i},{j} traverse={prog:.0f}%"
                        if self.grid.target:
                            msg += f" tgt={self.grid.target[0]},{self.grid.target[1]}"
                    print(msg)

                time.sleep(dt)
        except KeyboardInterrupt:
            print(f"[{self.name}|sys{self.sysid}] CTRL-C → RTL")
        finally:
            self.close()


# ================= CLI =================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="One swarm drone")
    ap.add_argument("--conn", default="udp:127.0.0.1:14555")
    ap.add_argument("--name", default=None)
    ap.add_argument("--origin-lat", type=float, required=True)
    ap.add_argument("--origin-lon", type=float, required=True)
    ap.add_argument("--geofence-miles", type=float, default=1.0)
    # motion
    ap.add_argument("--alt", type=float, default=20.0)
    ap.add_argument("--vxy-max", type=float, default=30.0)
    ap.add_argument("--vz-max", type=float, default=2.0)
    ap.add_argument("--axy-max", type=float, default=3.0)
    ap.add_argument("--az-max", type=float, default=1.5)
    ap.add_argument("--rate-hz", type=float, default=10.0)
    # comms
    ap.add_argument("--gossip-group", default="239.255.0.1")
    ap.add_argument("--gossip-port", type=int, default=5005)
    ap.add_argument("--comm-grid-radius", type=int, default=2)
    ap.add_argument("--comm-mode", default="full", choices=["full", "tx_only", "off"])
    # labels
    ap.add_argument("--experiment-tag", default="")
    ap.add_argument("--strategy-tag", default="")
    ap.add_argument("--seed", type=int, default=None)
    # mission
    ap.add_argument("--mission", default="grid_frontier",
                    choices=["grid_frontier", "random", "expanding", "lawnmower", "hold"])
    ap.add_argument("--policy", default="random")
    ap.add_argument("--grid-miles", type=float, default=1.0)
    ap.add_argument("--grid-cells", type=int, default=10)
    ap.add_argument("--traverse-frac", type=float, default=0.80)
    ap.add_argument("--map-radius-m", type=float, default=20.0, help="legacy, ignored")
    ap.add_argument("--cell-complete-frac", type=float, default=0.2, help="legacy, ignored")
    # partitions
    ap.add_argument("--partition-scheme", default="none",
                    choices=["none", "halves", "stripes_i", "stripes_j", "quadrants"])
    ap.add_argument("--partition-id", type=int, default=0)
    ap.add_argument("--partition-n", type=int, default=1)
    # end conditions
    ap.add_argument("--end-when", default="never", choices=["never", "all_done", "time"])
    ap.add_argument("--max-seconds", type=float, default=0.0)
    # perception
    ap.add_argument("--config", default="",
                    help="Experiment YAML/JSON; only its perception: section is read here.")
    ap.add_argument("--verbose", action="store_true")
    return ap


def load_perception_cfg(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        from fog.config import load_config
        return load_config(path).get("perception", {}) or {}
    except Exception as e:
        print(f"[agent] WARN: could not read perception config: {e}")
        return {}


def main() -> None:
    args = build_parser().parse_args()

    params = Params(
        alt_target=args.alt, vxy_max=args.vxy_max, vz_max=args.vz_max,
        axy_max=args.axy_max, az_max=args.az_max, rate_hz=args.rate_hz,
        gossip_group=args.gossip_group, gossip_port=args.gossip_port,
        comm_grid_radius=args.comm_grid_radius, comm_mode=args.comm_mode,
        end_when=args.end_when, max_seconds=args.max_seconds, verbose=args.verbose,
    )

    grid_cfg = None
    if args.mission == "grid_frontier":
        grid_cfg = GridCfg(
            lat0=args.origin_lat, lon0=args.origin_lon,
            miles=args.grid_miles, cells=args.grid_cells,
            traverse_frac=args.traverse_frac,
            map_radius_m=args.map_radius_m, complete_frac=args.cell_complete_frac,
        )

    agent = Agent(
        args.name or os.path.basename(args.conn), args.conn,
        args.origin_lat, args.origin_lon,
        geofence_miles=args.geofence_miles, params=params,
        mission=args.mission, grid_cfg=grid_cfg, policy_name=args.policy,
        experiment_tag=args.experiment_tag, strategy_tag=args.strategy_tag,
        seed=args.seed,
        partition_scheme=args.partition_scheme,
        partition_id=args.partition_id, partition_n=args.partition_n,
        perception_cfg=load_perception_cfg(args.config),
    )

    def _stop(*_: Any) -> None:
        agent.stop = True

    signal.signal(signal.SIGINT, _stop)
    try:
        signal.signal(signal.SIGTERM, _stop)
    except Exception:
        pass

    agent.run()


if __name__ == "__main__":
    main()