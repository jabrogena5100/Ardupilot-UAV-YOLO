#!/usr/bin/env python3
"""
swarm_agent.py — TRAVERSE coverage + in-range MAP (visited & done) + VISIT sharing + rich debug

Key fixes in this build:
 - Always TX POSE/VISIT/MAP even when comm_mode=='off' (so fog can see progress).
 - Clean stop on local completion when end_when=='all_done' to avoid console spam.
"""

from __future__ import annotations
import argparse, json, math, os, random, signal, socket, struct, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

from custom_vehicle import CustomVehicle
import swarm_strategies as strat
from wildfire_sim import FireSim

# --------------- geo helpers ---------------
R_EARTH = 6378137.0
M_PER_MILE = 1609.344

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def latlon_to_ne(lat_deg, lon_deg, lat0_deg, lon0_deg):
    lat = math.radians(lat_deg); lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg); lon0 = math.radians(lon0_deg)
    dn = (lat - lat0) * R_EARTH
    de = (lon - lon0) * R_EARTH * math.cos(lat0)
    return dn, de

def cheby_dist(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

# --------------- gossip sockets ---------------
class Gossip:
    def __init__(self, group="239.255.0.1", port=5005, iface="0.0.0.0"):
        self.group, self.port = group, port
        # RX
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.rx.bind(("", port))
        except OSError:
            self.rx.bind((iface, port))
        mreq = struct.pack("=4sl", socket.inet_aton(group), socket.INADDR_ANY)
        self.rx.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.rx.setblocking(False)
        # TX
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, struct.pack('@i', 1))

    def send(self, payload: Dict):
        try:
            self.tx.sendto(json.dumps(payload).encode(), (self.group, self.port))
        except Exception:
            pass

    def recv_all(self, max_packets=128) -> List[Dict]:
        out=[]
        for _ in range(max_packets):
            try:
                d,_ = self.rx.recvfrom(8192)
            except BlockingIOError:
                break
            try:
                out.append(json.loads(d.decode()))
            except Exception:
                pass
        return out

# --------------- grid mission (TRAVERSE) ---------------
@dataclass
class GridCfg:
    lat0: float
    lon0: float
    miles: float = 1.0           # side-length in miles
    cells: int = 10              # NxN
    traverse_frac: float = 0.80  # edge-to-edge threshold of cell width
    # legacy (kept for CLI compat only):
    map_radius_m: float = 20.0
    complete_frac: float = 0.2

    @property
    def side_m(self) -> float:
        return self.miles * M_PER_MILE
    @property
    def cell_m(self) -> float:
        return self.side_m / self.cells

class GridMission:
    """TRAVERSE: cell (i,j) DONE when max(span_e,span_n) ≥ traverse_frac*cell_m."""
    def __init__(self, cfg: GridCfg):
        self.cfg = cfg
        N = cfg.cells
        self.min_e = [[float("inf")]*N for _ in range(N)]
        self.max_e = [[float("-inf")]*N for _ in range(N)]
        self.min_n = [[float("inf")]*N for _ in range(N)]
        self.max_n = [[float("-inf")]*N for _ in range(N)]
        self.done  = [[False]*N for _ in range(N)]
        self.claimed: Dict[Tuple[int,int], int] = {}   # (i,j) -> sysid
        self.prev_pos: Optional[Tuple[float,float]] = None
        self.target: Optional[Tuple[int,int]] = None

        # latched traverse run state
        self.run: Optional[Dict] = None   # {'i','j','axis','side','goal':(n,e),'last_flip_t'}

        # tunables
        self.flip_dist_m = 2.0
        self.flip_min_sec = 1.0

        # epsilon for edge snapping (agent side)
        self.edge_eps_m = 0.5

    # ---- cell helpers ----
    def _pos_to_cell_inclusive(self, dn: float, de: float) -> Tuple[int,int]:
        """Inclusive & clamped cell index with small epsilon to keep boundary points inside."""
        half = self.cfg.side_m / 2.0
        cell = self.cfg.cell_m
        # Expand by epsilon and clamp
        dn = clamp(dn, -half - self.edge_eps_m, half + self.edge_eps_m)
        de = clamp(de, -half - self.edge_eps_m, half + self.edge_eps_m)
        # Convert to indices (0..N), snap N→N-1
        u = (de + half) / cell
        v = (dn + half) / cell
        i = int(u); j = int(v)
        if abs(u - self.cfg.cells) < 1e-9: i = self.cfg.cells - 1
        if abs(v - self.cfg.cells) < 1e-9: j = self.cfg.cells - 1
        # Clamp
        i = max(0, min(self.cfg.cells - 1, i))
        j = max(0, min(self.cfg.cells - 1, j))
        return i, j

    def pos_to_cell(self, lat, lon) -> Tuple[int,int]:
        dn, de = latlon_to_ne(lat, lon, self.cfg.lat0, self.cfg.lon0)
        return self._pos_to_cell_inclusive(dn, de)

    def in_bounds(self, i, j) -> bool:
        return 0 <= i < self.cfg.cells and 0 <= j < self.cfg.cells

    def cell_center_ne(self, i, j) -> Tuple[float,float]:
        half = self.cfg.side_m / 2.0
        cn = -half + j*self.cfg.cell_m + 0.5*self.cfg.cell_m
        ce = -half + i*self.cfg.cell_m + 0.5*self.cfg.cell_m
        return cn, ce

    def local_east_north_in_cell(self, dn: float, de: float, i: int, j: int) -> Tuple[float,float]:
        half = self.cfg.side_m / 2.0
        e0 = -half + i*self.cfg.cell_m
        n0 = -half + j*self.cfg.cell_m
        local_e = clamp(de - e0, 0.0, self.cfg.cell_m)
        local_n = clamp(dn - n0, 0.0, self.cfg.cell_m)
        return local_e, local_n

    def cell_progress(self, i, j) -> float:
        if not self.in_bounds(i,j):
            return 0.0
        span_e = self.max_e[i][j] - self.min_e[i][j]
        span_n = self.max_n[i][j] - self.min_n[i][j]
        span = max(0.0, span_e, span_n)
        return clamp(span / self.cfg.cell_m, 0.0, 1.0)

    # ---- update spans from position ----
    def update_progress(self, lat, lon):
        dn, de = latlon_to_ne(lat, lon, self.cfg.lat0, self.cfg.lon0)
        i, j = self._pos_to_cell_inclusive(dn, de)
        if not self.in_bounds(i, j):
            self.prev_pos = (dn, de); return
        le, ln = self.local_east_north_in_cell(dn, de, i, j)
        if le < self.min_e[i][j]: self.min_e[i][j] = le
        if le > self.max_e[i][j]: self.max_e[i][j] = le
        if ln < self.min_n[i][j]: self.min_n[i][j] = ln
        if ln > self.max_n[i][j]: self.max_n[i][j] = ln
        if not self.done[i][j]:
            span_e = self.max_e[i][j] - self.min_e[i][j]
            span_n = self.max_n[i][j] - self.min_n[i][j]
            if max(span_e, span_n) >= self.cfg.traverse_frac * self.cfg.cell_m:
                self.done[i][j] = True
        self.prev_pos = (dn, de)

    # ---- nearest frontier ----
    def nearest_frontier(self, i0, j0,
                         allowed: Optional[Iterable[Tuple[int,int]]] = None,
                         avoid: Optional[Iterable[Tuple[int,int]]] = None) -> Optional[Tuple[int,int]]:
        N = self.cfg.cells
        allowed_set = set(allowed) if allowed is not None else None
        avoid_set   = set(avoid) if avoid is not None else set()
        for r in range(N):
            i_min, i_max = max(0, i0-r), min(N-1, i0+r)
            j_min, j_max = max(0, j0-r), min(N-1, j0+r)
            # ring edges
            for i in range(i_min, i_max+1):
                for j in (j_min, j_max):
                    if self.in_bounds(i,j) and not self.done[i][j]:
                        if (allowed_set is None or (i,j) in allowed_set) and (i,j) not in avoid_set:
                            if (i,j) not in self.claimed:
                                return (i,j)
            for j in range(j_min+1, j_max):
                for i in (i_min, i_max):
                    if self.in_bounds(i,j) and not self.done[i][j]:
                        if (allowed_set is None or (i,j) in allowed_set) and (i,j) not in avoid_set:
                            if (i,j) not in self.claimed:
                                return (i,j)
        return None

    # ---- latched traverse goals ----
    def _edge_goal_from_latch(self, i:int, j:int, axis:str, side:int) -> Tuple[float,float]:
        half = self.cfg.side_m / 2.0
        cell = self.cfg.cell_m
        k = 0.10 if side==0 else 0.90
        if axis == 'E':
            goal_e = -half + i*cell + k*cell
            goal_n = -half + j*cell + 0.5*cell
        else:
            goal_n = -half + j*cell + k*cell
            goal_e = -half + i*cell + 0.5*cell
        return goal_n, goal_e

    def _make_run(self, ti:int, tj:int, dn:float, de:float, now:float, vp=None):
        cell = self.cfg.cell_m
        le, ln = self.local_east_north_in_cell(dn, de, ti, tj)
        span_e = self.max_e[ti][tj] - self.min_e[ti][tj]
        span_n = self.max_n[ti][tj] - self.min_n[ti][tj]
        axis = 'E' if span_e <= span_n else 'N'
        side = 1 if (le <= 0.5*cell if axis=='E' else ln <= 0.5*cell) else 0
        goal_n, goal_e = self._edge_goal_from_latch(ti, tj, axis, side)
        self.run = {'i':ti,'j':tj,'axis':axis,'side':side,'goal':(goal_n,goal_e),'last_flip_t':now}
        if vp: vp(f"TRAVERSE: latch axis={'E-W' if axis=='E' else 'N-S'} side={'high' if side==1 else 'low'} → edge goal set")

    def ensure_run_and_goal(self, ti:int, tj:int, dn:float, de:float, now:float, vp=None) -> Tuple[float,float]:
        if self.run is None or self.run.get('i')!=ti or self.run.get('j')!=tj:
            self._make_run(ti, tj, dn, de, now, vp)
        goal_n, goal_e = self.run['goal']
        dist = math.hypot(goal_n - dn, goal_e - de)
        if dist <= self.flip_dist_m and not self.done[ti][tj]:
            if (now - self.run['last_flip_t']) >= self.flip_min_sec:
                self.run['side'] = 1 - int(self.run['side'])
                goal_n, goal_e = self._edge_goal_from_latch(ti, tj, self.run['axis'], self.run['side'])
                self.run['goal'] = (goal_n, goal_e)
                self.run['last_flip_t'] = now
                if vp: vp(f"TRAVERSE: flip side (dist={dist:.1f}m)")
        return goal_n, goal_e

    def clear_run(self):
        self.run = None

    def all_done(self) -> bool:
        N = self.cfg.cells
        for i in range(N):
            for j in range(N):
                if not self.done[i][j]:
                    return False
        return True

# --------------- agent ---------------
@dataclass
class Params:
    alt_target: float = 20.0
    vxy_max: float = 30.0     # fast coverage
    vz_max: float = 2.0
    axy_max: float = 3.0
    az_max: float = 1.5
    rate_hz: float = 10.0
    gossip_group: str = "239.255.0.1"
    gossip_port: int = 5005
    comm_grid_radius: int = 2
    comm_mode: str = "full"   # full | tx_only | off
    end_when: str = "never"   # never | all_done | time
    max_seconds: float = 0.0
    verbose: bool = False

class Agent:
    def __init__(self, name: str, conn: str, origin_lat: float, origin_lon: float,
                 geofence_miles: float, params: Params,
                 mission: str = "grid_frontier",
                 grid_cfg: Optional[GridCfg] = None,
                 policy_name: str = "random",
                 experiment_tag: str = "",
                 strategy_tag: str = "",
                 seed: Optional[int] = None,
                 partition_scheme: str = "none",
                 partition_id: int = 0,
                 partition_n: int = 1,
                 fire_cfg: Optional[Dict] = None):
        self.name = name
        self.params = params
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.half_side_m = (geofence_miles * M_PER_MILE) / 2.0
        self.v = CustomVehicle(conn, autorequest_rates=True)
        self.gossip = Gossip(params.gossip_group, params.gossip_port)
        self.stop = False

        self.experiment_tag = experiment_tag
        self.strategy_tag = strategy_tag or policy_name

        if seed is not None:
            random.seed(seed)
        self.policy = strat.make_policy(policy_name, alt_target=params.alt_target,
                                        vxy_max=params.vxy_max, vz_max=params.vz_max,
                                        lat0=origin_lat, lon0=origin_lon)

        self.mission = mission.lower()
        self.grid: Optional[GridMission] = None
        if self.mission == "grid_frontier":
            if grid_cfg is None:
                grid_cfg = GridCfg(lat0=origin_lat, lon0=origin_lon)
            self.grid = GridMission(grid_cfg)

        self.partition_scheme = partition_scheme
        self.partition_id = max(0, partition_id)
        self.partition_n = max(1, partition_n)

        self.vx_prev = self.vy_prev = self.vz_prev = 0.0
        self.sysid = None
        self.t0 = time.time()

        # VISIT-sharing (TTL avoid)
        self.cur_cell: Optional[Tuple[int,int]] = None
        self.prev_cell: Optional[Tuple[int,int]] = None
        self.last_visit_emit_t = 0.0
        self.visit_emit_period = 0.5
        self.neighbor_visit_ttl_s = 5.0
        self.neighbor_visits: Dict[Tuple[int,int], float] = {}

        # MAP-gossip (persistent union)
        self.N = self.grid.cfg.cells if self.grid else 10
        self._vis_bits = 0          # visited bitmap (union)
        self._done_bits = 0         # done bitmap (union)
        self._map_emit_period = 2.0
        self._last_map_emit = 0.0
        self._merged_avoid: set[tuple[int,int]] = set()
        self.hazard_cells: Dict[Tuple[int,int], float] = {}  # (i,j)->expire_t
        self.hazard_ttl_s = 10.0
        self.fire_cells: Dict[Tuple[int,int], Tuple[int,float]] = {}    # (state, expire_t)
        self.fire_ttl_s = 8.0      # burning TTL
        self.fire_burnt_ttl_s = 20.0
        self.fire_target: Optional[Tuple[int,int]] = None
        self._rtl_requested = False

        # last-cell nudge
        self._nudged_once_for_idx: Optional[int] = None  # avoid repeat VISIT spam

        # completion-announcement guard
        self._complete_printed = False

        # fire sim (optional)
        self.fire_cfg = fire_cfg or {}
        self.fire = None
        self._next_fire_step = 0.0
        try:
            enabled = bool(self.fire_cfg.get("enabled", False))
            if enabled and self.grid is not None:
                self.fire = FireSim.from_config({"fire_model": self.fire_cfg}, self.N)
                self.hazard_ttl_s = float(self.fire_cfg.get("hazard_ttl_s", 10.0))
                init = self.fire_cfg.get("initial_ignitions") or []
                self.fire.ignite_many(init)
                self._vp("FireSim enabled")
        except Exception as e:
            print(f"[{self.name}|sys{self.sysid}] WARN: FireSim init failed: {e}")
            self.fire = None

        # verbose printer
        self._vp = (lambda s: print(f"[{self.name}|sys{self.sysid}] {s}")) if self.params.verbose else (lambda s: None)

    # ---- bit helpers ----
    def _bit_index(self, i:int, j:int) -> int:
        return j * self.N + i  # row-major

    def _set_vis(self, i:int, j:int):
        if 0 <= i < self.N and 0 <= j < self.N:
            self._vis_bits |= (1 << self._bit_index(i,j))

    def _set_done(self, i:int, j:int):
        if 0 <= i < self.N and 0 <= j < self.N:
            self._done_bits |= (1 << self._bit_index(i,j))
            self._merged_avoid.add((i,j))

    def _bits_to_hex(self, bits:int) -> str:
        width = (self.N*self.N + 3)//4
        return f"{bits:0{width}x}"

    def _hex_to_bits(self, hx:str) -> Optional[int]:
        try:
            return int(hx, 16)
        except Exception:
            return None

    def _merge_vis_hex(self, hx:str):
        other = self._hex_to_bits(hx)
        if other is None: return
        merged = self._vis_bits | other
        if merged == self._vis_bits: return
        self._vis_bits = merged

    def _merge_done_hex(self, hx:str):
        other = self._hex_to_bits(hx)
        if other is None: return
        merged = self._done_bits | other
        if merged == self._done_bits: return
        self._done_bits = merged
        # also mark our local grid as done for those cells
        if self.grid:
            for j in range(self.N):
                for i in range(self.N):
                    if (merged >> self._bit_index(i,j)) & 1:
                        if 0 <= i < self.grid.cfg.cells and 0 <= j < self.grid.cfg.cells:
                            self.grid.done[i][j] = True
                            self._merged_avoid.add((i,j))

    def _count_bits(self, bits: int) -> int:
        return bits.bit_count() if hasattr(bits, "bit_count") else bin(bits).count("1")

    def _find_single_remaining_cell(self, bits: int) -> Optional[Tuple[int,int,int]]:
        """Return (i,j,idx) if exactly one bit is 0 in the first N*N bits; else None."""
        total = self.N * self.N
        mask = (1 << total) - 1
        b = bits & mask
        if self._count_bits(b) != total - 1:
            return None
        for idx in range(total):
            if ((b >> idx) & 1) == 0:
                i = idx % self.N
                j = idx // self.N
                return (i, j, idx)
        return None

    # ---- setup / teardown ----
    def ensure_ready(self):
        self.v.wait_for_healthy(timeout=20.0, min_gps_fix=3)
        self.v.set_mode("GUIDED"); self.v.wait_for_mode("GUIDED", timeout=10.0)
        self.v.arm(True); self.v.wait_for_armed(True, timeout=12.0)
        self.v.takeoff(self.params.alt_target)
        self.v.wait_for(lambda s: (s.get("alt_rel_m") or 0.0) >= 0.9*self.params.alt_target, timeout=60.0)
        try:
            self.sysid = int(getattr(self.v, "target_system", 0) or 0)
        except Exception:
            self.sysid = 0
        self._vp("READY: GUIDED/ARMED/ALT ok")

    def close(self):    
        """
        Best-effort safe shutdown at the end of an experiment.

        Current behavior:
        - Send a brief "brake" to zero out velocity.
        - Command RTL so the vehicle returns to its home/launch position.
        - Wait briefly for RTL mode acknowledgement.
        - Close the MAVLink connection.
        """
        try:
            # Zero out velocity before handing control back to the autopilot.
            self.v.brake()
            # Command Return-To-Launch on experiment end.
            try:
                self.v.set_mode("RTL")
                self.v.wait_for_mode("RTL", timeout=5.0)
            except Exception:
                pass
            try:
                self.v.rtl()
            except Exception:
                pass
        except Exception:
            # Don't let shutdown errors crash the agent process.
            pass
        self.v.close()

    # ---- partitions ----
    def allowed_cells(self) -> Optional[List[Tuple[int,int]]]:
        if not self.grid or self.partition_scheme == "none":
            return None
        N = self.grid.cfg.cells
        mid = N//2
        out: List[Tuple[int,int]] = []
        for i in range(N):
            for j in range(N):
                keep = True
                if self.partition_scheme == "halves":
                    keep = (i < mid and self.partition_id % 2 == 0) or (i >= mid and self.partition_id % 2 == 1)
                elif self.partition_scheme == "stripes_i":
                    keep = (i % self.partition_n) == (self.partition_id % self.partition_n)
                elif self.partition_scheme == "stripes_j":
                    keep = (j % self.partition_n) == (self.partition_id % self.partition_n)
                elif self.partition_scheme == "quadrants":
                    q = (0 if i < mid else 1) + (0 if j < mid else 2)
                    keep = (q == (self.partition_id % 4))
                if keep: out.append((i,j))
        return out

    # ---- gossip helpers ----
    def state_packet(self) -> Dict:
        st = self.v.get_state()
        lat, lon = st.get("lat"), st.get("lon")
        cell = None
        if self.grid and lat is not None and lon is not None:
            i,j = self.grid.pos_to_cell(lat, lon)
            if self.grid.in_bounds(i,j):
                cell = [i,j]
                if self.cur_cell != (i, j):
                    self.prev_cell = self.cur_cell
                self.cur_cell = (i,j)
                self._set_vis(i, j)  # mark visited persistently
        claim = None
        if self.grid and self.grid.target is not None:
            claim = [int(self.grid.target[0]), int(self.grid.target[1])]
        return {
            "type": "POSE",
            "name": self.name,
            "sys": int(self.sysid or 0),
            "lat": lat, "lon": lon,
            "alt": st.get("alt_rel_m"),
            "vx": st.get("vx"), "vy": st.get("vy"), "vz": st.get("vz"),
            "t": time.time(),
            "mission": self.mission,
            "cell": cell,
            "claim": claim,
            "exp": self.experiment_tag,
            "strat": self.strategy_tag
        }

    def _emit_visit_now(self, now: float, override_cell: Optional[Tuple[int,int]] = None):
        """Immediate VISIT emit (ignores visit_emit_period). Always TX (fog needs this)."""
        ci, cj = (override_cell if override_cell is not None else (self.cur_cell or (None, None)))
        if ci is None or cj is None:
            return
        self.gossip.send({
            "type":"VISIT","sys": int(self.sysid or 0),
            "i": int(ci), "j": int(cj),
            "t": now, "strat": self.strategy_tag
        })
        self.last_visit_emit_t = now
        self._vp(f"VISIT: immediate emit {(ci,cj)}")

    def emit_visit_if_due(self, now: float):
        if not self.cur_cell:
            return
        if (now - self.last_visit_emit_t) >= self.visit_emit_period:
            self._emit_visit_now(now)

    def emit_map_if_due(self, now: float, cell_for_range: Optional[List[int]]):
        if (now - self._last_map_emit) >= self._map_emit_period:
            pkt = {
                "type": "MAP",
                "sys": int(self.sysid or 0),
                "t": now,
                "N": self.N,
                "vis": self._bits_to_hex(self._vis_bits),
                "done": self._bits_to_hex(self._done_bits),
                "strat": self.strategy_tag
            }
            if cell_for_range:
                pkt["cell"] = cell_for_range
            self.gossip.send(pkt)
            self._last_map_emit = now
            self._vp("MAP: emit bitmap (vis+done)")

    def prune_neighbor_visits(self, now: float):
        expired = [k for k,exp in self.neighbor_visits.items() if exp <= now]
        for k in expired: self.neighbor_visits.pop(k, None)

    def prune_hazards(self, now: float):
        expired = [k for k, exp in self.hazard_cells.items() if exp <= now]
        for k in expired:
            self.hazard_cells.pop(k, None)
        expired_f = [k for k, (_, exp) in self.fire_cells.items() if exp <= now]
        for k in expired_f:
            self.fire_cells.pop(k, None)
        if self.fire_target and self.fire_target not in self.fire_cells:
            self.fire_target = None

    def recv_neighbors(self, mine: Dict) -> List[Dict]:
        # Only called when RX is allowed by comm_mode
        msgs = self.gossip.recv_all()
        out = []
        my_cell = tuple(mine.get("cell")) if mine.get("cell") else None
        now = time.time()
        for m in msgs:
            if m.get("type") == "EXPERIMENT_END":
                print(f"[{self.name}|sys{self.sysid}] STOP: EXPERIMENT_END reason={m.get('reason','')}")
                self.stop = True; continue
            if m.get("sys") == mine.get("sys"):
                continue
            # VISIT (TTL avoidance)
            if m.get("type") == "VISIT":
                i, j = m.get("i"), m.get("j")
                if my_cell and i is not None and j is not None:
                    if cheby_dist(my_cell, (int(i), int(j))) <= self.params.comm_grid_radius:
                        self.neighbor_visits[(int(i), int(j))] = now + self.neighbor_visit_ttl_s
                        self._vp(f"VISIT: accept {(int(i),int(j))}")
                continue
            # MAP (bitmap union)
            if m.get("type") == "MAP":
                if my_cell and isinstance(m.get("N"), int) and m.get("vis") is not None:
                    if m["N"] != self.N:
                        continue
                    if m.get("cell") and isinstance(m["cell"], list) and len(m["cell"])==2:
                        oi, oj = int(m["cell"][0]), int(m["cell"][1])
                        if cheby_dist(my_cell, (oi, oj)) > self.params.comm_grid_radius:
                            continue
                    # merge visited & done
                    self._merge_vis_hex(str(m.get("vis")))
                    if m.get("done") is not None:
                        self._merge_done_hex(str(m.get("done")))
                    self._vp("MAP: merged vis/done from neighbor")
                continue
            if m.get("type") == "HAZARD_UPDATE":
                if m.get("hazard") == "fire":
                    i, j = int(m.get("i", -1)), int(m.get("j", -1))
                    if 0 <= i < self.N and 0 <= j < self.N:
                        # use tighter TTL for fire detections
                        state = int(m.get("state", 1))
                        ttl = self.fire_burnt_ttl_s if state == 2 else self.fire_ttl_s
                        self.fire_cells[(i, j)] = (state, now + ttl)
                continue
            # POSE/CLAIM considered in-range only
            if self.mission == "grid_frontier" and my_cell and m.get("cell"):
                ci,cj = my_cell; oi,oj = int(m["cell"][0]), int(m["cell"][1])
                if abs(ci-oi) <= self.params.comm_grid_radius and abs(cj-oj) <= self.params.comm_grid_radius:
                    out.append(m)
        self.prune_neighbor_visits(now)
        self.prune_hazards(now)
        return out

    # ---- geofence ----
    def geofence_push(self, vn: float, ve: float, lat: Optional[float], lon: Optional[float]) -> Tuple[float,float]:
        if lat is None or lon is None:
            return vn, ve
        en, ee = latlon_to_ne(lat, lon, self.origin_lat, self.origin_lon)
        half = self.half_side_m
        push_n = 0.0; push_e = 0.0
        margin = 10.0
        if en >  half:  push_n -= (en - half)
        if en < -half:  push_n += (-half - en)
        if ee >  half:  push_e -= (ee - half)
        if ee < -half:  push_e += (-half - ee)
        if push_n != 0.0 or push_e != 0.0:
            d = math.hypot(push_n, push_e) + 1e-6
            vn_new = -3.0 * (push_n/d)
            ve_new = -3.0 * (push_e/d)
            if self.params.verbose:
                print(f"[{self.name}|sys{self.sysid}] GEOFENCE: pushback (v→{vn_new:+.2f},{ve_new:+.2f})")
            return vn_new, ve_new
        else:
            def taper(pos):
                return clamp((half - abs(pos)) / margin, 0.1, 1.0) if abs(pos) > (half - margin) else 1.0
            vn *= taper(en); ve *= taper(ee)
        return vn, ve

    def hard_recentering(self, lat: Optional[float], lon: Optional[float]) -> Optional[tuple[float,float,str]]:
        if lat is None or lon is None:
            return None
        en, ee = latlon_to_ne(lat, lon, self.origin_lat, self.origin_lon)
        half_geo = self.half_side_m
        half_grid = self.grid.cfg.side_m / 2.0 if self.grid else None
        margin = 20.0
        outside_geo = (abs(en) > half_geo + margin) or (abs(ee) > half_geo + margin)
        outside_grid = False
        if half_grid is not None:
            outside_grid = (abs(en) > half_grid + margin) or (abs(ee) > half_grid + margin)
        if outside_geo or outside_grid:
            dn = -en; de = -ee
            d = math.hypot(dn, de) + 1e-6
            speed = max(15.0, self.params.vxy_max)
            vn = (dn / d) * speed
            ve = (de / d) * speed
            return (vn, ve, "RECENTER")
        return None

    def reset_mission_state(self):
        if self.grid:
            self.grid.target = None
            self.grid.clear_run()
            self.grid.claimed.clear()
        self.neighbor_visits.clear()  # short-lived avoid

    # ---- mission logic ----
    def step_grid(self, mine: Dict, neigh: List[Dict]) -> Tuple[float,float,float,str]:
        assert self.grid is not None
        lat, lon = mine.get("lat"), mine.get("lon")
        now = time.time()

        # update spans
        if lat is not None and lon is not None:
            before_done = False
            if self.cur_cell:
                i,j = self.cur_cell
                before_done = self.grid.done[i][j]
            self.grid.update_progress(lat, lon)
            if self.cur_cell:
                i,j = self.cur_cell
                if self.grid.done[i][j] and not before_done:
                    self._set_done(i, j)
                    self._emit_visit_now(now)

        # Hard recenter if outside geofence or grid
        rc = self.hard_recentering(lat, lon)
        if rc is not None:
            vn, ve, tag = rc
            alt = mine.get("alt") or self.params.alt_target
            vz = clamp(-0.4 * (self.params.alt_target - alt), -self.params.vz_max, self.params.vz_max)
            if lat is not None and lon is not None:
                en, ee = latlon_to_ne(lat, lon, self.origin_lat, self.origin_lon)
                if math.hypot(en, ee) <= 5.0:
                    self.reset_mission_state()
                    ve = max(ve, 0.5)
            return vn, ve, vz, tag

        # merge neighbor claims (in-range) only when RX enabled
        if self.params.comm_mode == "full":
            for m in neigh:
                c = m.get("claim")
                if c and len(c) == 2:
                    self.grid.claimed[(int(c[0]), int(c[1]))] = int(m.get("sys") or 0)

        # drop target if done or stolen
        if self.grid.target is not None:
            i, j = self.grid.target
            if self.grid.done[i][j]:
                self._vp(f"DONE: cell ({i},{j}) traverse ≥ {int(self.grid.cfg.traverse_frac*100)}%")
                self.grid.claimed.pop((i, j), None)
                self.grid.target = None
                self.grid.clear_run()
            elif self.params.comm_mode == "full":
                owner = self.grid.claimed.get((i, j))
                if owner is not None and owner != self.sysid:
                    self._vp(f"TARGET: stolen by sys={owner} → drop")
                    self.grid.target = None
                    self.grid.clear_run()

        # choose fire target if any detections (priority)
        if self.fire_cells and lat is not None and lon is not None:
            ci, cj = self.grid.pos_to_cell(lat, lon)
            burning = [p for p, (st, _) in self.fire_cells.items() if st == 1]
            burnt = [p for p, (st, _) in self.fire_cells.items() if st == 2]
            candidate = burning or burnt
            candidate = [p for p in candidate if p not in self._merged_avoid]
            if candidate:
                if self.fire_target not in candidate:
                    self.fire_target = min(candidate, key=lambda p: cheby_dist((ci, cj), p))
                    self._vp(f"FIRE: target {self.fire_target} state={self.fire_cells[self.fire_target][0]}")
                ti, tj = self.fire_target
                goal_n, goal_e = self.grid.cell_center_ne(ti, tj)
                me_n, me_e = latlon_to_ne(lat, lon, self.grid.cfg.lat0, self.grid.cfg.lon0)
                dn, de = goal_n - me_n, goal_e - me_e
                d = math.hypot(dn, de)
                speed = self.params.vxy_max
                vmag = min(speed, d)
                vn = (dn / d) * vmag if d > 1e-3 else 0.0
                ve = (de / d) * vmag if d > 1e-3 else 0.0
                alt = mine.get("alt") or self.params.alt_target
                vz = clamp(-0.4 * (self.params.alt_target - alt), -self.params.vz_max, self.params.vz_max)
                mode = f"FIRE→{ti},{tj}"
                # if reached, drop it
                if d < 1.0:
                    self.fire_cells.pop((ti, tj), None)
                    self.fire_target = None
                return vn, ve, vz, mode

        # choose target
        if self.grid.target is None and lat is not None and lon is not None:
            ci, cj = self.grid.pos_to_cell(lat, lon)
            allowed = self.allowed_cells()
            avoid = set(self.neighbor_visits.keys()).union(self._merged_avoid).union(set(self.hazard_cells.keys()))
            self.grid.target = self.grid.nearest_frontier(ci, cj, allowed, list(avoid))
            if self.grid.target is not None:
                self.grid.claimed[self.grid.target] = int(self.sysid or 0)
                self._vp(f"TARGET: pick {self.grid.target} (nearest-frontier)")
            else:
                tgt = self.grid.nearest_frontier(ci, cj, allowed=allowed, avoid=None)
                if tgt is None:
                    tgt = self.grid.nearest_frontier(ci, cj, allowed=None, avoid=None)
                if tgt is not None:
                    self.grid.target = tgt
                    self.grid.claimed[self.grid.target] = int(self.sysid or 0)
                    self._vp(f"TARGET: forced pick {self.grid.target} (stuck-guard)")

        # local completion (informative only; run() will actually stop the loop)
        if self.params.end_when == "all_done" and self.grid.all_done():
            return 0.0, 0.0, 0.0, "HOLD:complete"

        # fallback if no target
        if self.grid.target is None or lat is None or lon is None:
            vn, ve, vz, mode = self.policy.step(mine, [])
            return vn, ve, vz, f"SEARCH:fallback({mode})"

        # We have a target
        ti, tj = self.grid.target
        me_n, me_e = latlon_to_ne(lat, lon, self.grid.cfg.lat0, self.grid.cfg.lon0)

        # approach center first if outside target cell
        ci, cj = self.grid.pos_to_cell(lat, lon)
        if not (ci == ti and cj == tj):
            goal_n, goal_e = self.grid.cell_center_ne(ti, tj)
            self._vp(f"APPROACH: center of {self.grid.target}")
        else:
            if self.grid.done[ti][tj]:
                self._vp(f"DONE (race): cell {self.grid.target} already complete → retarget")
                self.grid.claimed.pop((ti, tj), None)
                self.grid.target = None
                self.grid.clear_run()
                vn, ve, vz, mode = self.policy.step(mine, [])
                return vn, ve, vz, f"SEARCH:retarget"
            goal_n, goal_e = self.grid.ensure_run_and_goal(ti, tj, me_n, me_e, time.time(), self._vp)

        # vector to goal
        dn, de = goal_n - me_n, goal_e - me_e
        d = math.hypot(dn, de)
        speed = self.params.vxy_max
        vmag = min(speed, d)
        if d < 1e-3:
            vn = ve = 0.0
        else:
            vn = (dn / d) * vmag
            ve = (de / d) * vmag

        # altitude hold
        alt = mine.get("alt") or self.params.alt_target
        vz = clamp(-0.4 * (self.params.alt_target - alt), -self.params.vz_max, self.params.vz_max)

        mode = f"TRAVERSE→{ti},{tj}" if (ci==ti and cj==tj) else f"APPROACH→{ti},{tj}"
        return vn, ve, vz, mode

    def _maybe_nudge_last_cell(self, now: float):
        union_bits = self._vis_bits | self._done_bits
        rem = self._find_single_remaining_cell(union_bits)
        if rem is None:
            return
        ri, rj, ridx = rem
        if self.grid:
            if self.grid.target != (ri, rj):
                self.grid.target = (ri, rj)
                self.grid.claimed[(ri, rj)] = int(self.sysid or 0)
                self.neighbor_visits.clear()
                self._vp(f"NUDGE: forcing target to remaining cell {(ri,rj)}")
        if self.cur_cell is not None:
            ci, cj = self.cur_cell
            if cheby_dist((ci, cj), (ri, rj)) <= 1:
                if self._nudged_once_for_idx != ridx:
                    self._emit_visit_now(now, override_cell=(ri, rj))
                    self._nudged_once_for_idx = ridx
                    self._vp(f"NUDGE: emitted VISIT for remaining cell {(ri,rj)}")

    # ---- fire sim (optional) ----
    def _update_fire(self, now: float):
        if self.fire is None:
            return
        if now < self._next_fire_step:
            return
        self._next_fire_step = now + self.fire.dt_seconds
        changed = self.fire.step()
        # refresh hazard TTL for burning cells
        for i in range(self.N):
            for j in range(self.N):
                if self.fire.cells[i][j].state == 1:
                    self.hazard_cells[(i, j)] = now + self.hazard_ttl_s
        # broadcast any burning cells so others can avoid
        for msg in self.fire.hazard_messages(sysid=self.sysid):
            self.gossip.send(msg)

    def _request_rtl(self):
        """One-shot RTL request to stop motion and return home."""
        if self._rtl_requested:
            return
        self._rtl_requested = True
        try:
            self.v.brake()
        except Exception:
            pass
        try:
            self.v.set_mode("RTL")
        except Exception:
            pass
        try:
            self.v.wait_for_mode("RTL", timeout=5.0)
        except Exception:
            pass
        try:
            self.v.rtl()
        except Exception:
            pass

    # ---- main loop ----
    def run(self):
        self.ensure_ready()
        dt = 1.0 / max(5.0, self.params.rate_hz)
        last_print = 0.0
        try:
            while not self.stop:
                # time-based end
                if self.params.end_when == "time" and self.params.max_seconds > 0 and (time.time()-self.t0) >= self.params.max_seconds:
                    print(f"[{self.name}|sys{self.sysid}] STOP: max time reached")
                    self._request_rtl()
                    break

                loop_now = time.time()
                self._update_fire(loop_now)

                mine = self.state_packet()
                if self.stop:
                    self._request_rtl()
                    break

                # If we just entered a new cell, emit a VISIT immediately
                if self.cur_cell and self.cur_cell != self.prev_cell:
                    self._emit_visit_now(mine["t"])

                # Always TX so fog can see progress (even when comm_mode=='off')
                self.gossip.send(mine)                  # POSE
                self.emit_visit_if_due(mine["t"])       # VISIT (periodic backup)
                self.emit_map_if_due(mine["t"], mine.get("cell"))  # MAP (vis+done)

                # RX only when comm_mode == "full"
                neigh = self.recv_neighbors(mine) if self.params.comm_mode == "full" else []

                # Endgame helper: if exactly one cell remains, nudge it
                self._maybe_nudge_last_cell(mine["t"])
                self.prune_hazards(mine["t"])

                # behavior
                if self.mission == "grid_frontier" and self.grid is not None:
                    vx, vy, vz, mode = self.step_grid(mine, neigh)
                else:
                    vx, vy, vz, mode = self.policy.step(mine, neigh)

                # If local all_done and configured to end, stop cleanly (prevents console spam)
                if self.params.end_when == "all_done" and self.grid and self.grid.all_done():
                    if not self._complete_printed:
                        print(f"[{self.name}|sys{self.sysid}] STOP: all cells complete (local view)")
                        self._complete_printed = True
                    self._request_rtl()
                    break

                # geofence
                vx, vy = self.geofence_push(vx, vy, mine.get("lat"), mine.get("lon"))

                # accel limiting
                dvxy = self.params.axy_max * dt
                dvz  = self.params.az_max * dt
                vx_clamped = clamp(vx, self.vx_prev - dvxy, self.vx_prev + dvxy)
                vy_clamped = clamp(vy, self.vy_prev - dvxy, self.vy_prev + dvxy)
                vz_clamped = clamp(vz, self.vz_prev - dvz,  self.vz_prev + dvz)
                vx, vy, vz = vx_clamped, vy_clamped, vz_clamped
                self.vx_prev, self.vy_prev, self.vz_prev = vx, vy, vz

                # send
                try:
                    # Use body-frame velocity, same style as test_velocity.py.
                    # BODY_NED: x = forward, y = right, z = down.
                    # Our vx,vy are N/E, but as long as yaw is ~north after takeoff,
                    # body-forward ≈ north and body-right ≈ east, so this is fine.
                    if not self._rtl_requested:
                        self.v.send_ned_velocity(vx, vy, vz, body_frame=False)
                except Exception as e:
                    if self.params.verbose:
                        print(f"[{self.name}|sys{self.sysid}] WARN send_body_velocity: {e}")
                    # continue loop; transient MAVLink hiccup

                # HUD (skip spam if we're holding complete)
                now = time.time()
                if now - last_print > 1.0 and not mode.startswith("HOLD:complete"):
                    last_print = now
                    cell = mine.get("cell")
                    msg = f"[{self.name}|sys{self.sysid}] v=({vx:+.2f},{vy:+.2f},{vz:+.2f}) mode={mode} exp={self.experiment_tag} strat={self.strategy_tag}"
                    if self.grid and cell:
                        i,j = int(cell[0]), int(cell[1])
                        prog = 100*self.grid.cell_progress(i,j) if self.grid.in_bounds(i,j) else 0.0
                        msg += f" cell={i},{j} traverse={prog:.0f}%"
                        if self.grid.target:
                            ti,tj = self.grid.target
                            msg += f" tgt={ti},{tj}"
                    print(msg)

                time.sleep(dt)
                if self.stop:
                    print(f"[{self.name}|sys{self.sysid}] STOP: EXPERIMENT_END honored")
                    self._request_rtl()
                    break
        except KeyboardInterrupt:
            print(f"[{self.name}|sys{self.sysid}] CTRL-C received → RTL")
        finally:
            self.close()

# --------------- CLI ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="udp:127.0.0.1:14555")
    ap.add_argument("--name", default=None)
    # origin & geofence
    ap.add_argument("--origin-lat", type=float, required=True,
                    help="Latitude of origin (center of square geofence)")
    ap.add_argument("--origin-lon", type=float, required=True,
                    help="Longitude of origin (center of square geofence)")
    ap.add_argument("--geofence-miles", type=float, default=1.0,
                    help="Square side-length in miles (default 1.0 → ±0.5 mile)")
    # motion / vehicle
    ap.add_argument("--alt", type=float, default=20.0)
    ap.add_argument("--vxy-max", type=float, default=30.0)
    ap.add_argument("--vz-max", type=float, default=2.0)
    ap.add_argument("--axy-max", type=float, default=3.0)
    ap.add_argument("--az-max", type=float, default=1.5)
    ap.add_argument("--rate-hz", type=float, default=10.0)
    # gossip & comms
    ap.add_argument("--gossip-group", default="239.255.0.1")
    ap.add_argument("--gossip-port", type=int, default=5005)
    ap.add_argument("--comm-grid-radius", type=int, default=2,
                    help="Neighborhood radius in cells (Chebyshev) for in-range comms")
    ap.add_argument("--comm-mode", default="full", choices=["full","tx_only","off"],
                    help="full: TX+RX, tx_only: send-only, off: RX off but TX always ON (to fog)")
    # experiment labels
    ap.add_argument("--experiment-tag", default="")
    ap.add_argument("--strategy-tag", default="")
    ap.add_argument("--seed", type=int, default=None)
    # mission / strategy
    ap.add_argument("--mission", default="grid_frontier",
                    choices=["grid_frontier","random","expanding","lawnmower","hold"])
    ap.add_argument("--policy", default="random",
                    help="Fallback/alt policy name for non-grid missions")
    # grid config (TRAVERSE)
    ap.add_argument("--grid-miles", type=float, default=1.0)
    ap.add_argument("--grid-cells", type=int, default=10)
    ap.add_argument("--traverse-frac", type=float, default=0.80,
                    help="Edge-to-edge threshold as a fraction of cell width")
    # legacy (ignored by TRAVERSE, kept for CLI compat)
    ap.add_argument("--map-radius-m", type=float, default=20.0)
    ap.add_argument("--cell-complete-frac", type=float, default=0.2)
    # partitions
    ap.add_argument("--partition-scheme", default="none", choices=["none","halves","stripes_i","stripes_j","quadrants"])
    ap.add_argument("--partition-id", type=int, default=0)
    ap.add_argument("--partition-n", type=int, default=1)
    # end conditions
    ap.add_argument("--end-when", default="never", choices=["never","all_done","time"])
    ap.add_argument("--max-seconds", type=float, default=0.0)
    # fire sim (optional)
    ap.add_argument("--fire-config-json", default="", help="Fire model config JSON (matches fire_model section)")
    # verbosity
    ap.add_argument("--verbose", action="store_true", help="Extra debug prints")

    args = ap.parse_args()

    P = Params(alt_target=args.alt, vxy_max=args.vxy_max, vz_max=args.vz_max,
               axy_max=args.axy_max, az_max=args.az_max, rate_hz=args.rate_hz,
               gossip_group=args.gossip_group, gossip_port=args.gossip_port,
               comm_grid_radius=args.comm_grid_radius, comm_mode=args.comm_mode,
               end_when=args.end_when, max_seconds=args.max_seconds,
               verbose=args.verbose)

    grid_cfg = None
    if args.mission == "grid_frontier":
        grid_cfg = GridCfg(lat0=args.origin_lat, lon0=args.origin_lon,
                           miles=args.grid_miles, cells=args.grid_cells,
                           traverse_frac=args.traverse_frac,
                           map_radius_m=args.map_radius_m,
                           complete_frac=args.cell_complete_frac)

    name = args.name or os.path.basename(args.conn)
    fire_cfg = {}
    if args.fire_config_json:
        try:
            fire_cfg = json.loads(args.fire_config_json)
        except Exception as e:
            print(f"[agent] WARNING: could not parse fire-config-json: {e}")

    agent = Agent(name, args.conn, args.origin_lat, args.origin_lon,
                  geofence_miles=args.geofence_miles, params=P,
                  mission=args.mission, grid_cfg=grid_cfg, policy_name=args.policy,
                  experiment_tag=args.experiment_tag, strategy_tag=args.strategy_tag,
                  seed=args.seed, fire_cfg=fire_cfg,
                  partition_scheme=args.partition_scheme, partition_id=args.partition_id,
                  partition_n=args.partition_n)

    def _sigint(*_):
        agent.stop = True
    signal.signal(signal.SIGINT, _sigint)
    def _sigterm(*_):
        agent.stop = True
    try:
        signal.signal(signal.SIGTERM, _sigterm)
    except Exception:
        pass

    agent.run()

if __name__ == "__main__":
    main()
