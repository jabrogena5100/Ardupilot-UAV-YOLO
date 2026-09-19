#!/usr/bin/env python3
"""
fog_tracker.py — ground-station scoreboard, judge, and fire monitor

# TODO: optional Leaflet/Mapbox map overlay (OSM tiles) — defer until we have time.
# - Leaflet: https://leafletjs.com/
# - Mapbox tiles (tokened): https://www.mapbox.com/
# Current viz uses a lightweight canvas; keep as default/offline-safe.

What it does (for new engineers)
--------------------------------
- Listens on gossip for agent packets:
    • VISIT: marks cell visited (preferred)
    • POSE: fallback visit + traverse span tracking (edge-to-edge coverage)
    • (Fire sim is owned by fog; HAZARD_UPDATE is optional and not required.)
- Runs the wildfire simulation locally (FireSim) using fire_model config.
- Prints live status every tick:
    visited X/Y, traverse_done U/Y, fire burning A, burnt B.
- Ends the experiment when the selected condition is met:
    visited | traverse_all | time (see --end-when).
- Writes CSVs under runs/<timestamp>-<tag>/ :
    coverage_t_<tag>.csv   → time series of visited/traverse fractions
    fog_cells_<tag>.csv    → per-cell visitation/traverse metadata
    fire_t_<tag>.csv       → fire burning/burnt counts over time
    experiment_end_<tag>.txt → reason, time, and summary stats

Key inputs expected from agents
--------------------------------
- VISIT: {"type":"VISIT","sys":1,"i":2,"j":7,"t":...,"strat":"coop"}
- POSE:  {"type":"POSE","sys":1,"lat":...,"lon":...,"t":...,"strat":"coop","cell":[i,j]}

CLI example
-----------
python3 fog_tracker.py \\
  --origin-lat 21.2970 --origin-lon -157.8170 \\
  --grid-miles 1.0 --grid-cells 10 --traverse-frac 0.80 \\
  --gossip-group 239.255.0.1 --gossip-port 5005 \\
  --tick-rate-hz 2 --end-when visited --max-seconds 1800
"""

from __future__ import annotations
import argparse, csv, json, math, socket, struct, sys, time, os, shutil, threading, io
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from wildfire_sim import FireSim  # fog now runs the fire simulation itself

# Optional YAML support (for --config .yaml/.yml)
try:
    import yaml  # type: ignore
except Exception:
    yaml = None
# optional headless viz (PNG if matplotlib available; otherwise JSON/HTML only)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ---------------- geometry helpers ----------------
R_EARTH = 6378137.0
M_PER_MILE = 1609.34

def ll_to_ne_m(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float,float]:
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    n = R_EARTH * dlat
    e = R_EARTH * dlon * math.cos(math.radians((lat + lat0) / 2.0))
    return n, e

def clamp(v, lo, hi): return max(lo, min(hi, v))

def cell_index(n: float, e: float, half_side_m: float, cell_m: float, N: int) -> Optional[Tuple[int,int]]:
    # global frame is centered; convert to [0, side]
    x = e + half_side_m
    y = n + half_side_m
    if x < 0.0 or y < 0.0 or x >= 2*half_side_m or y >= 2*half_side_m:
        return None
    i = int(x // cell_m)
    j = int(y // cell_m)
    if 0 <= i < N and 0 <= j < N:
        return (i, j)
    return None

def local_xy_in_cell(n: float, e: float, i: int, j: int, half_side_m: float, cell_m: float) -> Tuple[float,float]:
    x = e + half_side_m
    y = n + half_side_m
    # southwest corner of (i,j)
    e0 = i * cell_m
    n0 = j * cell_m
    le = clamp(x - e0, 0.0, cell_m)  # [0, cell_m]
    ln = clamp(y - n0, 0.0, cell_m)  # [0, cell_m]
    return le, ln

# ---------------- small helpers (NEW) ----------------
def _safe_tag(s: str) -> str:
    if not s: return "exp"
    return "".join(ch if (ch.isalnum() or ch in ("-","_")) else "_" for ch in s)

def _load_config(path: str) -> Dict:
    if not path:
        return {}
    try:
        with open(path, "r") as f:
            txt = f.read()
        ext = os.path.splitext(path)[1].lower()
        if ext in (".json",):
            return json.loads(txt)
        if ext in (".yaml", ".yml"):
            if yaml is None:
                print("[fog] WARNING: YAML config provided but PyYAML not installed; continuing without config fields.")
                return {}
            data = yaml.safe_load(txt)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[fog] WARNING: failed to read config '{path}': {e}")
    return {}

# ---------------- multicast ----------------
class GossipBus:
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

    def recv_all(self, limit=256) -> List[Dict]:
        out=[]
        for _ in range(limit):
            try:
                d,_ = self.rx.recvfrom(8192)
            except BlockingIOError:
                break
            try:
                out.append(json.loads(d.decode()))
            except Exception:
                pass
        return out

    def send(self, payload: Dict):
        try:
            self.tx.sendto(json.dumps(payload).encode(), (self.group, self.port))
        except Exception:
            pass

# ---------------- cell stats ----------------
class Cell:
    __slots__ = ("visited","visited_first_sys","visited_first_strat","visited_first_t",
                 "visits","last_t",
                 "done","min_e","max_e","min_n","max_n")
    def __init__(self):
        # visited = first touch (VISIT or POSE.cell)
        self.visited = False
        self.visited_first_sys = None
        self.visited_first_strat = None
        self.visited_first_t = float("inf")
        self.visits = 0
        self.last_t = 0.0
        # traverse spans (for stats)
        self.done = False
        self.min_e = float("inf"); self.max_e = float("-inf")
        self.min_n = float("inf"); self.max_n = float("-inf")

# ---------------- fog core ----------------
class Fog:
    def __init__(self, lat0:float, lon0:float, miles:float, cells:int, traverse_frac:float):
        self.lat0, self.lon0 = lat0, lon0
        self.side_m = miles * M_PER_MILE
        self.half = self.side_m / 2.0
        self.N = int(cells)
        self.cell_m = self.side_m / self.N
        self.traverse_frac = traverse_frac

        self.grid = [[Cell() for _ in range(self.N)] for __ in range(self.N)]
        self.strat_by_sys: Dict[int,str] = {}
        self.last_cell_by_sys: Dict[int,Tuple[int,int]] = {}
        self.agent_info: Dict[int,Dict] = {}
        self.coverage_ts: List[Tuple[float,float,float]] = []  # (t, visited_frac, traverse_frac)
        # fire state (mirrors agent fire model states: 0=unburned,1=burning,2=burnt)
        self.fire_state: List[List[int]] = [[0 for _ in range(self.N)] for __ in range(self.N)]
        self.fire_heat: List[List[float]] = [[0.0 for _ in range(self.N)] for __ in range(self.N)]
        self.fire_ts: List[Tuple[float,int,int,float,float,int]] = []  # (t, burn_cnt, burnt_cnt, burn_frac, burnt_frac, detected_cnt)
        self.fire_sim: Optional[FireSim] = None
        self._viz_lock = threading.Lock()
        self._viz_png: bytes = b""

    # ---- helpers to mark visited ----
    def _mark_visited(self, i:int, j:int, sysid:Optional[int], t:float, strat:Optional[str]):
        c = self.grid[i][j]
        c.visits += 1
        c.last_t = max(c.last_t, t)
        if not c.visited:
            c.visited = True
            c.visited_first_sys = sysid
            c.visited_first_strat = strat
            c.visited_first_t = t

    # ---- handle POSE (tracks spans; can also fallback-mark visited) ----
    def ingest_pose(self, sysid: Optional[int], t: float, lat: float, lon: float,
                    cell_hint: Optional[List[int]], strat: Optional[str]):
        if sysid is not None and strat:
            self.strat_by_sys[int(sysid)] = strat

        # 1) If agent provided cell indices, trust them to count a visit.
        if isinstance(cell_hint, list) and len(cell_hint) == 2:
            try:
                i_hint = int(cell_hint[0]); j_hint = int(cell_hint[1])
                if 0 <= i_hint < self.N and 0 <= j_hint < self.N:
                    if sysid is not None:
                        self.last_cell_by_sys[int(sysid)] = (i_hint, j_hint)
                        self.agent_info[int(sysid)] = {"cell": [i_hint, j_hint], "last_t": t}
                    self._mark_visited(i_hint, j_hint,
                                       int(sysid) if sysid is not None else None,
                                       t, self.strat_by_sys.get(int(sysid), strat))
            except Exception:
                pass  # ignore bad hints and continue with lat/lon

        # 2) Use lat/lon to update TRAVERSE spans (and also visit fallback).
        n, e = ll_to_ne_m(lat, lon, self.lat0, self.lon0)
        idx = cell_index(n, e, self.half, self.cell_m, self.N)
        if idx is None:
            return
        i, j = idx
        if sysid is not None:
            self.last_cell_by_sys[int(sysid)] = (i, j)
            info = self.agent_info.get(int(sysid), {})
            info.update({"cell": [i, j], "last_t": t, "lat": lat, "lon": lon})
            self.agent_info[int(sysid)] = info

        # fallback: treat presence as a visit too (lightweight mark)
        self._mark_visited(i, j, int(sysid) if sysid is not None else None,
                           t, self.strat_by_sys.get(int(sysid), strat))

        # update TRAVERSE spans
        le, ln = local_xy_in_cell(n, e, i, j, self.half, self.cell_m)
        c = self.grid[i][j]
        if le < c.min_e: c.min_e = le
        if le > c.max_e: c.max_e = le
        if ln < c.min_n: c.min_n = ln
        if ln > c.max_n: c.max_n = ln

        if not c.done:
            span_e = c.max_e - c.min_e
            span_n = c.max_n - c.min_n
            if max(span_e, span_n) >= self.traverse_frac * self.cell_m:
                c.done = True

    # ---- handle VISIT (preferred for "visited" semantics) ----
    def ingest_visit(self, sysid: Optional[int], t: float, i: int, j: int, strat: Optional[str]):
        if not (0 <= i < self.N and 0 <= j < self.N):
            return
        if sysid is not None and strat:
            self.strat_by_sys[int(sysid)] = strat
        if sysid is not None:
            self.last_cell_by_sys[int(sysid)] = (i, j)
            info = self.agent_info.get(int(sysid), {})
            info.update({"cell": [i, j], "last_t": t})
            self.agent_info[int(sysid)] = info
        self._mark_visited(i, j, int(sysid) if sysid is not None else None, t, self.strat_by_sys.get(int(sysid), strat))

    # ---- snapshot coverage ----
    def snapshot(self, t_now: float) -> Tuple[float,float,int,int]:
        visited_cnt = 0
        traverse_cnt = 0
        for i in range(self.N):
            for j in range(self.N):
                c = self.grid[i][j]
                # Treat TRAVERSE-done as visited for coverage accounting
                if c.visited or c.done:
                    visited_cnt += 1
                if c.done:
                    traverse_cnt += 1
        visited_frac = visited_cnt / (self.N * self.N)
        traverse_frac = traverse_cnt / (self.N * self.N)
        self.coverage_ts.append((t_now, visited_frac, traverse_frac))
        return visited_frac, traverse_frac, visited_cnt, traverse_cnt

    # ---- end checks ----
    def all_visited(self) -> bool:
        for i in range(self.N):
            for j in range(self.N):
                if not self.grid[i][j].visited:
                    return False
        return True

    def all_traverse_done(self) -> bool:
        for i in range(self.N):
            for j in range(self.N):
                if not self.grid[i][j].done:
                    return False
        return True

    # ---- fire ingest / snapshot ----
    def ingest_fire(self, i: int, j: int, state: Optional[int], heat: Optional[float], t: float):
        if not (0 <= i < self.N and 0 <= j < self.N):
            return
        s = 0 if state is None else int(state)
        h = 0.0 if heat is None else float(heat)
        self.fire_state[i][j] = max(0, min(2, s))
        self.fire_heat[i][j] = clamp(h, 0.0, 1.0)

    def sync_fire_from_sim(self, sim: FireSim):
        snap = sim.snapshot()
        for i in range(self.N):
            for j in range(self.N):
                self.fire_state[i][j] = snap[i][j]
                self.fire_heat[i][j] = clamp(sim.cells[i][j].heat, 0.0, 1.0)

    def fire_snapshot(self, t_now: float) -> Tuple[int,int,float,float]:
        burn = 0
        burnt = 0
        total = self.N * self.N
        for i in range(self.N):
            for j in range(self.N):
                s = self.fire_state[i][j]
                if s == 1:
                    burn += 1
                elif s == 2:
                    burnt += 1
        burn_frac = burn / total
        burnt_frac = burnt / total
        detected = self.fire_detections()
        self.fire_ts.append((t_now, burn, burnt, burn_frac, burnt_frac, detected))
        return burn, burnt, burn_frac, burnt_frac, detected

    def burning_cells(self) -> List[Tuple[int,int]]:
        cells = []
        for i in range(self.N):
            for j in range(self.N):
                if self.fire_state[i][j] == 1:
                    cells.append((i, j))
        return cells

    def fire_detections(self) -> int:
        """Count drones whose last known cell is within Chebyshev 1 of a burning cell."""
        burning = self.burning_cells()
        if not burning or not self.last_cell_by_sys:
            return 0
        detected = 0
        for _, (ci, cj) in self.last_cell_by_sys.items():
            for bi, bj in burning:
                if max(abs(ci - bi), abs(cj - bj)) <= 1:
                    detected += 1
                    break
        return detected

    def burnt_cells(self) -> List[Tuple[int,int]]:
        cells = []
        for i in range(self.N):
            for j in range(self.N):
                if self.fire_state[i][j] == 2:
                    cells.append((i, j))
        return cells

    def current_stats(self, t: Optional[float] = None, meta: Optional[Dict] = None) -> Dict:
        """Lightweight snapshot for viz/JSON (no side-effects)."""
        # coverage stats
        visited_cnt = 0
        traverse_cnt = 0
        total = self.N * self.N
        for i in range(self.N):
            for j in range(self.N):
                c = self.grid[i][j]
                if c.visited or c.done:
                    visited_cnt += 1
                if c.done:
                    traverse_cnt += 1
        visited_frac = visited_cnt / total
        traverse_frac = traverse_cnt / total
        # fire stats
        burn_cnt = len(self.burning_cells())
        burnt_cnt = len(self.burnt_cells())
        burn_frac = burn_cnt / total
        burnt_frac = burnt_cnt / total
        detected = self.fire_detections()
        # agent info with age
        agents = {}
        now = time.time()
        for sid, info in self.agent_info.items():
            age = now - info.get("last_t", now)
            agents[int(sid)] = {
                "cell": info.get("cell"),
                "last_t": info.get("last_t"),
                "age_s": round(age, 1),
                "lat": info.get("lat"),
                "lon": info.get("lon"),
            }
        out = {
            "N": self.N,
            "visited_cnt": visited_cnt,
            "traverse_cnt": traverse_cnt,
            "visited_frac": visited_frac,
            "traverse_frac": traverse_frac,
            "burning_cnt": burn_cnt,
            "burnt_cnt": burnt_cnt,
            "burning_frac": burn_frac,
            "burnt_frac": burnt_frac,
            "detected_cnt": detected,
            "agents": agents,
            "burning": self.burning_cells(),
            "burnt": self.burnt_cells(),
        }
        if t is not None:
            out["t"] = t
        if meta:
            out.update(meta)
        return out

    # ---- viz helpers ----
    def render_png(self) -> Optional[bytes]:
        """Render current grid/fog/fire state to a PNG (headless)."""
        if plt is None:
            return None
        fig, ax = plt.subplots(figsize=(6,6), dpi=80)
        ax.set_xlim(0, self.N); ax.set_ylim(0, self.N)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("k")

        # base colors: visited/done
        vis = [[self.grid[i][j].visited or self.grid[i][j].done for j in range(self.N)] for i in range(self.N)]
        for i in range(self.N):
            for j in range(self.N):
                if vis[i][j]:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, color="#1f77b4", alpha=0.25))
        # fire overlay
        for i in range(self.N):
            for j in range(self.N):
                s = self.fire_state[i][j]
                if s == 1:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, color="orangered", alpha=0.6))
                elif s == 2:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, color="dimgray", alpha=0.4))
        # agent markers
        for sysid, (ci, cj) in self.last_cell_by_sys.items():
            ax.plot(ci+0.5, cj+0.5, marker="o", color="lime", markersize=6)
            ax.text(ci+0.5, cj+0.5, f"{sysid}", color="black", ha="center", va="center", fontsize=6)
        ax.invert_yaxis()
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return buf.getvalue()

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin-lat", type=float, required=True)
    ap.add_argument("--origin-lon", type=float, required=True)
    ap.add_argument("--grid-miles", type=float, default=1.0)
    ap.add_argument("--grid-cells", type=int, default=10)
    ap.add_argument("--traverse-frac", type=float, default=0.80)
    ap.add_argument("--gossip-group", default="239.255.0.1")
    ap.add_argument("--gossip-port", type=int, default=5005)
    ap.add_argument("--tick-rate-hz", type=float, default=2.0)
    ap.add_argument("--end-when", choices=["visited","traverse_all","time","all_done"], default="visited",
                    help="'visited' ends on first-touch coverage; 'traverse_all' means every cell spans threshold; "
                         "'all_done' kept as alias of 'traverse_all' for backward-compat; 'time' ends at max-seconds.")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--experiment-tag", default="")
    # NEW: optional config path (JSON or YAML)
    ap.add_argument("--config", default="", help="Path to experiment config (.json, .yaml/.yml). Results folder will be created under the config's directory.")
    ap.add_argument("--viz-port", type=int, default=0, help="If >0, serve a lightweight PNG/JSON view on this TCP port (headless).")
    args = ap.parse_args()

    # map legacy alias
    if args.end_when == "all_done":
        args.end_when = "traverse_all"

    # --- NEW: load config & set results directory/filenames ---
    cfg = _load_config(args.config) if args.config else {}
    cfg_path_abs = os.path.abspath(args.config) if args.config else ""
    cfg_dir = os.path.dirname(cfg_path_abs) if cfg_path_abs else os.getcwd()

    # resolve tag preference: CLI > config.experiment_tag > config filename stem > "exp"
    cfg_tag = ""
    try:
        cfg_tag = str(cfg.get("experiment_tag", "") or "")
    except Exception:
        cfg_tag = ""
    fname_tag = os.path.splitext(os.path.basename(cfg_path_abs))[0] if cfg_path_abs else ""
    tag = _safe_tag(args.experiment_tag.strip() or cfg_tag or fname_tag or "exp")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(cfg_dir, "runs", f"{stamp}-{tag}")
    os.makedirs(outdir, exist_ok=True)

    cov_path   = os.path.join(outdir, f"coverage_t_{tag}.csv")
    cells_path = os.path.join(outdir, f"fog_cells_{tag}.csv")
    end_path   = os.path.join(outdir, f"experiment_end_{tag}.txt")
    fire_path  = os.path.join(outdir, f"fire_t_{tag}.csv")
    args_json  = os.path.join(outdir, "args.json")
    if cfg_path_abs and os.path.exists(cfg_path_abs):
        try:
            shutil.copyfile(cfg_path_abs, os.path.join(outdir, os.path.basename(cfg_path_abs)))
        except Exception as e:
            print(f"[fog] WARNING: could not copy config into results folder: {e}")

    # stash effective args for reproducibility
    eff = {
        "origin_lat": args.origin_lat,
        "origin_lon": args.origin_lon,
        "grid_miles": args.grid_miles,
        "grid_cells": args.grid_cells,
        "traverse_frac": args.traverse_frac,
        "gossip_group": args.gossip_group,
        "gossip_port": args.gossip_port,
        "tick_rate_hz": args.tick_rate_hz,
        "end_when": args.end_when,
        "max_seconds": args.max_seconds,
        "experiment_tag_effective": tag,
        "stamp": stamp,
        "outdir": outdir,
        "config_path": cfg_path_abs,
    }
    try:
        with open(args_json, "w") as f:
            json.dump(eff, f, indent=2)
    except Exception as e:
        print(f"[fog] WARNING: failed writing args.json: {e}")

    # pull a couple optional fields from config to embed in CSVs (robust to missing keys)
    cfg_mission  = ""
    cfg_strategy = ""
    cfg_agents_n = 0
    try:
        if isinstance(cfg.get("agents", None), list):
            cfg_agents_n = len(cfg["agents"])
            # If all agents share same mission/strategy_tag, pick from the first
            if cfg_agents_n > 0 and isinstance(cfg["agents"][0], dict):
                cfg_mission  = str(cfg["agents"][0].get("mission", "") or "")
                cfg_strategy = str(cfg["agents"][0].get("strategy_tag", "") or "")
    except Exception:
        pass

    bus = GossipBus(args.gossip_group, args.gossip_port)
    fog = Fog(args.origin_lat, args.origin_lon, miles=args.grid_miles, cells=args.grid_cells,
              traverse_frac=args.traverse_frac)
    fire_sim = None
    next_fire_step = 0.0
    fire_cfg = {}
    viz_meta = {"tag": tag, "end_when": args.end_when, "grid_cells": fog.N}
    try:
        fire_cfg = cfg.get("fire_model", {}) if isinstance(cfg, dict) else {}
        if fire_cfg.get("enabled", False):
            fire_sim = FireSim.from_config(cfg, fog.N)
            fog.fire_sim = fire_sim
            init = fire_cfg.get("initial_ignitions") or []
            fire_sim.ignite_many(init)
            fog.sync_fire_from_sim(fire_sim)
            print(f"[fog] FireSim enabled (cells={fog.N}x{fog.N}, dt={fire_sim.dt_seconds}s)")
            viz_meta["fire"] = {
                "enabled": True,
                "neighborhood": fire_sim.neighborhood,
                "p_ignite": fire_sim.p_ignite,
                "burn_steps": fire_sim.burn_steps,
                "burn_time_dist": fire_sim.burn_time_dist,
                "dt_seconds": fire_sim.dt_seconds,
            }
        else:
            viz_meta["fire"] = {"enabled": False}
    except Exception as e:
        print(f"[fog] WARNING: FireSim init failed: {e}")
        fire_sim = None

    t0 = time.time()
    next_tick = 0.0
    # optional viz server (PNG if matplotlib, otherwise JSON+HTML canvas)
    viz_thread = None
    viz_stop = threading.Event()
    if args.viz_port > 0:
        def _viz_loop():
            from http.server import HTTPServer, BaseHTTPRequestHandler
            HTML = """
<!doctype html><html><head><meta charset='utf-8'><title>Fog Grid</title></head>
<body style="margin:0;background:#111;color:#eee;font-family:monospace;display:flex;flex-direction:row;">
<div style="flex:1;padding:6px;">
  <div id="meta" style="padding-bottom:6px;"></div>
  <div id="stats" style="padding-bottom:6px;"></div>
  <canvas id="c" width="600" height="600"></canvas>
</div>
<div id="agents" style="width:260px;padding:8px;border-left:1px solid #333;overflow-y:auto;"></div>
<script>
const c = document.getElementById('c'); const ctx = c.getContext('2d');
const stats = document.getElementById('stats'); const meta = document.getElementById('meta'); const agentsDiv = document.getElementById('agents');
function pct(x){return (x*100).toFixed(1)+'%';}
function draw(state){
  const N = state.N || 10; const w = c.width/N, h = c.height/N;
  ctx.fillStyle='#000'; ctx.fillRect(0,0,c.width,c.height);
  ctx.strokeStyle='rgba(255,255,255,0.15)'; ctx.lineWidth=1;
  for(let i=0;i<=N;i++){ ctx.beginPath(); ctx.moveTo(i*w,0); ctx.lineTo(i*w,c.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0,i*h); ctx.lineTo(c.width,i*h); ctx.stroke(); }
  ctx.fillStyle='orangered'; (state.burning||[]).forEach(([i,j])=>{ctx.fillRect(i*w, j*h, w, h);});
  ctx.fillStyle='dimgray'; (state.burnt||[]).forEach(([i,j])=>{ctx.fillRect(i*w, j*h, w, h);});
  ctx.fillStyle='rgba(31,119,180,0.2)'; (state.visited||[]).forEach(([i,j])=>{ctx.fillRect(i*w, j*h, w, h);});
  ctx.fillStyle='lime';
  for (const [id, info] of Object.entries(state.agents||{})){
    const cell = info.cell||[0,0]; const [i,j]=cell;
    ctx.beginPath(); ctx.arc((i+0.5)*w,(j+0.5)*h, Math.max(3,w*0.2), 0, 2*Math.PI);
    ctx.fill(); ctx.fillText(id,(i+0.35)*w,(j+0.7)*h);
  }
  meta.innerText = `tag ${state.tag||''} | grid ${state.grid_cells||N}x${state.grid_cells||N} | end_when ${state.end_when||''} `
    + `| fire ${state.fire && state.fire.enabled ? 'on' : 'off'}`
    + (state.fire && state.fire.enabled ? ` p=${(state.fire.p_ignite||0).toFixed(2)} burn_steps=${state.fire.burn_steps||''} dist=${state.fire.burn_time_dist||''}` : '');
  stats.innerText = `visited ${state.visited_cnt}/${N*N} (${pct(state.visited_frac||0)}) | `
    + `traverse ${state.traverse_cnt}/${N*N} (${pct(state.traverse_frac||0)}) | `
    + `fire burning ${state.burning_cnt||0} (${pct(state.burning_frac||0)}) `
    + `burnt ${state.burnt_cnt||0} (${pct(state.burnt_frac||0)}) `
    + `detected ${state.detected_cnt||0} | t=${(state.t||0).toFixed(1)}s`;
  let html = '<div style="font-weight:bold;margin-bottom:4px;">Agents</div>';
  for (const [id, info] of Object.entries(state.agents||{})){
    const cell = info.cell ? `${info.cell[0]},${info.cell[1]}` : 'n/a';
    const age = info.age_s !== undefined ? `${info.age_s}s` : '';
    const lat = info.lat !== undefined ? `lat ${(+info.lat).toFixed(5)}` : '';
    const lon = info.lon !== undefined ? `lon ${(+info.lon).toFixed(5)}` : '';
    html += `<div style="margin-bottom:6px;border-bottom:1px solid #333;padding-bottom:4px;">`
         + `<div style="font-weight:bold;">Drone ${id}</div>`
         + `<div>cell ${cell}</div>`
         + `<div>last ${age}</div>`
         + `<div>${lat} ${lon}</div>`
         + `</div>`;
  }
  agentsDiv.innerHTML = html;
}
async function loop(){
  try{ const r=await fetch('state.json?'+Math.random()); const s=await r.json(); draw(s);}catch(e){}
  requestAnimationFrame(loop);
}
loop();
</script>
</body></html>
"""
            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path.startswith("/") and ("state.json" not in self.path and "grid.png" not in self.path):
                        body = HTML.encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    if self.path.startswith("/grid.png"):
                        png = fog.render_png()
                        if png is None:
                            self.send_response(503); self.end_headers(); return
                        self.send_response(200)
                        self.send_header("Content-Type", "image/png")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        self.wfile.write(png)
                        return
                    if self.path.startswith("/state.json"):
                        body = json.dumps(fog.current_stats(t=time.time()-t0, meta=viz_meta)).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    self.send_response(404); self.end_headers()
                def log_message(self, *a, **kw): return
            srv = HTTPServer(("0.0.0.0", args.viz_port), Handler)
            srv.timeout = 1.0
            print(f"[fog] viz server on http://0.0.0.0:{args.viz_port} (HTML canvas + state.json; grid.png if matplotlib)")
            while not viz_stop.is_set():
                srv.handle_request()
            srv.server_close()
        viz_thread = threading.Thread(target=_viz_loop, daemon=True)
        viz_thread.start()

    print(f"[fog] started • grid={args.grid_cells}x{args.grid_cells} over {args.grid_miles:.2f} mi "
          f"• end_when={args.end_when} • tag='{tag}' • out='{outdir}'")

    # CSV buffers (NEW: add tag + a few config columns at the end)
    cov_rows = [("t","visited_frac","traverse_frac","tag","mission","strategy","agents_n")]
    cells_header = ("i","j","visited","visited_first_sys","visited_first_strat","visited_first_t",
                    "visits","last_t","traverse_done","tag","mission","strategy","agents_n")
    fire_rows = [("t","burning_cnt","burnt_cnt","burning_frac","burnt_frac","detected_cnt","tag","mission","strategy","agents_n")]

    def broadcast_end(reason: str):
        bus.send({"type":"EXPERIMENT_END", "reason":reason, "t": time.time(), "exp": args.experiment_tag})
        print(f"[fog] >>> EXPERIMENT_END reason='{reason}'")

    while True:
        # ingest all messages
        for m in bus.recv_all():
            typ = m.get("type")
            if typ == "VISIT":
                fog.ingest_visit(m.get("sys"), m.get("t", time.time()),
                                 int(m.get("i", -1)), int(m.get("j", -1)),
                                 m.get("strat"))
            elif typ == "POSE":
                lat, lon = m.get("lat"), m.get("lon")
                if lat is not None and lon is not None:
                    fog.ingest_pose(m.get("sys"), m.get("t", time.time()),
                                    float(lat), float(lon), m.get("cell"), m.get("strat"))

        # advance fire sim locally (oracle)
        if fire_sim is not None and time.time() >= next_fire_step:
            changed = fire_sim.step()
            fog.sync_fire_from_sim(fire_sim)
            next_fire_step = time.time() + max(0.001, fire_sim.dt_seconds)
            # broadcast changes so agents can map fire (state 1=burning, 2=burnt)
            for i, j, state, heat in changed:
                bus.send({
                    "type": "HAZARD_UPDATE",
                    "hazard": "fire",
                    "i": i,
                    "j": j,
                    "state": state,
                    "heat": round(heat, 2),
                    "t": time.time()
                })

        # periodic tick
        now = time.time() - t0
        if now >= next_tick:
            vfrac, tfrac, vcnt, tcnt = fog.snapshot(now)
            burn_cnt, burnt_cnt, burn_frac, burnt_frac, detected_cnt = fog.fire_snapshot(now)
            # NEW: include tag + a couple cfg fields in each row
            cov_rows.append((f"{now:.2f}", f"{vfrac:.6f}", f"{tfrac:.6f}", tag, cfg_mission, cfg_strategy, str(cfg_agents_n)))
            fire_rows.append((f"{now:.2f}", str(burn_cnt), str(burnt_cnt), f"{burn_frac:.6f}", f"{burnt_frac:.6f}", str(detected_cnt), tag, cfg_mission, cfg_strategy, str(cfg_agents_n)))

            print(f"[fog] visited {vcnt}/{fog.N*fog.N} ({100*vfrac:.1f}%) | "
                  f"traverse_done {tcnt}/{fog.N*fog.N} ({100*tfrac:.1f}%) | "
                  f"fire burning {burn_cnt} ({100*burn_frac:.1f}%) burnt {burnt_cnt} ({100*burnt_frac:.1f}%) "
                  f"detected {detected_cnt}")

            # --- END CHECKS (robust) ---
            total = fog.N * fog.N
            should_end = False
            reason = ""

            # Prefer the explicit counts we just printed
            if args.end_when == "visited" and vcnt >= total:
                should_end, reason = True, "full_coverage_visited"
            elif args.end_when in ("traverse_all", "all_done") and tcnt >= total:
                should_end, reason = True, "full_coverage_traverse"
            elif args.end_when == "time" and args.max_seconds > 0.0 and now >= args.max_seconds:
                should_end, reason = True, "timeout"

            # Safety guard: if not using 'time', trust visited count anyway
            if not should_end and args.end_when != "time" and vcnt >= total:
                should_end, reason = True, "full_coverage_visited(count_guard)"

            if should_end:
                # Broadcast END a few times for reliability
                for _ in range(3):
                    broadcast_end(reason)
                    time.sleep(0.05)

                # dump CSVs (to tagged, timestamped folder)
                try:
                    with open(cov_path,"w",newline="") as f:
                        csv.writer(f).writerows(cov_rows)
                except Exception as e:
                    print(f"[fog] ERROR writing {cov_path}: {e}")
                try:
                    with open(fire_path,"w",newline="") as f:
                        csv.writer(f).writerows(fire_rows)
                except Exception as e:
                    print(f"[fog] ERROR writing {fire_path}: {e}")

                try:
                    with open(cells_path,"w",newline="") as f:
                        w = csv.writer(f); w.writerow(cells_header)
                        for i in range(fog.N):
                            for j in range(fog.N):
                                c = fog.grid[i][j]
                                w.writerow((i, j,
                                            int(c.visited), c.visited_first_sys, c.visited_first_strat,
                                            (0.0 if c.visited_first_t==float('inf') else round(c.visited_first_t - t0, 2)),
                                            c.visits, round(c.last_t - t0, 2) if c.last_t else 0.0,
                                            int(c.done),
                                            tag, cfg_mission, cfg_strategy, str(cfg_agents_n)))
                except Exception as e:
                    print(f"[fog] ERROR writing {cells_path}: {e}")

                try:
                    with open(end_path,"w") as f:
                        f.write(f"reason={reason}\n")
                        f.write(f"time_s={now:.2f}\n")
                        f.write(f"visited_frac={vfrac:.6f}\n")
                        f.write(f"traverse_frac={tfrac:.6f}\n")
                        f.write(f"tag={tag}\n")
                        if cfg_mission:  f.write(f"mission={cfg_mission}\n")
                        if cfg_strategy: f.write(f"strategy={cfg_strategy}\n")
                        f.write(f"agents_n={cfg_agents_n}\n")
                except Exception as e:
                    print(f"[fog] ERROR writing {end_path}: {e}")

                print(f"[fog] wrote {cov_path}, {cells_path}, {end_path} — exiting.")

                # Hard-exit to avoid any lingering threads/handlers keeping the process alive
                viz_stop.set()
                os._exit(0)
        time.sleep(0.002)

if __name__ == "__main__":
    main()
