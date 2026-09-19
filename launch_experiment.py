#!/usr/bin/env python3
"""
launch_experiment.py — start everything from ONE config file.

What this gives you
-------------------
• A single YAML/JSON config that sets origin, geofence, grid, comms, timing, and agent list.
• Spawns N swarm agents (with their per-agent strategy/partition/comm-mode) + optional fog_tracker.
• No more giant command lines. One file → one command.

Usage
-----
python3 launch_experiment.py --config experiment.yaml
# or
python3 launch_experiment.py --config experiment.json

Config schema (YAML)
--------------------
# experiment.yaml
experiment_tag: demo-001
origin:
  lat: 21.2970
  lon: -157.8170
geofence_miles: 1.0

comms:
  group: 239.255.0.1
  port: 5005
  comm_grid_radius: 2

grid:
  miles: 1.0
  cells: 10
  traverse_frac: 0.80        # TRAVERSE model threshold (fraction of cell width)
  # legacy (ignored by TRAVERSE model but kept for agents’ CLI compatibility):
  map_radius_m: 20
  cell_complete_frac: 0.2

timing:
  end_when: all_done          # never | all_done | time
  max_seconds: 0              # used when end_when=time

fog_tracker:
  enable: true
  tick_rate_hz: 2
  max_seconds: 600            # also used when end_when=time

# Default vehicle params (can be overridden per-agent)
vehicle_defaults:
  alt: 20
  vxy_max: 6
  vz_max: 1.5
  axy_max: 2
  az_max: 1
  rate_hz: 10

# Agents (each entry spawns one swarm_agent.py instance)
agents:
  - name: A
    conn: udp:127.0.0.1:14555
    mission: grid_frontier          # grid_frontier | random | expanding | lawnmower | hold
    policy: random                  # fallback/alt policy
    strategy_tag: coop
    seed: 42
    comm_mode: full                 # full | tx_only | off
    partition:
      scheme: none                  # none | halves | stripes_i | stripes_j | quadrants
      n: 1
      id: 0
"""

from __future__ import annotations
import argparse, json, os, subprocess, sys, time, signal
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

HERE = Path(__file__).resolve().parent
PY = sys.executable or "python3"

SA = str(HERE / "core" / "swarm_agent.py")
FT = str(HERE / "core" / "fog_tracker.py")

# ------------- helpers -------------

def load_config(path: Path):
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is not installed. Run: pip install pyyaml, or use JSON config.")
        return yaml.safe_load(text)
    return json.loads(text)


def ensure_defaults(cfg: dict):
    cfg.setdefault("experiment_tag", "exp")
    cfg.setdefault("origin", {})
    cfg.setdefault("geofence_miles", 1.0)
    cfg.setdefault("comms", {})
    cfg.setdefault("grid", {})
    cfg.setdefault("timing", {})
    cfg.setdefault("fog_tracker", {})
    cfg.setdefault("fire_model", {})
    cfg.setdefault("vehicle_defaults", {})
    cfg.setdefault("agents", [])

    # nested defaults
    origin = cfg["origin"]
    if not {"lat","lon"} <= origin.keys():
        raise ValueError("origin.lat and origin.lon are required")

    comms = cfg["comms"]
    comms.setdefault("group", "239.255.0.1")
    comms.setdefault("port", 5005)
    comms.setdefault("comm_grid_radius", 2)

    fire_model = cfg["fire_model"]
    fire_model.setdefault("enabled", False)

    grid = cfg["grid"]
    grid.setdefault("miles", cfg.get("geofence_miles", 1.0))
    grid.setdefault("cells", 10)
    grid.setdefault("auto_cell_m", None)  # optional: compute cells from desired cell size (m)
    # TRAVERSE knob
    grid.setdefault("traverse_frac", 0.80)
    # legacy (kept so agents' CLI accepts them even if unused by TRAVERSE)
    grid.setdefault("map_radius_m", 20)
    grid.setdefault("cell_complete_frac", 0.2)

    timing = cfg["timing"]
    timing.setdefault("end_when", "all_done")
    timing.setdefault("max_seconds", 0)

    vdef = cfg["vehicle_defaults"]
    vdef.setdefault("alt", 20)
    vdef.setdefault("vxy_max", 6)
    vdef.setdefault("vz_max", 1.5)
    vdef.setdefault("axy_max", 2)
    vdef.setdefault("az_max", 1)
    vdef.setdefault("rate_hz", 10)

    fog = cfg["fog_tracker"]
    fog.setdefault("enable", True)
    fog.setdefault("tick_rate_hz", 2)
    fog.setdefault("max_seconds", 0)
    # fog end_when default lives in fog_tracker.py; optional here:
    fog.setdefault("end_when", "visited")
    fog.setdefault("viz_port", 0)

    # normalize agents
    for a in cfg["agents"]:
        a.setdefault("mission", "grid_frontier")
        a.setdefault("policy", "random")
        a.setdefault("strategy_tag", a.get("mission", ""))
        a.setdefault("comm_mode", "full")
        a.setdefault("seed", None)
        a.setdefault("partition", {})
        p = a["partition"]
        p.setdefault("scheme", "none")
        p.setdefault("n", 1)
        p.setdefault("id", 0)

    return cfg


def agent_cmd(cfg: dict, agent: dict):
    origin = cfg["origin"]
    comms = cfg["comms"]
    grid = cfg["grid"]
    timing = cfg["timing"]
    vdef = cfg["vehicle_defaults"]

def agent_cmd(cfg: dict, agent: dict, verbose: bool = False):
    origin = cfg["origin"]
    comms = cfg["comms"]
    grid = cfg["grid"]
    timing = cfg["timing"]
    vdef = cfg["vehicle_defaults"]

    args = [PY, SA,
        "--conn", str(agent["conn"]),
        "--name", str(agent.get("name") or agent["conn"]),
        "--origin-lat", str(origin["lat"]),
        "--origin-lon", str(origin["lon"]),
        "--geofence-miles", str(cfg.get("geofence_miles", 1.0)),
        "--alt", str(agent.get("alt", vdef["alt"])),
        "--vxy-max", str(agent.get("vxy_max", vdef["vxy_max"])),
        "--vz-max", str(agent.get("vz_max", vdef["vz_max"])),
        "--axy-max", str(agent.get("axy_max", vdef["axy_max"])),
        "--az-max", str(agent.get("az_max", vdef["az_max"])),
        "--rate-hz", str(agent.get("rate_hz", vdef["rate_hz"])),
        "--gossip-group", str(comms["group"]),
        "--gossip-port", str(comms["port"]),
        "--comm-grid-radius", str(comms["comm_grid_radius"]),
        "--comm-mode", str(agent.get("comm_mode", "full")),
        "--experiment-tag", str(cfg.get("experiment_tag", "")),
        "--strategy-tag", str(agent.get("strategy_tag", "")),
        "--mission", str(agent.get("mission", "grid_frontier")),
        "--policy", str(agent.get("policy", "random")),
        "--grid-miles", str(grid["miles"]),
        "--grid-cells", str(grid["cells"]),
        "--traverse-frac", str(grid["traverse_frac"]),
        # legacy flags remain for compatibility (TRAVERSE ignores them internally)
        "--map-radius-m", str(grid["map_radius_m"]),
        "--cell-complete-frac", str(grid["cell_complete_frac"]),
        # partitions
        "--partition-scheme", str(agent["partition"]["scheme"]),
        "--partition-n", str(agent["partition"]["n"]),
        "--partition-id", str(agent["partition"]["id"]),
        # end conditions (local agent stop)
        "--end-when", str(timing["end_when"]),
        "--max-seconds", str(timing["max_seconds"]),
    ]
    seed = agent.get("seed")
    if seed is not None:
        args += ["--seed", str(seed)]

    # NEW: only pass --verbose if launcher asked for it
    if verbose:
        args.append("--verbose")

    return args


def fog_cmd(cfg: dict, cfg_path: Path):
    origin = cfg["origin"]
    comms = cfg["comms"]
    grid = cfg["grid"]
    fog = cfg["fog_tracker"]
    timing = cfg["timing"]
    args = [
        PY, FT,
        "--origin-lat", str(origin["lat"]),
        "--origin-lon", str(origin["lon"]),
        "--grid-miles", str(grid["miles"]),
        "--grid-cells", str(grid["cells"]),
        "--traverse-frac", str(grid["traverse_frac"]),
        "--gossip-group", str(comms["group"]),
        "--gossip-port", str(comms["port"]),
        "--tick-rate-hz", str(fog["tick_rate_hz"]),
        # Fog should honor its own section's end_when (default 'visited')
        "--end-when", str(fog.get("end_when", "visited")),
        "--max-seconds", str(fog["max_seconds"]),
        # Handy for tagging and fog's new runs/ folder copy
        "--experiment-tag", str(cfg.get("experiment_tag", "")),
        "--config", str(cfg_path),
    ]
    if int(fog.get("viz_port", 0)) > 0:
        args += ["--viz-port", str(fog["viz_port"])]
    return args


def maybe_auto_cells(cfg: dict):
    """
    If grid.auto_cell_m is set, compute grid.cells from desired cell size (meters).
    """
    grid = cfg.get("grid", {})
    auto = grid.get("auto_cell_m")
    if auto is None:
        return cfg
    try:
        target = float(auto)
        side_m = float(grid.get("miles", cfg.get("geofence_miles", 1.0))) * 1609.34
        cells = max(2, int(round(side_m / target)))
        grid["cells"] = cells
        cfg["grid"] = grid
        print(f"[launcher] auto grid cells={cells} for ~{target:.0f} m cell size")
    except Exception as e:
        print(f"[launcher] WARN: auto_cell_m failed ({e}); using grid.cells={grid.get('cells')}")
    return cfg


def launch(cfg_path: Path, verbose: bool = False):
    cfg = ensure_defaults(load_config(cfg_path))
    cfg = maybe_auto_cells(cfg)

    procs = []

    # launch fog first (optional)
    if cfg["fog_tracker"].get("enable", True):
        cmd = fog_cmd(cfg, cfg_path)
        print("[launcher] fog_tracker:", " ".join(cmd))
        procs.append(subprocess.Popen(cmd))
        time.sleep(0.2)

    # launch agents
    for agent in cfg["agents"]:
        cmd = agent_cmd(cfg, agent, verbose=verbose)
        print(f"[launcher] agent {agent.get('name', agent['conn'])}:", " ".join(cmd))
        procs.append(subprocess.Popen(cmd))
        time.sleep(0.2)

    # wait for children
    try:
        while procs:
            still = []
            for p in procs:
                rc = p.poll()
                if rc is None:
                    still.append(p)
            procs = still
            time.sleep(0.5)
        print("[launcher] All child processes exited — experiment complete.")
    except KeyboardInterrupt:
        print("[launcher] Ctrl+C received, forwarding SIGINT to children (RTL).")
        for p in procs:
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass
        time.sleep(3.0)
    finally:
        for p in procs:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass

# ------------- CLI -------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to experiment.yaml or .json")
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging in all swarm_agent processes",
    )
    args = ap.parse_args()
    launch(Path(args.config), verbose=args.verbose)
