#!/usr/bin/env python3
"""
demo_multi.py - launch the FOUR-UAV demonstration: four ArduPilot SITL copters fly a quadrant-partitioned grid-coverage mission over
a simulated wildfire, watched live in a browser.

    python3 demo_multi.py --drones 4              # then open http://localhost:8080/multi.html
    python3 demo_multi.py --drones 4 --dry-run    # check prerequisites only

This is a DEMONSTRATION of the multi-UAV architecture, NOT a research experiment. All research results in this project are
single-UAV runs and are unchanged. Nothing here is a result.

It reuses the existing machinery and adds no flight or simulation code:
  launch_sitl_swarm.py   4 x ArduCopter SITL + MAVProxy (tmux), via eval/level3_pairs.run_one (which also cleans everything up)
  fog_tracker.py         the ground station: owns the simulated fire, the shared coverage map and the run clock
  swarm_agent.py x 4     one agent per UAV; mission grid_frontier; partition scheme "quadrants" (UAV k owns one quadrant)
  dashboard/server.py    the existing read-only dashboard (port 8080); this launcher only adds the page multi.html and a metadata file
No detector runs (perception disabled), so no smoke detections exist or are shown. The fire settings are illustrative demo values.
Outputs of the run (fog CSVs, logs, MAVLink logs) go to a NEW directory under local_data/demo_multi/.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from eval import level3_pairs as LP  # noqa: E402
from mission import partitions as PART  # noqa: E402
import run_fresh as RF  # noqa: E402  (its preflight() is reused)

COLORS = ["#00e5ff", "#ffd400", "#c77dff", "#7cfc00"]
REGION = {0: "south-west quadrant", 1: "south-east quadrant", 2: "north-west quadrant", 3: "north-east quadrant"}
BANNER = ["MULTI-UAV ARCHITECTURE DEMONSTRATION — Four simulated ArduPilot drones performing grid coverage.",
          "This demonstration is separate from the validated single-UAV research experiments."]
TRUTH = [
    ["Driven by ArduPilot SITL:", "each UAV is a separate ArduCopter SITL instance. Its position, altitude, heading, speed and flight mode are real MAVLink "
     "telemetry; the agents steer it with real MAVLink commands. Trails are only the positions received."],
    ["Driven by the project's own code:", "the quadrant assignment, the grid_frontier target choice, the shared coverage map and the run clock "
     "(fog_tracker.py, swarm_agent.py, mission/)."],
    ["Simulated, not real:", "the wildfire is a cellular-automaton model owned by the ground station. The UAVs never see it (oracle off), and with no "
     "detector in this demo it does not influence their flight."],
    ["Illustrative, not validated:", "the fire parameters and the quadrant coverage pattern were chosen for display. No performance claim is made, "
     "nothing here was measured as an experiment, and multi-UAV behaviour was not experimentally validated. No probability map is used."],
]


def build_config(a):
    base = LP.build_base(a.drones)
    fm = base["fire_model"]
    fm.update({"p_ignite": a.p_ignite, "burn_steps": a.burn_steps, "burn_time_dist": "fixed",
               "initial_ignitions": [[int(x) for x in a.ignite.split(",")]], "random_seed": a.fire_seed})
    gate = fm["start_gate"]
    gate["min_alt_m"] = 0.8 * a.alt
    base["vehicle_defaults"] = {"alt": a.alt, "vxy_max": a.speed}
    base["experiment_tag"] = "demo-multi-4uav"
    base["timing"]["max_seconds"] = a.duration
    base["fog_tracker"]["max_seconds"] = a.duration
    for k, ag in enumerate(base["agents"]):
        ag["strategy_tag"] = "demo"
        ag["seed"] = 100 + k
    base["perception"]["enabled"] = False          # no detector in this demonstration
    base["perception"]["camera_service"]["enabled"] = False
    return base


def build_meta(cfg, a):
    N = int(cfg["grid"]["cells"])
    uavs = []
    for k, ag in enumerate(cfg["agents"]):
        p = ag["partition"]
        cells = PART.allowed_cells(p["scheme"], N, p["id"], p["n"])
        uavs.append({"n": k + 1, "label": f"UAV {k + 1}", "sys": k + 1, "color": COLORS[k % 4], "region": REGION.get(p["id"], ""),
                     "cells": [list(c) for c in cells]})
    return {"banner": BANNER, "origin": cfg["origin"], "miles": cfg["grid"]["miles"], "N": N, "uavs": uavs, "perception": False,
            "truth": TRUTH, "duration_s": a.duration,
            "fire": {"p_ignite": a.p_ignite, "burn_steps": a.burn_steps, "ignition": a.ignite, "illustrative": True}}


def mem_available_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def monitor(url, stop):
    last = ""
    while not stop.wait(20):
        try:
            s = json.load(urllib.request.urlopen(url + "/api/state", timeout=3))
            ag = s.get("agents", {})
            live = sum(1 for v in ag.values() if not v.get("stale"))
            w = s.get("world", {})
            line = f"[demo] UAVs reporting {live}/4 | t={w.get('t', '-')} s | coverage {100 * w.get('visited_frac', 0):.0f}% | burning {w.get('burning_cnt', '-')}"
            if line != last:
                print(line, flush=True)
            last = line
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drones", type=int, default=4, choices=[4], help="only the 4-drone quadrant demonstration is supported")
    ap.add_argument("--duration", type=int, default=900, help="simulated seconds after takeoff (default 900; 4 UAVs reach about 75% coverage in 600 s)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to reach the page from outside a VM")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--exit-when-done", action="store_true", help="stop the dashboard when the run ends (default: keep showing the final state)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default="")
    ap.add_argument("--alt", type=float, default=60.0)
    ap.add_argument("--speed", type=float, default=15.0)
    ap.add_argument("--ignite", default="5,5", help="ignition cell i,j (default 5,5)")
    ap.add_argument("--p-ignite", dest="p_ignite", type=float, default=0.0025, help="default = the Level 3 fire settings (values copied, files untouched)")
    ap.add_argument("--burn-steps", dest="burn_steps", type=int, default=300)
    ap.add_argument("--fire-seed", dest="fire_seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    problems = RF.preflight([], a.python)
    mem = mem_available_mb()
    if mem is not None and mem < 1000:
        print(f"[demo] WARNING: only {mem:.0f} MB RAM available; four SITL copters + agents + fog may not fit "
              "(close other programs or give the VM more memory). Continuing.", file=sys.stderr)
    if not LP._port_free(a.port):
        problems.append(f"port {a.port} is in use (another dashboard?); use --port, or stop it")
    for p in problems:
        print(f"[demo] PROBLEM: {p}", file=sys.stderr)
    if a.dry_run:
        print("[demo] dry run: nothing started." + (" (fix the problems above first)" if problems else " Prerequisites look fine."))
        return 1 if problems else 0
    if problems:
        return 2

    out = os.path.abspath(a.out or os.path.join(ROOT, "local_data", "demo_multi", time.strftime("%Y%m%d-%H%M%S")))
    os.makedirs(out, exist_ok=True)
    cfg = build_config(a)
    import yaml
    cfg_path = os.path.join(out, "demo_multi.yaml")
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
    meta_path = os.path.join(out, "demo_meta.json")
    json.dump(build_meta(cfg, a), open(meta_path, "w"), indent=1)

    shown = "localhost" if a.host == "127.0.0.1" else a.host
    if a.host == "0.0.0.0":                      # reachable from other machines: print an address they can use
        try:
            shown = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()[0]
        except (IndexError, OSError):
            shown = "localhost"
    url = f"http://{shown}:{a.port}"
    dash = subprocess.Popen([a.python, os.path.join(ROOT, "dashboard", "server.py"), "--host", a.host, "--port", str(a.port),
                             "--demo-meta", meta_path], start_new_session=True)
    time.sleep(1.5)
    if dash.poll() is not None:
        print("[demo] the dashboard failed to start; see the message above.", file=sys.stderr)
        return 2
    print(f"[demo] MULTI-UAV DEMONSTRATION (4 ArduPilot SITL copters; not a research experiment)\n"
          f"[demo] open {url}/multi.html   (Ctrl+C stops everything)\n"
          f"[demo] startup takes about 2 minutes (4 SITL boots), then {a.duration} s of simulated mission. Output: {out}", flush=True)
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"{url}/multi.html")).start()
    stop = threading.Event()
    threading.Thread(target=monitor, args=(f"http://127.0.0.1:{a.port}", stop), daemon=True).start()

    def _term(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)          # `kill <pid>` shuts down as cleanly as Ctrl+C
    rc = 0
    try:
        for attempt in (1, 2):
            try:
                res = os.path.join(out, "results")
                meta = LP.run_one(cfg_path, a.python, res, os.path.join(res, "sitl_work"))
                json.dump(meta, open(os.path.join(res, "run_meta.json"), "w"), indent=1)
                print(f"[demo] mission finished (fog exit {meta.get('fog_exit_code')}, {meta.get('wall_s')} s wall); "
                      f"SITL and agents are stopped. Files: {res}")
                break
            except RuntimeError as e:
                if attempt == 2 or "start gate" not in str(e):
                    raise
                print(f"[demo] start gate did not open ({e}); retrying once", flush=True)
                time.sleep(20)
        if not a.exit_when_done:
            print(f"[demo] the dashboard keeps showing the final state at {url}/multi.html ; press Ctrl+C to quit", flush=True)
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[demo] stopping ...")
    except RuntimeError as e:
        print(f"[demo] FAILED: {e}\n[demo] look at {out}/results/{{fog,sitl,agent_A}}.log; see docs/PROFESSOR_GUIDE.md section 7.2/7.3", file=sys.stderr)
        rc = 1
    finally:
        stop.set()
        try:
            os.killpg(os.getpgid(dash.pid), signal.SIGINT)
        except ProcessLookupError:
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
