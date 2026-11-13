#!/usr/bin/env python3
"""
run_fresh.py - run a NEW Level 3 simulation (ArduPilot SITL + the real planner) for one scenario, with a thin wrapper
around eval/level3_pairs.py so a first-time user does not have to write configs by hand.

    python3 run_fresh.py --arm baseline            # no neural network
    python3 run_fresh.py --arm A                   # Model A (synthetic-trained YOLO), seed 0
    python3 run_fresh.py --arm B                   # Model B (real/Boreal-trained YOLO, "v2"), seed 0
    python3 run_fresh.py --arm all --dashboard     # all three, one after the other, with the live browser view
    python3 run_fresh.py --arm A --seed 2 --scenario 2004 --dry-run   # only print what would run

Each run takes about 12 minutes (45 s SITL boot + start gate + 600 s of simulated mission) and needs about 1.4 GB RAM.
The wrapper adds nothing scientific: it calls the same planner (eval/level3_pairs.py plan/run) that produced the 56 recorded
runs, so the scenario, fire, camera and thresholds are the frozen ones, and the arms of one scenario differ only in the
perception keys. Outputs go to NEW directories under local_data/fresh_runs/; the recorded runs are never touched.
Fresh runs are NOT guaranteed to reproduce the recorded numbers exactly (wall-clock scheduling; see docs/PROFESSOR_GUIDE.md).
"""
import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from eval import level3_pairs as LP  # noqa: E402
from eval import level3_scenarios as SC  # noqa: E402

FRESH = os.path.join(ROOT, "local_data", "fresh_runs")


def arm_id(arm: str, seed: int) -> str:
    if arm == "baseline":
        return "baseline"
    return f"A{seed}" if arm == "A" else f"B2s{seed}"


def split_of(scenario: int) -> str:
    sc = SC.load()
    for split in ("calibration", "heldout"):
        if any(s["seed"] == scenario for s in sc[split]):
            return split
    raise SystemExit(f"unknown scenario {scenario}; choose one of "
                     f"{[s['seed'] for s in sc['calibration']]} (calibration) or {[s['seed'] for s in sc['heldout']]} (held-out)")


def preflight(arm_ids, python: str) -> list:
    """Return a list of problems (empty = ready)."""
    bad = []
    for tool in ("tmux", "sim_vehicle.py"):
        if not shutil.which(tool):
            bad.append(f"`{tool}` not found on PATH (see docs/PROFESSOR_GUIDE.md section 5)")
    sv = shutil.which("sim_vehicle.py")
    if sv:
        ap_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(sv))))
        if not os.path.exists(os.path.join(ap_root, "build", "sitl", "bin", "arducopter")):
            bad.append(f"ArduPilot SITL copter is not built ({ap_root}/build/sitl/bin/arducopter missing): "
                       "cd there && ./waf configure --board sitl && ./waf copter")
        loc = os.path.join(ap_root, "Tools", "autotest", "locations.txt")
        if os.path.exists(loc) and not any(l.startswith("UH=") for l in open(loc)):
            bad.append(f"start location UH is not defined: echo 'UH=21.296868,-157.816036,100,270' >> {loc}")
    if not shutil.which("mavproxy.py"):
        bad.append("`mavproxy.py` not found on PATH: activate the virtual environment that has `pip install -r requirements-sim.txt`")
    for mod in ("yaml", "numpy", "cv2", "PIL", "pymavlink"):
        r = subprocess.run([python, "-c", f"import {mod}"], capture_output=True)
        if r.returncode:
            bad.append(f"{python} cannot import {mod}: pip install -r requirements-sim.txt")
    need_yolo = [a for a in arm_ids if LP.ARMS[a] is not None]
    if need_yolo:
        r = subprocess.run([python, "-c", "import ultralytics, torch"], capture_output=True)
        if r.returncode:
            bad.append(f"{python} cannot import ultralytics/torch: pip install -r requirements-yolo.txt")
        for a in need_yolo:
            w = os.path.join(ROOT, LP.ARMS[a][0])
            if not os.path.exists(w):
                bad.append(f"weights for arm {a} missing: {LP.ARMS[a][0]} (not in git; see docs/PROFESSOR_GUIDE.md section 2.3)")
    leftovers = LP._pids(LP.NAMES)
    if leftovers:
        bad.append(f"SITL/agent/fog processes already running (pids {sorted(leftovers)}); stop them first")
    if not LP._port_free(LP.PORT_CAMERA):
        bad.append(f"TCP port {LP.PORT_CAMERA} (camera service) is in use")
    return bad


def tail(path: str, n: int = 15) -> str:
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except OSError:
        return f"(no {path})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["baseline", "A", "B", "all"], required=True)
    ap.add_argument("--seed", type=int, choices=[0, 1, 2], default=0, help="which trained copy of Model A / B (default 0)")
    ap.add_argument("--scenario", type=int, default=2001, help="scenario seed (default 2001, a held-out scenario with a recorded run)")
    ap.add_argument("--python", default=sys.executable, help="interpreter that has the requirements (default: the current one)")
    ap.add_argument("--out", default="", help=f"output directory (default {FRESH}/<timestamp>-<scenario>)")
    ap.add_argument("--dashboard", action="store_true", help="also start the live dashboard at http://localhost:8080")
    ap.add_argument("--dashboard-host", default="127.0.0.1", help="0.0.0.0 to reach it from outside a VM")
    ap.add_argument("--dashboard-port", type=int, default=8080)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    arms = ["baseline", "A", "B"] if a.arm == "all" else [a.arm]
    ids = [arm_id(x, a.seed) for x in arms]
    split = split_of(a.scenario)
    out = os.path.abspath(a.out or os.path.join(FRESH, time.strftime("%Y%m%d-%H%M%S") + f"-{a.scenario}"))
    print(f"[fresh] scenario {a.scenario} ({split}); arms {ids}; output {out}")
    print("[fresh] this is a NEW simulation, not a replay: about 12 minutes per arm.")

    problems = preflight(ids, a.python)
    for p in problems:
        print(f"[fresh] PROBLEM: {p}", file=sys.stderr)
    if a.dry_run:
        print("[fresh] dry run: nothing started." + ("" if not problems else " (fix the problems above first)"))
        return 1 if problems else 0
    if problems:
        return 2

    if os.path.isdir(os.path.join(out, "results")) and os.listdir(os.path.join(out, "results")):
        print(f"[fresh] {out}/results already holds results; refusing to mix runs. Use --out with a new directory.", file=sys.stderr)
        return 2
    if a.dashboard and not LP._port_free(a.dashboard_port):
        print(f"[fresh] dashboard port {a.dashboard_port} is in use; choose --dashboard-port or stop the other program.", file=sys.stderr)
        return 2
    plan_dir = os.path.join(out, "plan")
    LP.plan(split, ids, plan_dir, 1, final=(split == "heldout"), seeds=[a.scenario])
    dash = None
    if a.dashboard:
        dash = subprocess.Popen([a.python, os.path.join(ROOT, "dashboard", "server.py"), "--host", a.dashboard_host,
                                 "--port", str(a.dashboard_port)], start_new_session=True)
        shown = "localhost" if a.dashboard_host in ("127.0.0.1", "0.0.0.0") else a.dashboard_host
        print(f"[fresh] live dashboard: http://{shown}:{a.dashboard_port}  (it shows the run once the UAV has taken off, ~2 min in)")
    rc = 0
    try:
        for arm in ids:
            cfg = os.path.join(plan_dir, f"l3-{arm}-{a.scenario}.yaml")
            res = os.path.join(out, "results", f"l3-{arm}-{a.scenario}")
            print(f"\n[fresh] === {arm}: starting {time.strftime('%H:%M:%S')} (Ctrl+C aborts and cleans up) ===", flush=True)
            r = subprocess.run([a.python, os.path.join(ROOT, "eval", "level3_pairs.py"), "run", "--config", cfg, "--python", a.python,
                                "--out", res], start_new_session=False)
            if r.returncode:
                print(f"[fresh] {arm} FAILED (exit {r.returncode}). Last lines of {res}/sitl.log:\n{tail(res + '/sitl.log')}"
                      f"\nLast lines of fog.log:\n{tail(res + '/fog.log')}\n[fresh] Likely causes: SITL did not boot (see PROFESSOR_GUIDE 7.2/7.3), "
                      "the start gate never opened on a slow machine, or a port is taken.", file=sys.stderr)
                rc = 1
                break
            end = glob.glob(os.path.join(res, "experiment_end_*.txt"))
            print(f"[fresh] {arm} finished; files in {res}")
            if end:
                kv = dict(l.strip().split("=", 1) for l in open(end[0]) if "=" in l)
                print("[fresh]   " + ", ".join(f"{k}={kv.get(k)}" for k in ("reason", "time_s", "visited_frac", "det_true", "det_false",
                                                                        "first_detection_t", "detection_lag_s")))
    except KeyboardInterrupt:
        print("\n[fresh] interrupted; the run helper cleaned up its SITL processes. Partial output kept in", out)
        rc = 130
    finally:
        if dash:
            try:
                os.killpg(os.getpgid(dash.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
    if rc == 0:
        print(f"\n[fresh] done. Results: {out}/results/<run>/ (experiment_end_*.txt, detections_*.csv, coverage_t_*.csv, manifest.json)")
        print(f"[fresh] compare with the recorded run: docs/experiments/level3_final/per_run.csv (row l3-<arm>-{a.scenario})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
