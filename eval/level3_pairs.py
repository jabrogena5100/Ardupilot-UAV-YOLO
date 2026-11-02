#!/usr/bin/env python3
"""
eval/level3_pairs.py — paired Level 3 runs: identical scenario and planner, only the perception arm differs.

An ARM is `baseline` (perception off: the same planner/flight code, no detections) or a registry checkpoint (Model A seeds 0-2,
Model B v2 seeds 0-2). A SCENARIO (eval/level3_scenarios.json) fixes the fire seed and ignition cell, the rendered environment, the
agents' seeds, altitude, speed, camera and fire-spread settings (eval/level3_fire.json, frozen). The ONLY keys that may differ between arms
of one scenario are the perception arm keys; `conf_threshold` must equal the arm's calibrated value in eval/level3_thresholds.json.

    python3 eval/level3_pairs.py plan --split calibration --out local_data/level3/pilot
    python3 eval/level3_pairs.py run  --config local_data/level3/pilot/l3-B2s0-1001.yaml --python ~/venvs/yolo-cpu/bin/python

`--split heldout` is refused unless --final-evaluation is given AND the thresholds and fire files exist (they are frozen before evaluation).
This produces RUNS, not conclusions.
"""
import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval import level3_scenarios as SC  # noqa: E402

B = "local_data/backups"
ARMS = {
    "baseline": None,
    "A0": (f"{B}/synth_smoke_a_v1/synth_smoke_a_v1/synth_smoke_best_v1.pt", "synth_smoke_v1_seed0"),
    "A1": (f"{B}/level2_v2_seeds/synth_smoke_a_v1_seed1/synth_smoke_best_v1_seed1.pt", "synth_smoke_v1_seed1"),
    "A2": (f"{B}/level2_v2_seeds/synth_smoke_a_v1_seed2/synth_smoke_best_v1_seed2.pt", "synth_smoke_v1_seed2"),
    "B2s0": (f"{B}/boreal_v2/real_boreal_v2_best_seed0.pt", "real_boreal_v2_seed0"),
    "B2s1": (f"{B}/boreal_v2/real_boreal_v2_best_seed1.pt", "real_boreal_v2_seed1"),
    "B2s2": (f"{B}/boreal_v2/real_boreal_v2_best_seed2.pt", "real_boreal_v2_seed2"),
}
ARM_KEYS = ("enabled", "weights", "weights_sha256", "model_tag", "conf_threshold")   # the ONLY perception keys allowed to differ
FIRE_PATH = os.path.join(ROOT, "eval", "level3_fire.json")
THR_PATH = os.path.join(ROOT, "eval", "level3_thresholds.json")
DEFAULT_TAU = 0.25
PORT_CAMERA = 8765


def load_fire() -> dict:
    return json.load(open(FIRE_PATH))


def load_thresholds() -> dict:
    return json.load(open(THR_PATH))["arms"] if os.path.exists(THR_PATH) else {}


def tau_for(arm: str, table: dict) -> float:
    return float(table[arm]["tau"]) if arm in table else DEFAULT_TAU


def build_base(n_agents: int = 1) -> dict:
    """The arm-independent part of every run. Everything here is identical across arms and, except the scenario fields filled in
    by make_config, across scenarios."""
    fire = load_fire()
    T = int(fire["T_s"])
    agents = [{"name": chr(65 + k), "conn": f"udp:127.0.0.1:{14555 + k}", "mission": "grid_frontier", "strategy_tag": "l3",
               "comm_mode": "full", "seed": 0} for k in range(n_agents)]
    if n_agents == 4:
        for k, a in enumerate(agents):
            a["partition"] = {"scheme": "quadrants", "n": 4, "id": k}
    return {
        "experiment_tag": "l3", "origin": {"lat": SC.LAT0, "lon": SC.LON0}, "grid": {"miles": 1.0, "cells": 10, "traverse_frac": 0.8},
        "comms": {"group": "239.255.0.1", "port": 5005, "comm_grid_radius": 2},
        "timing": {"end_when": "time", "max_seconds": T}, "fog_tracker": {"enable": True, "tick_rate_hz": 2, "max_seconds": T, "end_when": "time"},
        "fire_model": {"enabled": True, "oracle_hazards": False, "neighborhood": fire["neighborhood"], "p_ignite": fire["p_ignite"],
                       "burn_steps": fire["burn_steps"], "burn_time_dist": fire["burn_time_dist"], "dt_seconds": fire["dt_seconds"],
                       "wind_direction_deg": 0.0, "wind_strength": fire["wind_strength"], "initial_ignitions": [], "random_seed": 0,
                       "start_gate": {"agents": n_agents, "min_alt_m": 100, "settle_s": 5, "timeout_s": 420}},
        "vehicle_defaults": {"alt": 120, "vxy_max": 15},
        "perception": {"enabled": False, "async": True, "rate_hz": 0.25, "truth_radius_cells": 1, "detection_ttl_s": 20.0, "investigate_cooldown_s": 60.0,
                       "camera": {"hfov_deg": 78, "pitch_deg": 60, "width": 1280, "height": 720, "yaw_to_track": True},
                       "frame_source": {"url": f"http://127.0.0.1:{PORT_CAMERA}", "timeout_s": 120},
                       "camera_service": {"enabled": True, "port": PORT_CAMERA, "scenario_seed": 0}},
        "agents": agents,
    }


def make_config(base: dict, arm: str, scenario: dict, table: dict = None) -> dict:
    table = load_thresholds() if table is None else table
    cfg = copy.deepcopy(base)
    cfg["experiment_tag"] = f"l3-{arm}-{scenario['seed']}"
    fm = cfg["fire_model"]
    fm["random_seed"] = int(scenario["fire_seed"])
    fm["initial_ignitions"] = [list(scenario["ignition"])]
    p = cfg["perception"]
    p["camera_service"]["scenario_seed"] = int(scenario["seed"])
    for k in ARM_KEYS:
        p.pop(k, None)
    p["conf_threshold"] = tau_for(arm, table)
    if ARMS[arm] is None:
        p["enabled"], p["model_tag"] = False, "baseline"
    else:
        p["enabled"] = True
        p["weights"], p["model_tag"] = ARMS[arm]
    for k, a in enumerate(cfg["agents"]):         # agent seeds derive from the scenario seed, never from the arm
        a["seed"] = int(scenario["seed"]) * 100 + k + 1
    return cfg


def strip_arm(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c.pop("experiment_tag", None)
    for k in ARM_KEYS:
        c.get("perception", {}).pop(k, None)
    return c


def assert_paired(cfgs_by_arm: dict, table: dict = None) -> None:
    """Every arm of one scenario must be identical once the arm keys and the tag are removed, and each arm's threshold must be
    its calibrated value."""
    table = load_thresholds() if table is None else table
    ref_arm = next(iter(cfgs_by_arm))
    ref = strip_arm(cfgs_by_arm[ref_arm])
    for arm, c in cfgs_by_arm.items():
        if strip_arm(c) != ref:
            raise ValueError(f"arm {arm} differs from {ref_arm} in more than the perception arm")
        if float(c["perception"]["conf_threshold"]) != tau_for(arm, table):
            raise ValueError(f"arm {arm}: conf_threshold is not its calibrated value")


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


def plan(split: str, arms, out: str, n_agents: int = 1, final: bool = False, seeds=None) -> list:
    sc = SC.load()
    if split == "heldout":
        if not final:
            raise PermissionError("held-out scenarios may only be planned with --final-evaluation")
        if not (os.path.exists(THR_PATH) and os.path.exists(FIRE_PATH)):
            raise PermissionError("thresholds and fire settings must be frozen before the held-out evaluation")
    scens = [s for s in sc[split] if seeds is None or s["seed"] in seeds]
    base = build_base(n_agents)
    table = load_thresholds()
    os.makedirs(out, exist_ok=True)
    rows = []
    for s in scens:
        cfgs = {a: make_config(base, a, s, table) for a in arms}
        assert_paired(cfgs, table)
        for a, c in cfgs.items():
            path = os.path.join(out, f"{c['experiment_tag']}.yaml")
            yaml.safe_dump(c, open(path, "w"), sort_keys=False)
            rows.append({"scenario": s["seed"], "split": split, "arm": a, "config": path})
    with open(os.path.join(out, "matrix.json"), "w") as f:
        json.dump({"split": split, "arms": list(arms), "n_agents": n_agents, "runs": rows, "scenarios_sha256": _sha(SC.PATH),
                   "fire_sha256": _sha(FIRE_PATH), "thresholds_sha256": _sha(THR_PATH)}, f, indent=1)
    return rows


# ---------------------------------------------------------------- run

NAMES = {"arducopter", "mavproxy.py", "sim_vehicle.py", "swarm_agent.py", "fog_tracker.py"}


def _pids(names) -> set:
    """PIDs whose argv actually runs one of `names`; shells that merely mention a name in a command string do not count."""
    me = {os.getpid(), os.getppid()}
    out = set()
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) in me:
            continue
        try:
            argv = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        argv = [a.decode(errors="ignore") for a in argv if a]
        if not argv or os.path.basename(argv[0]) in ("bash", "sh", "dash", "zsh", "pgrep", "grep"):
            continue
        if any(os.path.basename(a) in names for a in argv[:3]):
            out.add(int(d))
    return out


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)    # as the camera server does: TIME_WAIT leftovers of the previous run are not "in use"
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _git(*a):
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout.strip()


def _kill_group(p, sig=signal.SIGTERM):
    try:
        os.killpg(os.getpgid(p.pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def run_one(config: str, python: str, out_dir: str, workdir: str, sitl_boot_s: float = 45.0) -> dict:
    cfg = yaml.safe_load(open(config))
    agents = cfg.get("agents", [])
    n = len(agents)
    ori, grid = cfg["origin"], cfg.get("grid", {})
    maxs = float(cfg["fog_tracker"]["max_seconds"])
    gate_to = float(cfg["fire_model"].get("start_gate", {}).get("timeout_s", 420))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(workdir, exist_ok=True)
    leftovers = _pids(NAMES)
    if leftovers:
        raise RuntimeError(f"SITL/agent/fog processes already running {sorted(leftovers)}; refusing to start (and to kill them)")
    cam_port = cfg["perception"]["camera_service"]["port"]
    if not _port_free(cam_port):
        raise RuntimeError(f"camera service port {cam_port} is in use")
    env = dict(os.environ, PYTHONHASHSEED="0")
    started = time.time()
    meta = {"config": os.path.abspath(config), "config_sha256": _sha(config), "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)), "python": python, "pythonhashseed": "0",
            "thresholds_sha256": _sha(THR_PATH), "fire_sha256": _sha(FIRE_PATH), "scenarios_sha256": _sha(SC.PATH)}
    procs, logs = [], []

    def popen(cmd, name, cwd):
        f = open(os.path.join(out_dir, name), "w")
        logs.append(f)
        p = subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        procs.append(p)
        return p

    try:
        fog_cmd = [python, "-u", os.path.join(ROOT, "fog_tracker.py"), "--origin-lat", str(ori["lat"]), "--origin-lon", str(ori["lon"]),
                   "--grid-miles", str(grid.get("miles", 1.0)), "--grid-cells", str(grid.get("cells", 10)), "--end-when", "time",
                   "--max-seconds", str(maxs), "--config", config]
        fog = popen(fog_cmd, "fog.log", ROOT)                     # fog first: it is the run clock and waits at the start gate
        popen([sys.executable, os.path.join(ROOT, "launch_sitl_swarm.py"), "--count", str(n)], "sitl.log", workdir)
        time.sleep(sitl_boot_s + 15 * max(0, n - 1))
        vd = cfg.get("vehicle_defaults", {})
        for a in agents:
            cmd = [python, "-u", os.path.join(ROOT, "swarm_agent.py"), "--conn", a["conn"], "--origin-lat", str(ori["lat"]),
                   "--origin-lon", str(ori["lon"]), "--mission", a.get("mission", "grid_frontier"), "--alt", str(vd.get("alt", 120)),
                   "--vxy-max", str(vd.get("vxy_max", 15)), "--config", config, "--seed", str(a.get("seed", 0)),
                   "--strategy-tag", str(a.get("strategy_tag", "")), "--experiment-tag", cfg["experiment_tag"], "--verbose"]
            if "partition" in a:
                cmd += ["--partition-scheme", a["partition"]["scheme"], "--partition-n", str(a["partition"]["n"]),
                        "--partition-id", str(a["partition"]["id"])]
            popen(cmd, f"agent_{a['name']}.log", ROOT)
        try:
            fog.wait(timeout=gate_to + maxs + 240 + sitl_boot_s)
        except subprocess.TimeoutExpired:
            raise RuntimeError("fog did not finish in time")
        meta["fog_exit_code"] = fog.returncode
        if fog.returncode != 0:
            raise RuntimeError(f"fog exited with {fog.returncode} (start gate never opened?)")
        m = re.search(r"wrote results to (\S+)", open(os.path.join(out_dir, "fog.log")).read())
        res = m.group(1) if m else None
        if res and os.path.isdir(res):
            for f in os.listdir(res):
                shutil.copy(os.path.join(res, f), os.path.join(out_dir, f))
        meta["results_dir"] = res
        return meta
    finally:
        for p in procs:
            _kill_group(p, signal.SIGTERM)
        subprocess.run(["tmux", "kill-session", "-t", "sitl-swarm"], capture_output=True)
        time.sleep(3)
        for p in procs:
            _kill_group(p, signal.SIGKILL)
        for pid in _pids({"arducopter", "mavproxy.py", "sim_vehicle.py"}):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for f in logs:
            f.close()
        time.sleep(2)
        meta["end_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta["wall_s"] = round(time.time() - started, 1)
        meta["leftover_after_cleanup"] = sorted(_pids(NAMES))
        json.dump(meta, open(os.path.join(out_dir, "run_meta.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--split", choices=["calibration", "heldout"], default="calibration")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--agents", type=int, default=1)
    p.add_argument("--final-evaluation", action="store_true")
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--out", default="")
    r.add_argument("--workdir", default="")
    r.add_argument("--attempts", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "plan":
        rows = plan(a.split, a.arms, a.out, a.agents, a.final_evaluation, a.seeds)
        print(f"wrote {len(rows)} configs to {a.out}")
    else:
        tag = os.path.splitext(os.path.basename(a.config))[0]
        out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.config)), "results", tag)
        for attempt in range(1, a.attempts + 1):             # a run whose start gate never opened (SITL slow to get a fix) is retried clean
            try:
                meta = run_one(a.config, a.python, out, a.workdir or os.path.join(out, "sitl_work"))
                meta["attempt"] = attempt
                json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1)
                print(json.dumps(meta))
                break
            except RuntimeError as e:
                if attempt >= a.attempts or "start gate" not in str(e):
                    raise
                failed = f"{out}.attempt{attempt}_failed"
                shutil.rmtree(failed, ignore_errors=True)
                shutil.move(out, failed)
                print(f"attempt {attempt} failed ({e}); kept in {failed}; retrying", flush=True)
                time.sleep(20)


if __name__ == "__main__":
    main()
