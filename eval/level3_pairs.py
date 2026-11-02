#!/usr/bin/env python3
"""
eval/level3_pairs.py — paired Level 3 runs: identical scenario, only the perception arm differs.

An ARM is one of: `baseline` (perception disabled: coverage only), or a registry checkpoint (Model A seeds 0-2,
Model B v2 seeds 0-2). A SCENARIO is a seed that fixes everything else: the FireSim random seed, the rendered environment
(`camera_service.scenario_seed`), the agents' seeds, altitude, speed, camera and threshold. `plan` writes one config per
(scenario seed, arm) from a base config and REFUSES to write a set in which two arms of the same scenario differ in
anything but the perception arm. `run` executes one config end to end (SITL -> agents -> fog) sequentially.

    python3 eval/level3_pairs.py plan --base exp_swarm_perception_single.yaml --seeds 1 2 3 --out local_data/level3/plan1
    python3 eval/level3_pairs.py run  --config local_data/level3/plan1/l3-B2s0-seed1.yaml --python ~/venvs/yolo-cpu/bin/python

This produces RUNS, not conclusions: nothing here supports a claim about coverage, detection time or misdirection until
the planned matrix has actually been run with enough seeds and analysed.
"""
import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
ARM_KEYS = ("enabled", "weights", "weights_sha256", "model_tag")     # the ONLY perception keys allowed to differ


def make_config(base: dict, arm: str, seed: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["experiment_tag"] = f"l3-{arm}-seed{seed}"
    cfg.setdefault("fire_model", {})["random_seed"] = int(seed)
    p = cfg.setdefault("perception", {})
    p.setdefault("camera_service", {})["scenario_seed"] = int(seed)
    p["camera_service"]["enabled"] = True                 # same service for every arm: baseline just never asks it for frames
    for k in ARM_KEYS:
        p.pop(k, None)
    if ARMS[arm] is None:
        p["enabled"] = False
        p["model_tag"] = "baseline"
    else:
        p["enabled"] = True
        p["weights"], p["model_tag"] = ARMS[arm]
    for k, a in enumerate(cfg.get("agents", [])):         # agent seeds derive from the scenario seed, not from the arm
        a["seed"] = int(seed) * 100 + k + 1
    return cfg


def strip_arm(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c.pop("experiment_tag", None)
    for k in ARM_KEYS:
        c.get("perception", {}).pop(k, None)
    return c


def assert_paired(cfgs_by_arm: dict) -> None:
    """Every arm of one scenario must be identical once the arm keys and the tag are removed."""
    ref_arm = next(iter(cfgs_by_arm))
    ref = strip_arm(cfgs_by_arm[ref_arm])
    for arm, c in cfgs_by_arm.items():
        if strip_arm(c) != ref:
            raise ValueError(f"arm {arm} differs from {ref_arm} in more than the perception arm")


def plan(base_path: str, seeds, arms, out: str) -> list:
    base = yaml.safe_load(open(base_path))
    os.makedirs(out, exist_ok=True)
    rows = []
    for seed in seeds:
        cfgs = {a: make_config(base, a, seed) for a in arms}
        assert_paired(cfgs)
        for a, c in cfgs.items():
            path = os.path.join(out, f"{c['experiment_tag']}.yaml")
            yaml.safe_dump(c, open(path, "w"), sort_keys=False)
            rows.append({"seed": seed, "arm": a, "config": path})
    with open(os.path.join(out, "matrix.json"), "w") as f:
        json.dump({"base": base_path, "seeds": list(seeds), "arms": list(arms), "runs": rows}, f, indent=1)
    return rows


# ---------------------------------------------------------------- run

def _pids(names) -> set:
    """PIDs whose argv actually runs one of `names` (a program or script basename); shells that merely mention a name
    in a command string do not count."""
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


def run_one(config: str, python: str, out_dir: str, workdir: str, sitl_boot_s: float = 45.0) -> dict:
    cfg = yaml.safe_load(open(config))
    agents = cfg.get("agents", [])
    n = len(agents)
    ori = cfg["origin"]
    grid = cfg.get("grid", {})
    maxs = float(cfg.get("fog_tracker", {}).get("max_seconds", 420))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(workdir, exist_ok=True)
    ours = {"arducopter", "mavproxy.py", "sim_vehicle.py", "swarm_agent.py", "fog_tracker.py"}
    if _pids(ours):
        raise RuntimeError("SITL/agent/fog processes are already running; refusing to start (and to kill them)")
    procs = []
    log = lambda name: open(os.path.join(out_dir, name), "w")
    try:
        subprocess.Popen([sys.executable, os.path.join(ROOT, "launch_sitl_swarm.py"), "--count", str(n)], cwd=workdir,
                         stdout=log("sitl.log"), stderr=subprocess.STDOUT)
        time.sleep(sitl_boot_s + 15 * max(0, n - 1))
        vd = cfg.get("vehicle_defaults", {})
        for a in agents:
            cmd = [python, "-u", os.path.join(ROOT, "swarm_agent.py"), "--conn", a["conn"], "--origin-lat", str(ori["lat"]),
                   "--origin-lon", str(ori["lon"]), "--mission", a.get("mission", "grid_frontier"), "--alt", str(vd.get("alt", 120)),
                   "--vxy-max", str(vd.get("vxy_max", 15)), "--config", config, "--seed", str(a.get("seed", 0)),
                   "--strategy-tag", str(a.get("strategy_tag", "")), "--experiment-tag", cfg["experiment_tag"], "--verbose"]
            procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=log(f"agent_{a['name']}.log"), stderr=subprocess.STDOUT))
        deadline = time.time() + 240
        while time.time() < deadline:
            ready = sum(1 for a in agents if "READY" in open(os.path.join(out_dir, f"agent_{a['name']}.log")).read())
            if ready == n:
                break
            time.sleep(3)
        else:
            raise RuntimeError("agents did not reach READY")
        fog_log = os.path.join(out_dir, "fog.log")
        subprocess.run([python, "-u", os.path.join(ROOT, "fog_tracker.py"), "--origin-lat", str(ori["lat"]),
                        "--origin-lon", str(ori["lon"]), "--grid-miles", str(grid.get("miles", 1.0)),
                        "--grid-cells", str(grid.get("cells", 10)), "--end-when", "time", "--max-seconds", str(maxs),
                        "--config", config], cwd=ROOT, stdout=open(fog_log, "w"), stderr=subprocess.STDOUT, timeout=maxs + 180)
        m = re.search(r"wrote results to (\S+)", open(fog_log).read())
        res = m.group(1) if m else None
        if res and os.path.isdir(res):
            for f in os.listdir(res):
                shutil.copy(os.path.join(res, f), os.path.join(out_dir, f))
        return {"config": config, "results_dir": res, "out_dir": out_dir}
    finally:
        for p in procs:
            p.terminate()
        subprocess.run(["tmux", "kill-session", "-t", "sitl-swarm"], capture_output=True)
        time.sleep(3)
        for pid in _pids({"arducopter", "mavproxy.py", "sim_vehicle.py"}):
            try:
                os.kill(pid, 15)
            except ProcessLookupError:
                pass


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--base", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--out", default="")
    r.add_argument("--workdir", default="")
    a = ap.parse_args()
    if a.cmd == "plan":
        rows = plan(a.base, a.seeds, a.arms, a.out)
        print(f"wrote {len(rows)} configs to {a.out}")
    else:
        tag = os.path.splitext(os.path.basename(a.config))[0]
        out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.config)), "results", tag)
        print(json.dumps(run_one(a.config, a.python, out, a.workdir or os.path.join(out, "sitl_work"))))


if __name__ == "__main__":
    main()
