#!/usr/bin/env python3
"""
eval/level3_summarize.py — one row per Level 3 run directory (as written by eval/level3_pairs.py run).

Columns: arm, scenario, tau, det_true, det_false, precision, detection_lag_s (fog: first TRUE detection since ignition; blank = never),
visited_frac (coverage at the end), fire_cells (cells that burned), fire_cells_visited, first_look_median_s (median seconds from a
cell's ignition to its first visit, over visited fire cells), pct_investigate / pct_traverse (share of 1 Hz HUD lines by planner mode),
frames / infer_ms_median (agent perception log). Descriptive only: no test, no claim.

    python3 eval/level3_summarize.py RUN_DIR [RUN_DIR ...] > summary.csv
"""
import csv
import glob
import json
import os
import re
import statistics
import sys


def kv(path):
    d = {}
    for line in open(path):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            d[k] = v
    return d


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summarize(run_dir):
    end = kv(glob.glob(os.path.join(run_dir, "experiment_end_*.txt"))[0])
    cfg = [json.load(open(p)) for p in glob.glob(os.path.join(run_dir, "manifest.json"))][0]
    tag = end["tag"]
    arm, scen = (tag.split("-")[1], tag.split("-")[2]) if tag.startswith("l3-") else (tag, "")
    row = {"arm": arm, "scenario": scen, "tau": cfg.get("conf_threshold"), "reason": end.get("reason"),
           "det_true": int(end.get("det_true", 0)), "det_false": int(end.get("det_false", 0)),
           "precision": f(end.get("precision")), "detection_lag_s": f(end.get("detection_lag_s")),
           "visited_frac": f(end.get("visited_frac"))}
    fc = glob.glob(os.path.join(run_dir, "fire_cells_*.csv"))
    if fc:
        rows = list(csv.DictReader(open(fc[0])))
        looks = [float(r["first_visit_t"]) - float(r["ignite_t"]) for r in rows if r["first_visit_t"] != ""]
        row["fire_cells"] = len(rows)
        row["fire_cells_visited"] = len(looks)
        row["first_look_median_s"] = round(statistics.median(looks), 1) if looks else None
    modes = {}
    infer, frames = None, None
    for p in glob.glob(os.path.join(run_dir, "agent_*.log")):
        for line in open(p, errors="ignore"):
            m = re.search(r"mode=([A-Za-z_]+)", line)
            if m:
                modes[m.group(1)] = modes.get(m.group(1), 0) + 1
            m = re.search(r"(\d+) frames, (\d+) boxes.*median=(\d+)", line)
            if m:
                frames, infer = int(m.group(1)), int(m.group(3))
    tot = sum(modes.values()) or 1
    row["pct_investigate"] = round(100 * modes.get("INVESTIGATE", 0) / tot, 1)
    row["pct_traverse"] = round(100 * modes.get("TRAVERSE", 0) / tot, 1)
    row["frames"], row["infer_ms_median"] = frames, infer
    gate = os.path.join(run_dir, "start_gate.json")
    row["gate_wait_s"] = json.load(open(gate))["wait_s"] if os.path.exists(gate) else None
    meta = os.path.join(run_dir, "run_meta.json")
    row["wall_s"] = json.load(open(meta)).get("wall_s") if os.path.exists(meta) else None
    return row


def main():
    rows = [summarize(d) for d in sys.argv[1:]]
    cols = list(rows[0].keys()) if rows else []
    w = csv.DictWriter(sys.stdout, cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)


if __name__ == "__main__":
    main()
