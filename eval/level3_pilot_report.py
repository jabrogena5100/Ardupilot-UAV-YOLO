#!/usr/bin/env python3
"""
eval/level3_pilot_report.py — descriptive per-run record separating GEOMETRY (could the camera see the smoke?) from DETECTION (did YOLO find it?).

For each Level 3 run directory (results dir written by eval/level3_pairs.py run; needs sitl_work/mav.tlog, fire_cells_*.csv, detections_*.csv,
agent_A.log, experiment_end_*.txt):

  first_smoke_visible_s   first perception frame (period 4 s) at which the plume base of ANY burning cell was inside the image (eval/level3_visibility.py,
                          flown trajectory from the telemetry log, burning cells from fire_cells_*.csv ignite times); this uses fire truth OFFLINE for analysis only
  visible_frames          number of such frames (opportunities) and frames_ignition_cell for the ignition cell alone
  first_true_det_s        first TRUE detection (fog truth at ingest) after the start gate; det_true / det_false counts; first_false_det_s
  detection_gap_s         first_true_det_s - first_smoke_visible_s (None when one is missing)
  visited_frac, fire_cells_visited
  investigate_s           HUD seconds spent in INVESTIGATE mode; investigate_targets = distinct target cells in order of first appearance
                          (route changes caused by detections: the baseline has none by construction)
Descriptive only; no test, no claim.

    python3 eval/level3_pilot_report.py RUN_DIR [RUN_DIR ...] [--json out.json]
"""
import csv
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.level3_visibility import load_track, visibility  # noqa: E402
from sim.camera import CameraIntrinsics, GridSpec  # noqa: E402


def kv(path):
    d = {}
    for line in open(path):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            d[k] = v
    return d


def one(d, period=4.0, pitch=60.0):
    end = kv(glob.glob(os.path.join(d, "experiment_end_*.txt"))[0])
    tag = end["tag"]
    arm, scen = tag.split("-")[1], tag.split("-")[-1]
    tr = load_track(os.path.join(d, "sitl_work", "mav.tlog"))
    grid, intr = GridSpec.from_miles(1.0, 10), CameraIntrinsics(1280, 720, 78.0)
    fire = list(csv.DictReader(open(glob.glob(os.path.join(d, "fire_cells_*.csv"))[0])))
    first, frames_all, ign_cell, frames_ign = None, 0, None, 0
    for r in fire:
        t_on = max(0.0, float(r["ignite_t"]))
        c = (int(r["i"]), int(r["j"]))
        v = visibility(tr, grid, intr, pitch, (t_on, 600.0), period, 40.0, 40.0, cells=[c])[c]
        frames_all += v["frames_visible"]
        if v["first_visible_s"] is not None and (first is None or v["first_visible_s"] < first):
            first = v["first_visible_s"]
        if float(r["ignite_t"]) <= 0.5:
            ign_cell, frames_ign = c, v["frames_visible"]
    dets = list(csv.DictReader(open(glob.glob(os.path.join(d, "detections_*.csv"))[0])))
    tp = [float(r["t"]) for r in dets if r["correct"] == "1" and float(r["t"]) >= 0]
    fp = [float(r["t"]) for r in dets if r["correct"] == "0" and float(r["t"]) >= 0]
    inv, targets = 0, []
    for line in open(os.path.join(d, "agent_A.log")):
        m = re.search(r"mode=INVESTIGATE→(\d+),(\d+)", line)
        if m:
            inv += 1
            c = (int(m.group(1)), int(m.group(2)))
            if c not in targets:
                targets.append(c)
    return {
        "run": os.path.basename(d.rstrip("/")), "arm": arm, "scenario": scen, "ignition_cell": ign_cell,
        "first_smoke_visible_s": first, "visible_frames": frames_all, "frames_ignition_cell": frames_ign,
        "first_true_det_s": min(tp) if tp else None, "det_true": len(tp), "det_false": len(fp), "first_false_det_s": min(fp) if fp else None,
        "detection_gap_s": (round(min(tp) - first, 1) if tp and first is not None else None),
        "visited_frac": float(end["visited_frac"]), "burnt_frac_end": float(end.get("burnt_frac", "nan")),
        "fire_cells": len(fire), "fire_cells_visited": sum(1 for r in fire if r["visited"] == "1"),
        "investigate_s": inv, "investigate_targets": targets,
    }


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else ""
    if js:
        a = [x for x in a if x != js]
    rows = [one(d) for d in a]
    for r in rows:
        print(json.dumps(r))
    if js:
        json.dump(rows, open(js, "w"), indent=1)


if __name__ == "__main__":
    main()
