#!/usr/bin/env python3
"""
eval/level3_calibrate_detector.py — choose each arm's detection threshold tau on CALIBRATION scenarios, by a rule fixed in advance.

Stage `collect` (heavy; renders and runs every arm once at a confidence floor of 0.10):
  For each calibration scenario FireSim (frozen settings, eval/level3_fire.json) is stepped in time, and the camera (oblique, 60 deg
  below horizontal, 78 deg HFOV, 120 m, heading = direction of travel, 10 m/s) flies short PASSES:
    * fire passes (10 per scenario, one per 60 s): approach a random burning cell from ~240 m, random heading and lateral offset,
      8 frames 4 s apart;
    * empty-ish passes (10 per scenario): random start and heading anywhere in the grid, 8 frames (the fire may or may not be in view).
  Each frame is rendered once and every arm's detector runs on it. Sim-side truth per frame: the burning cells at that instant and the
  smoke visibility from the renderer's alpha (alpha >= 0.08 over >= 1 % of the frame = VISIBLE; > 0 but < 1 % = AMBIGUOUS, excluded
  from recall and from the negative-frame rate; 0 = EMPTY). These are the dataset's own label-validity constants.
Stage `select` (cheap, offline): for tau in 0.10, 0.15, ... 0.70 per arm compute
    precision      TP / (TP + FP) over detections (a detection is TRUE when a BURNING cell lies within 1 cell (Chebyshev) of its cell at
                   that instant: the same rule as Fog.ingest_detection)
    recall         visible frames with >= 1 TRUE detection / visible frames
    FP / frame     FALSE detections per frame (all frames)
    empty-frame false-alarm rate   EMPTY frames with >= 1 detection / EMPTY frames
    latency        in fire passes: seconds from the first VISIBLE frame to the first frame with a TRUE detection (median over passes that
                   had a visible frame; passes never detected are counted as censored, not dropped)
PREDETERMINED SELECTION RULE (written before any frame was rendered): per arm, tau = argmax over the grid of F0.5 = 1.25 P R / (0.25 P + R)
(precision-weighted because a false alarm misdirects the planner), P = pooled precision (0 if no detections), R = recall; ties -> the higher
tau; if the maximum F0.5 is 0 the arm is flagged UNUSABLE and tau = 0.70. The held-out scenarios play no part; their seeds are refused.

    python3 eval/level3_calibrate_detector.py collect --out local_data/level3/calibration
    python3 eval/level3_calibrate_detector.py select  --in local_data/level3/calibration/frames.json --out eval/level3_thresholds.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval import level3_scenarios as SC  # noqa: E402

TAUS = [round(0.10 + 0.05 * k, 2) for k in range(13)]
ALT, SPEED, DT, FRAMES_PER_PASS = 120.0, 10.0, 4.0, 8
VISIBLE_FRAC, ALPHA_THR = 0.01, 0.08
LAT0, LON0 = SC.LAT0, SC.LON0


def _heldout_guard(seeds):
    SC.require_calibration(seeds)


def collect(out_dir, arms, passes_per_kind=10, T=600, limit_scen=None):
    """Two resumable stages (one 1 vCPU / 2 GB VM cannot hold six models at once, and a run lost to a restart is 75 min):
    1. render: every frame is rendered ONCE (same pixels for every arm) to out_dir/img/*.png (lossless) with its simulation-side truth
       in out_dir/records.jsonl (appended per pass; a pass already recorded is skipped on restart, the FireSim and rng are replayed);
    2. detect: one model at a time over the PNGs, floor 0.10, written to out_dir/dets_<arm>.json (an arm already written is skipped).
    Then frames.json is assembled (the same format `select` reads)."""
    from eval.level3_pairs import ARMS
    from mission.geo import ne_to_latlon
    from sim.camera import GridSpec
    from sim.frame_synth import CameraSpec, FrameSynth, PlumeModel, sources_from_fire_cells
    from wildfire_sim import BURNING, FireSim
    import cv2

    fire = json.load(open(os.path.join(ROOT, "eval", "level3_fire.json")))
    scen = SC.load()["calibration"][:limit_scen]
    _heldout_guard([s["seed"] for s in scen])
    g = GridSpec.from_miles(1.0, 10)
    cam_spec = CameraSpec()
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    rec_path = os.path.join(out_dir, "records.jsonl")
    done = set()
    if os.path.exists(rec_path):
        for line in open(rec_path):
            if line.strip():
                r = json.loads(line)
                done.add((r["scenario"], r["pass"]))
    t_start = time.time()
    for sc in scen:
        rng = np.random.default_rng(np.random.SeedSequence([sc["seed"], 4242]))
        synth = None
        sim = FireSim(10, p_ignite=fire["p_ignite"], neighborhood=fire["neighborhood"], burn_steps=fire["burn_steps"],
                      dt_seconds=1.0, burn_time_dist="fixed", wind_direction_deg=0.0, wind_strength=0.0, random_seed=sc["fire_seed"])
        sim.ignite_many([sc["ignition"]])
        sched = []
        for k in range(1, passes_per_kind + 1):
            sched.append((60 * k, "fire"))
            sched.append((60 * k + 30, "free"))
        t_sim = 0
        for t0, kind in sched:
            while t_sim < t0:
                sim.step()
                t_sim += 1
            snap0 = sim.snapshot()
            burning = [(i, j) for i in range(10) for j in range(10) if snap0[i][j] == BURNING]
            if kind == "fire" and burning:
                ti, tj = burning[int(rng.integers(len(burning)))]
                tn, te = g.center_ne(ti, tj)
                hdg = float(rng.uniform(0, 360))
                lat_off = float(rng.uniform(-60, 60))
                ux, uy = math.cos(math.radians(hdg)), math.sin(math.radians(hdg))          # (north, east) unit of travel
                n0 = tn - 240.0 * ux - lat_off * uy
                e0 = te - 240.0 * uy + lat_off * ux
            else:
                kind = "free"
                hdg = float(rng.uniform(0, 360))
                n0, e0 = (float(x) for x in rng.uniform(-700, 700, 2))
                ux, uy = math.cos(math.radians(hdg)), math.sin(math.radians(hdg))
            skip = (sc["seed"], f"{kind}{t0}") in done
            if not skip and synth is None:
                synth = FrameSynth(sc["seed"], LAT0, LON0, camera=cam_spec, plume=PlumeModel())
            recs = []
            for m in range(FRAMES_PER_PASS):
                if m:
                    for _ in range(int(DT)):
                        sim.step()
                        t_sim += 1
                if skip:
                    continue
                snap = sim.snapshot()
                burning = [(i, j) for i in range(10) for j in range(10) if snap[i][j] == BURNING]
                src = sources_from_fire_cells([(i, j, 1.0) for i, j in burning], g, PlumeModel(), sc["seed"])
                n, e = n0 + SPEED * DT * m * ux, e0 + SPEED * DT * m * uy
                lat, lon = ne_to_latlon(n, e, LAT0, LON0)
                img, alpha = synth.render_frame(lat, lon, ALT, hdg, src, f"{kind}{t0}-{m}")
                frac = float((alpha >= ALPHA_THR).mean())
                vis = "visible" if frac >= VISIBLE_FRAC else ("ambiguous" if alpha.max() > 0 and frac > 0 else "empty")
                name = f"{sc['seed']}_{kind}{t0}_{m}.png"
                cv2.imwrite(os.path.join(out_dir, "img", name), img)
                recs.append({"scenario": sc["seed"], "pass": f"{kind}{t0}", "kind": kind, "frame": m, "t_sim": t_sim,
                             "visibility": vis, "visible_frac": round(frac, 4), "burning": burning, "img": name,
                             "pose": {"lat": lat, "lon": lon, "heading_deg": hdg}})
            if recs:
                with open(rec_path, "a") as fh:                       # appended only once the whole pass is rendered
                    for r in recs:
                        fh.write(json.dumps(r) + "\n")
                print(f"[render] scenario {sc['seed']} pass {kind}{t0} done, {time.time() - t_start:.0f}s", flush=True)
    recs = [json.loads(line) for line in open(rec_path) if line.strip()]
    recs = [r for r in recs if r["scenario"] in {s["seed"] for s in scen}]
    # detect: one model resident at a time
    from perception import detector as D
    from sim.camera import Camera, CameraIntrinsics
    used = [a for a in arms if ARMS[a]]
    for a in used:
        dp = os.path.join(out_dir, f"dets_{a}.json")
        if os.path.exists(dp):
            continue
        D.verify_weights(os.path.join(ROOT, ARMS[a][0]), None)
        pr = D.make_ultralytics_predictor(os.path.join(ROOT, ARMS[a][0]), 0.10)
        out = []
        for r in recs:
            img = cv2.imread(os.path.join(out_dir, "img", r["img"]))
            cam = Camera.from_pose(type("P", (), {"lat": r["pose"]["lat"], "lon": r["pose"]["lon"], "alt": ALT,
                                                  "heading_deg": r["pose"]["heading_deg"]})(),
                                   CameraIntrinsics(img.shape[1], img.shape[0], cam_spec.hfov_deg), LAT0, LON0, cam_spec.pitch_deg)
            out.append([{"conf": float(c), "cell": D.box_cell([float(v) for v in b], cam, g, "bottom_center")} for b, c in pr(img)])
        json.dump(out, open(dp + ".tmp", "w"))
        os.replace(dp + ".tmp", dp)
        del pr
        import gc
        gc.collect()
        print(f"[detect] {a} done, {time.time() - t_start:.0f}s", flush=True)
    frames = []
    for k, r in enumerate(recs):
        r = {x: y for x, y in r.items() if x not in ("img", "pose")}
        r["dets"] = {}
        frames.append(r)
    alld = {a: json.load(open(os.path.join(out_dir, f"dets_{a}.json"))) for a in used}
    for k, f in enumerate(frames):
        f["dets"] = {a: alld[a][k] for a in used}
    json.dump({"alt_m": ALT, "pitch_deg": cam_spec.pitch_deg, "hfov_deg": cam_spec.hfov_deg, "speed_mps": SPEED, "dt_s": DT,
               "frames_per_pass": FRAMES_PER_PASS, "fire": {k: fire[k] for k in ("p_ignite", "burn_steps")},
               "scenarios": [s["seed"] for s in scen], "arms": used, "frames": frames},
              open(os.path.join(out_dir, "frames.json"), "w"))
    print("[collect] wrote", os.path.join(out_dir, "frames.json"))


def _true_det(cell, burning, r=1):
    if cell is None:
        return False
    return any(max(abs(cell[0] - i), abs(cell[1] - j)) <= r for i, j in burning)


def metrics(frames, arm, tau):
    tp = fp = 0
    vis = vis_hit = emp = emp_alarm = 0
    nframes = len(frames)
    first_vis, first_tp = {}, {}
    for f in frames:
        ds = [d for d in f["dets"][arm] if d["conf"] >= tau and d["cell"] is not None]
        t = [_true_det(d["cell"], f["burning"]) for d in ds]
        tp += sum(t)
        fp += len(t) - sum(t)
        if f["visibility"] == "visible":
            vis += 1
            vis_hit += any(t)
        elif f["visibility"] == "empty":
            emp += 1
            emp_alarm += bool(ds)
        if f["kind"] == "fire":
            key = (f["scenario"], f["pass"])
            if f["visibility"] == "visible":
                first_vis.setdefault(key, f["frame"])
            if any(t):
                first_tp.setdefault(key, f["frame"])
    lat, cens = [], 0
    for key, mv in first_vis.items():
        mt = first_tp.get(key)
        if mt is None:
            cens += 1
        else:
            lat.append(max(0, mt - mv) * DT)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = vis_hit / vis if vis else 0.0
    f05 = 1.25 * prec * rec / (0.25 * prec + rec) if prec + rec > 0 else 0.0
    return {"tau": tau, "tp": tp, "fp": fp, "precision": round(prec, 4), "recall": round(rec, 4), "f0.5": round(f05, 4),
            "fp_per_frame": round(fp / nframes, 4), "empty_frame_alarm_rate": round(emp_alarm / emp, 4) if emp else None,
            "latency_median_s": (float(np.median(lat)) if lat else None), "passes_detected": len(lat), "passes_censored": cens,
            "visible_frames": vis, "empty_frames": emp, "frames": nframes}


def select(in_path, out_path):
    d = json.load(open(in_path))
    SC.require_calibration(d["scenarios"])
    out = {"rule": __doc__.split("PREDETERMINED SELECTION RULE")[1].split("The held-out")[0].strip(), "calibration_scenarios": d["scenarios"],
           "camera": {"pitch_deg": d["pitch_deg"], "hfov_deg": d["hfov_deg"], "alt_m": d["alt_m"]}, "fire": d["fire"], "arms": {}}
    for arm in d["arms"]:
        sweep = [metrics(d["frames"], arm, tau) for tau in TAUS]
        best = max(sweep, key=lambda r: (r["f0.5"], r["tau"]))
        usable = best["f0.5"] > 0
        out["arms"][arm] = {"tau": best["tau"] if usable else 0.70, "usable": usable, "sweep": sweep}
        print(f"{arm}: tau={out['arms'][arm]['tau']} usable={usable} P={best['precision']} R={best['recall']} F0.5={best['f0.5']}")
    json.dump(out, open(out_path, "w"), indent=1)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--arms", nargs="+", default=["A0", "A1", "A2", "B2s0", "B2s1", "B2s2"])
    c.add_argument("--passes", type=int, default=10)
    c.add_argument("--limit-scenarios", type=int, default=None)
    s = sub.add_parser("select")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "collect":
        collect(a.out, a.arms, a.passes, limit_scen=a.limit_scenarios)
    else:
        select(a.inp, a.out)


if __name__ == "__main__":
    main()
