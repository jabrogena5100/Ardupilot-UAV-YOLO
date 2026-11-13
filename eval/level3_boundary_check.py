#!/usr/bin/env python3
"""
eval/level3_boundary_check.py — is a Model B threshold of 0.10 (the lower edge of the 0.10-0.70 grid) a real optimum or a grid artefact?

All three Model B v2 seeds were selected at tau = 0.10, the lowest grid value, so the F0.5 maximum may lie below it. This re-runs the SAME three
checkpoints on the SAME cached calibration frames (local_data/level3/calibration/img, calibration scenarios only) at a detector confidence floor of 0.01
and applies the SAME predetermined rule (argmax F0.5, ties to the higher tau) to the extended grid 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, ... 0.70.
No held-out seed is touched (`require_calibration`). The result is an amendment of the threshold file's grid, made AFTER seeing the boundary
selection, and is documented as such; it does not look at any pilot or held-out result.

    python3 eval/level3_boundary_check.py detect --out local_data/level3/calibration      # heavy: 3 models x 640 frames on CPU
    python3 eval/level3_boundary_check.py select --out local_data/level3/calibration/boundary_check.json
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval import level3_calibrate_detector as CD  # noqa: E402
from eval import level3_scenarios as SC  # noqa: E402

ARMS = ["B2s0", "B2s1", "B2s2"]
FLOOR = 0.01
TAUS_EXT = [0.01, 0.02, 0.03, 0.05, 0.07] + CD.TAUS


def detect(out_dir):
    import cv2
    from eval.level3_pairs import ARMS as ALL
    from perception import detector as D
    from sim.camera import Camera, CameraIntrinsics, GridSpec
    from sim.frame_synth import CameraSpec
    recs = [json.loads(line) for line in open(os.path.join(out_dir, "records.jsonl")) if line.strip()]
    SC.require_calibration([r["scenario"] for r in recs])
    g, spec = GridSpec.from_miles(1.0, 10), CameraSpec()
    for a in ARMS:
        dp = os.path.join(out_dir, f"dets_{a}_floor001.json")
        if os.path.exists(dp):
            continue
        D.verify_weights(os.path.join(ROOT, ALL[a][0]), None)
        pr = D.make_ultralytics_predictor(os.path.join(ROOT, ALL[a][0]), FLOOR)
        t0, out = time.time(), []
        for k, r in enumerate(recs):
            img = cv2.imread(os.path.join(out_dir, "img", r["img"]))
            cam = Camera.from_pose(type("P", (), {"lat": r["pose"]["lat"], "lon": r["pose"]["lon"], "alt": CD.ALT,
                                                  "heading_deg": r["pose"]["heading_deg"]})(),
                                   CameraIntrinsics(img.shape[1], img.shape[0], spec.hfov_deg), CD.LAT0, CD.LON0, spec.pitch_deg)
            out.append([{"conf": float(c), "cell": D.box_cell([float(v) for v in b], cam, g, "bottom_center")} for b, c in pr(img)])
            if k % 100 == 0:
                print(f"[{a}] {k}/{len(recs)} {time.time() - t0:.0f}s", flush=True)
        json.dump(out, open(dp + ".tmp", "w"))
        os.replace(dp + ".tmp", dp)
        del pr
        print(f"[boundary] {a} done {time.time() - t0:.0f}s", flush=True)


def select(out_path):
    base = os.path.dirname(out_path)
    d = json.load(open(os.path.join(base, "frames.json")))
    SC.require_calibration(d["scenarios"])
    res = {"grid": TAUS_EXT, "floor": FLOOR, "arms": {}}
    for a in ARMS:
        dets = json.load(open(os.path.join(base, f"dets_{a}_floor001.json")))
        frames = [dict(f, dets={a: dets[k]}) for k, f in enumerate(d["frames"])]
        sweep = [CD.metrics(frames, a, t) for t in TAUS_EXT]
        best = max(sweep, key=lambda r: (r["f0.5"], r["tau"]))
        res["arms"][a] = {"tau": best["tau"], "sweep": sweep}
        print(a, "tau", best["tau"], {k: best[k] for k in ("precision", "recall", "f0.5", "fp_per_frame", "empty_frame_alarm_rate", "latency_median_s")})
        print("   ", [(r["tau"], r["f0.5"]) for r in sweep if r["tau"] <= 0.2])
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["detect", "select"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    detect(a.out) if a.stage == "detect" else select(a.out)
