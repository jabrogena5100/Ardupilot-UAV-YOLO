#!/usr/bin/env python3
"""
eval/level3_visibility.py — geometric camera visibility of a candidate ignition cell along a recorded flight (no detector, no fire truth).

Input: the SITL telemetry log (`mav.tlog`) of a run. The baseline arm never reacts to fire, so its trajectory does not depend on where the
fire is; one recorded baseline flight therefore tells which cells the camera sweeps, and when, for ANY ignition cell. Output per cell, using the same
projection as the renderer (`sim/camera.py`, 60 deg pitch, 78 deg HFOV, 1280x720, look direction = vehicle heading as logged):

  frames_visible   frames (one every `--period` s, the perception rate) whose image contains the plume BASE (cell centre on the ground)
                   with `--margin` pixels to spare and a plume at least `--min-px` pixels tall (the plume is 80 m high)
  first_visible_s  time since the run clock start of the first such frame
  first_range_m    ground range of the cell centre at that frame

Run-clock start t=0 = first time relative altitude >= 100 m, plus the 5 s settle of the start gate (approximates fog's gate to within a second or two).
Used to choose calibration ignition cells BEFORE running simulations (rule in docs/experiments/level3_methodology.md section 12).

    python3 eval/level3_visibility.py RUN_DIR/sitl_work/mav.tlog [--window 0 600] [--json out.json]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mission.geo import latlon_to_ne  # noqa: E402
from sim.camera import Camera, CameraIntrinsics, CameraPose, GridSpec  # noqa: E402

LAT0, LON0 = 21.2970, -157.8170
PLUME_H_M = 80.0


def load_track(tlog):
    from pymavlink import mavutil
    m = mavutil.mavlink_connection(tlog, dt=False)
    rows = []
    while True:
        msg = m.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
        if msg is None:
            break
        rows.append((msg._timestamp, msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000.0, msg.hdg / 100.0))
    if not rows:
        raise SystemExit("no GLOBAL_POSITION_INT in " + tlog)
    a = np.array(rows)
    up = np.nonzero(a[:, 3] >= 100.0)[0]
    if len(up) == 0:
        raise SystemExit("vehicle never reached 100 m")
    t0 = a[up[0], 0] + 5.0
    n_e = np.array([latlon_to_ne(r[1], r[2], LAT0, LON0) for r in a])
    return {"t": a[:, 0] - t0, "n": n_e[:, 0], "e": n_e[:, 1], "alt": a[:, 3], "hdg": a[:, 4]}


def pose_at(tr, t):
    if t < tr["t"][0] or t > tr["t"][-1]:
        return None
    i = int(np.searchsorted(tr["t"], t))
    i = min(max(i, 1), len(tr["t"]) - 1)
    t0, t1 = tr["t"][i - 1], tr["t"][i]
    w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    lerp = lambda k: (1 - w) * tr[k][i - 1] + w * tr[k][i]          # noqa: E731
    d = ((tr["hdg"][i] - tr["hdg"][i - 1] + 180) % 360) - 180
    return float(lerp("n")), float(lerp("e")), float(lerp("alt")), float((tr["hdg"][i - 1] + w * d) % 360)


def visibility(tr, grid, intr, pitch, window, period, margin, min_px, cells=None):
    cells = cells or [(i, j) for i in range(grid.n_cells) for j in range(grid.n_cells)]
    out = {c: {"frames_visible": 0, "first_visible_s": None, "first_range_m": None, "times": []} for c in cells}
    t = window[0]
    while t <= window[1]:
        p = pose_at(tr, t)
        if p is not None and p[2] > 50.0:
            cam = Camera(intr, CameraPose(p[0], p[1], p[2], p[3], pitch))
            for c in cells:
                n, e = grid.center_ne(*c)
                uv, depth = cam.project(np.array([[n, e, 0.0], [n, e, PLUME_H_M]]))
                (ub, vb), (ut, vt) = uv
                if not np.all(np.isfinite(uv)):
                    continue
                inside = margin <= ub <= intr.width - margin and margin <= vb <= intr.height - margin
                if inside and (vb - vt) >= min_px:
                    o = out[c]
                    o["frames_visible"] += 1
                    o["times"].append(round(t, 1))
                    if o["first_visible_s"] is None:
                        o["first_visible_s"] = round(t, 1)
                        o["first_range_m"] = round(math.hypot(n - p[0], e - p[1]), 1)
        t += period
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tlog")
    ap.add_argument("--window", type=float, nargs=2, default=[0.0, 600.0])
    ap.add_argument("--period", type=float, default=4.0)
    ap.add_argument("--margin", type=float, default=40.0)
    ap.add_argument("--min-px", type=float, default=40.0)
    ap.add_argument("--pitch", type=float, default=60.0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    tr = load_track(a.tlog)
    grid = GridSpec.from_miles(1.0, 10)
    vis = visibility(tr, grid, CameraIntrinsics(1280, 720, 78.0), a.pitch, a.window, a.period, a.margin, a.min_px)
    print("grid of first_visible_s (rows j=9..0 north->south, cols i=0..9 west->east; '.' never):")
    for j in range(grid.n_cells - 1, -1, -1):
        print(f"j={j} " + " ".join(f"{vis[(i, j)]['first_visible_s']:6.0f}" if vis[(i, j)]["first_visible_s"] is not None else "     ." for i in range(grid.n_cells)))
    print("frames visible per cell:")
    for j in range(grid.n_cells - 1, -1, -1):
        print(f"j={j} " + " ".join(f"{vis[(i, j)]['frames_visible']:6d}" for i in range(grid.n_cells)))
    if a.json:
        json.dump({f"{i},{j}": v for (i, j), v in vis.items()}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
