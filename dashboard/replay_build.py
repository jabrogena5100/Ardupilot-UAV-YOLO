#!/usr/bin/env python3
"""
dashboard/replay_build.py — turn each saved SITL telemetry log (sitl_work/mav.tlog)
into a small trajectory cache the replay dashboard can serve cheaply.

    ~/venv-dronekit/bin/python dashboard/replay_build.py            # all runs with a tlog
    ~/venv-dronekit/bin/python dashboard/replay_build.py --force    # rebuild

Needs pymavlink (the only venv that has it is venv-dronekit); the dashboard server
itself does not. Reads the tlog only; writes only to local_data/level3/_replay_cache/.

Time alignment (no guessing): experiment t = 0 is the moment fog's start gate opened,
i.e. the POSE altitude had been >= gate min_alt for gate settle_s. The same rule is applied
to the tlog's GLOBAL_POSITION_INT (relative_alt) to find t = 0 in tlog wall-clock. As an
independent check, the first time the telemetry enters each cell is compared with fog's own
`visited_first_t` for that cell; the median/max residual is stored as `align_qc` and shown
in the UI. Samples are real tlog messages (first one at/after each whole second); nothing
is interpolated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from dashboard import replay_data as rd  # noqa: E402

R_EARTH = 6378137.0


def build(run: dict) -> dict:
    from pymavlink import mavutil
    d = os.path.join(rd.ROOT, run["dir"])
    data = rd.load_run(run["id"])
    meta = data["meta"]
    if None in meta["origin"]:
        return {"ok": False, "reason": "args.json missing in this run dir: grid origin unknown, cannot place the path on the grid"}
    lat0, lon0 = float(meta["origin"][0]), float(meta["origin"][1])
    N, side, cell = meta["N"], meta["side_m"], meta["cell_m"]
    half = side / 2.0

    cfg = rd._yaml(rd._first(__import__("glob").glob(os.path.join(d, "l3-*.yaml"))) or "") or {}
    gate = ((cfg.get("fire_model") or {}).get("start_gate") or {})
    gate_alt = float(gate.get("min_alt_m", 100.0))
    settle = float(gate.get("settle_s", 5.0))

    m = mavutil.mavlink_connection(os.path.join(d, "sitl_work", "mav.tlog"), robust_parsing=True)
    raw = []  # (wall_ts, n, e, alt_rel, hdg, gs)
    while True:
        x = m.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
        if x is None:
            break
        if x.lat == 0 and x.lon == 0:
            continue
        lat, lon = x.lat / 1e7, x.lon / 1e7
        n = math.radians(lat - lat0) * R_EARTH
        e = math.radians(lon - lon0) * R_EARTH * math.cos(math.radians((lat + lat0) / 2.0))
        hdg = None if x.hdg == 65535 else x.hdg / 100.0
        raw.append((x._timestamp, n, e, x.relative_alt / 1000.0, hdg, math.hypot(x.vx, x.vy) / 100.0))
    if not raw:
        raise RuntimeError("no GLOBAL_POSITION_INT in tlog")

    # t = 0: first moment relative alt has been >= gate_alt continuously for settle_s
    t_gate = None
    since = None
    for ts, _n, _e, alt, *_ in raw:
        if alt >= gate_alt:
            since = ts if since is None else since
            if ts - since >= settle:
                t_gate = ts
                break
        else:
            since = None
    if t_gate is None:
        return {"ok": False, "reason": f"relative altitude never held >= {gate_alt} m for {settle} s in the tlog"}

    # 1 Hz thinning of real messages, experiment time only (t >= 0)
    samples, nxt = [], 0.0
    for ts, n, e, alt, hdg, gs in raw:
        t = ts - t_gate
        if t < nxt:
            continue
        nxt = math.floor(t) + 1.0
        samples.append([round(t, 1), round(n, 1), round(e, 1), round(alt, 1),
                        None if hdg is None else round(hdg), round(gs, 1)])

    # independent alignment check against fog's own first-visit times
    resid = []
    first_in = {}
    for ts, n, e, *_ in raw:
        i, j = int((e + half) // cell), int((n + half) // cell)
        if 0 <= i < N and 0 <= j < N and (i, j) not in first_in:
            first_in[(i, j)] = ts - t_gate
    for c in data["cells"] or []:
        if c["first_t"] is not None and (c["i"], c["j"]) in first_in:
            resid.append(first_in[(c["i"], c["j"])] - c["first_t"])
    qc = None
    if resid:
        qc = {"cells_compared": len(resid), "median_s": round(statistics.median(resid), 1),
              "median_abs_s": round(statistics.median(abs(r) for r in resid), 1),
              "max_abs_s": round(max(abs(r) for r in resid), 1),
              "n_over_10s": sum(1 for r in resid if abs(r) > 10)}

    tl = [s[0] for s in samples]
    gaps = [[tl[k], tl[k + 1]] for k in range(len(tl) - 1) if tl[k + 1] - tl[k] > 3.0]
    return {"ok": True, "fields": ["t", "north_m", "east_m", "alt_rel_m", "hdg_deg", "groundspeed_ms"],
            "samples": samples, "gaps": gaps, "gate_alt_m": gate_alt, "settle_s": settle,
            "t_end_s": tl[-1] if tl else None, "align_qc": qc,
            "source": "sitl_work/mav.tlog GLOBAL_POSITION_INT, 1 Hz thinning, no interpolation"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", help="run id like pilot2/l3-A0-1001")
    a = ap.parse_args()
    os.makedirs(rd.CACHE, exist_ok=True)
    for run in rd.discover()["runs"]:
        if a.only and run["id"] != a.only:
            continue
        out = os.path.join(rd.CACHE, f"{run['pilot']}__{run['name']}.traj.json")
        if not run["has_telemetry"]:
            print(f"skip   {run['id']}: no tlog")
            continue
        if os.path.exists(out) and not a.force:
            print(f"cached {run['id']}")
            continue
        try:
            res = build(run)
        except Exception as ex:  # keep going; the UI reports the gap
            res = {"ok": False, "reason": f"{type(ex).__name__}: {ex}"}
        if res.get("ok"):
            with open(out, "w") as f:
                json.dump(res, f, separators=(",", ":"))
            print(f"built  {run['id']}: {len(res['samples'])} samples, align_qc={res['align_qc']}")
        else:
            print(f"FAILED {run['id']}: {res['reason']}")


if __name__ == "__main__":
    main()
