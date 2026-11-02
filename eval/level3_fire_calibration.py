#!/usr/bin/env python3
"""
eval/level3_fire_calibration.py — choose the fire-spread settings, on CALIBRATION scenarios only, by a rule fixed in advance.

FireSim is stepped offline (it is deterministic given its seed, independent of the agents) for every calibration scenario and every
(p_ignite, burn_steps) candidate over the mission horizon T. PREDETERMINED RULE (written before the sweep was run):

  a candidate is ADMISSIBLE if, in EVERY calibration scenario,
      LO <= (burning + burnt cells at T) / N^2 <= HI          (the fire is neither trivial nor saturating: room for coverage to matter)
      and at least one cell is still burning at T             (a target exists for the whole run)
      and at least EARLY cells are burning at t = 60 s        (a detectable event exists early enough for latency to be measured)

  RULE VERSION 1 (before any sweep): LO=0.08, HI=0.30, EARLY=3, grid p in {.002,.003,.005,.008,.012,.02} x burn in {150,300,450}.
      Result: NO candidate admissible (fire growth is a steep percolation-like transition between p=0.003 and 0.005; every candidate
      either saturated or had < 3 cells at 60 s). Recorded in docs/experiments/level3_calibration_v1.md.
  RULE VERSION 2 (AMENDED after seeing the version-1 CALIBRATION sweep only; held-out scenarios never run): LO=0.05, HI=0.35,
      EARLY=2, finer grid p in {.0020,.0025,.0030,.0035} x burn in {300,450,600}. The amendment is disclosed, not hidden.
  among admissible candidates choose the one whose MEAN burned fraction at T is closest to 0.20; ties -> smaller p_ignite, then larger burn_steps.
  If none is admissible the sweep is reported as failed and nothing is frozen.

Held-out seeds are refused. Output: JSON (all candidates) and, with --freeze, the chosen settings written to eval/level3_fire.json.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval import level3_scenarios as SC  # noqa: E402
from wildfire_sim import BURNING, BURNT, FireSim  # noqa: E402

N, T_DEFAULT = 10, 600
RULE_VERSION = 2
P_GRID = (0.0020, 0.0025, 0.0030, 0.0035)
BURN_GRID = (300, 450, 600)
LO, HI, EARLY, TARGET = 0.05, 0.35, 2, 0.20
OUT = os.path.join(ROOT, "eval", "level3_fire.json")


def run(p, burn, scen, T):
    sim = FireSim(N, p_ignite=p, neighborhood="von_neumann", burn_steps=burn, dt_seconds=1.0, burn_time_dist="fixed",
                  wind_direction_deg=0.0, wind_strength=0.0, random_seed=scen["fire_seed"])
    sim.ignite_many([scen["ignition"]])
    burning_at_60 = 0
    for t in range(1, T + 1):
        sim.step()
        if t == 60:
            burning_at_60 = sum(1 for r in sim.snapshot() for v in r if v == BURNING)
    snap = sim.snapshot()
    burning = sum(1 for r in snap for v in r if v == BURNING)
    burnt = sum(1 for r in snap for v in r if v == BURNT)
    return {"seed": scen["seed"], "burned_frac": (burning + burnt) / N ** 2, "burning_at_T": burning, "burning_at_60s": burning_at_60}


def sweep(T):
    cal = SC.load()["calibration"]
    SC.require_calibration([s["seed"] for s in cal])
    rows = []
    for p in P_GRID:
        for b in BURN_GRID:
            per = [run(p, b, s, T) for s in cal]
            ok = all(LO <= r["burned_frac"] <= HI and r["burning_at_T"] >= 1 and r["burning_at_60s"] >= EARLY for r in per)
            mean = sum(r["burned_frac"] for r in per) / len(per)
            rows.append({"p_ignite": p, "burn_steps": b, "admissible": ok, "mean_burned_frac": round(mean, 4), "per_scenario": per})
    adm = [r for r in rows if r["admissible"]]
    best = min(adm, key=lambda r: (abs(r["mean_burned_frac"] - TARGET), r["p_ignite"], -r["burn_steps"])) if adm else None
    return rows, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=T_DEFAULT)
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()
    rows, best = sweep(a.T)
    for r in rows:
        print(f"p={r['p_ignite']:<6} burn={r['burn_steps']:<4} mean={r['mean_burned_frac']:.3f} admissible={r['admissible']} "
              + " ".join(f"{x['burned_frac']:.2f}/{x['burning_at_T']}/{x['burning_at_60s']}" for x in r["per_scenario"]))
    print("CHOSEN:", None if best is None else {k: best[k] for k in ("p_ignite", "burn_steps", "mean_burned_frac")})
    if a.freeze and best is not None:
        json.dump({"T_s": a.T, "p_ignite": best["p_ignite"], "burn_steps": best["burn_steps"], "burn_time_dist": "fixed",
                   "neighborhood": "von_neumann", "wind_strength": 0.0, "dt_seconds": 1.0, "rule_version": RULE_VERSION, "rule": __doc__.split("PREDETERMINED RULE")[1].split("Held-out")[0].strip(),
                   "sweep": rows}, open(OUT, "w"), indent=1)
        print("froze", OUT)


if __name__ == "__main__":
    main()
