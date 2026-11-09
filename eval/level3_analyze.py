#!/usr/bin/env python3
"""
eval/level3_analyze.py — per-run metrics, paired statistics and figures for the Level 3 paired runs.

Input: result directories written by `eval/level3_pairs.py run` (each holds fog CSVs, agent_A.log, manifest.json, sitl_work/mav.tlog).
Arms are grouped into families: baseline, A (Model A seeds), B (Model B v2 seeds). Unit of analysis = SCENARIO: a family's value in a
scenario is the mean over its seeds, so scenarios (not runs) are the replicates and seeds are never pooled as if independent.

Per-run metrics (all measured from the run's recorded artifacts; fire truth is used OFFLINE, after the run):
  first_true_det_s     first detection scored TRUE at ingest by fog (baseline: no detector -> None); censored at T for the paired statistic
  ign_first_visit_s         earliest time any burning cell was first VISITED by the UAV minus that cell's ignition time (all arms incl. baseline); censored at T
  fire_cells_visited_frac   burning-or-burnt cells visited / cells that burned  (monitoring effectiveness, all arms)
  visited_frac         grid coverage at the end
  det_true / det_false     detections scored at ingest;  false_alarm_per_min
  inv_true_s / inv_false_s  1 Hz HUD seconds in INVESTIGATE mode on targets within Chebyshev 1 of a cell that burned at any time in the run
                       (true diversion) or not (FALSE-PRIORITY DIVERSION). Time-agnostic truth: a target near a cell that ignites later counts as true,
                       so inv_false_s is a lower bound on misdirection.
Statistics (per contrast, e.g. A-baseline, B-baseline, A-B): mean paired difference over scenarios, percentile bootstrap CI over
scenarios (resampling scenarios, 10,000 draws, fixed seed) and an exact two-sided sign-flip permutation p-value on the scenario
differences. With n scenarios the p-value cannot be smaller than 2/2^n. These describe scenario sampling only; the seed spread is reported separately.

    python3 eval/level3_analyze.py RUN_DIR [...] --out OUTDIR
"""
import argparse
import csv
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

T_END = 600.0
FAMILY = {"baseline": "baseline", "A0": "A", "A1": "A", "A2": "A", "B2s0": "B", "B2s1": "B", "B2s2": "B"}
METRICS = ["first_true_det_s", "ign_first_visit_s", "visited_frac", "det_false", "inv_false_s", "inv_true_s"]


def kv(path):
    d = {}
    for line in open(path):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            d[k] = v
    return d


def run_metrics(d):
    end = kv(glob.glob(os.path.join(d, "experiment_end_*.txt"))[0])
    tag = end["tag"]
    parts = tag.split("-")
    arm, scen = parts[1], int(parts[-1])
    fire = list(csv.DictReader(open(glob.glob(os.path.join(d, "fire_cells_*.csv"))[0])))
    dets = list(csv.DictReader(open(glob.glob(os.path.join(d, "detections_*.csv"))[0])))
    tp = sorted(float(r["t"]) for r in dets if r["correct"] == "1" and float(r["t"]) >= 0)
    fp = [float(r["t"]) for r in dets if r["correct"] == "0" and float(r["t"]) >= 0]
    burned = {(int(r["i"]), int(r["j"])) for r in fire}
    ign = [r for r in fire if float(r["ignite_t"]) <= 0.5]       # the scenario's ignition cell burns from t=0, so a visit is always during the fire
    ign_visit = float(ign[0]["first_visit_t"]) if ign and ign[0]["first_visit_t"] != "" else None
    inv_true = inv_false = 0
    targets = {}
    log = os.path.join(d, "agent_A.log")
    if os.path.exists(log):
        for line in open(log):
            m = re.search(r"mode=INVESTIGATE→(\d+),(\d+)", line)
            if not m:
                continue
            c = (int(m.group(1)), int(m.group(2)))
            near = any(max(abs(c[0] - b[0]), abs(c[1] - b[1])) <= 1 for b in burned)
            targets[c] = targets.get(c, 0) + 1
            if near:
                inv_true += 1
            else:
                inv_false += 1
    row = {
        "run": os.path.basename(d.rstrip("/")), "tag": tag, "arm": arm, "family": FAMILY[arm], "scenario": scen,
        "first_true_det_s": tp[0] if tp else None, "ign_first_visit_s": ign_visit,
        "fire_cells": len(fire), "fire_cells_visited": sum(1 for r in fire if r["visited"] == "1"),
        "visited_frac": float(end["visited_frac"]), "det_true": len(tp), "det_false": len(fp),
        "inv_true_s": inv_true, "inv_false_s": inv_false, "n_investigate_targets": len(targets),
    }
    row["fire_cells_visited_frac"] = row["fire_cells_visited"] / max(1, row["fire_cells"])
    return row


def censored(row, key):
    v = row.get(key)
    if key in ("first_true_det_s", "ign_first_visit_s") and v is None:
        return T_END
    return float(v)


def family_by_scenario(rows, family, key):
    """scenario -> mean over the family's seeds of the (censored) metric; also the number of seeds."""
    out = {}
    for r in rows:
        if r["family"] == family:
            out.setdefault(r["scenario"], []).append(censored(r, key))
    return {s: float(np.mean(v)) for s, v in out.items()}


def paired(x, y, n_boot=10000, seed=0):
    sc = sorted(set(x) & set(y))
    d = np.array([x[s] - y[s] for s in sc], dtype=float)
    n = len(d)
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    signs = np.array(list(itertools.product((-1, 1), repeat=n)), dtype=float) if n <= 16 else None
    if signs is not None:
        perm = (signs * d).mean(axis=1)
        p = float(np.mean(np.abs(perm) >= abs(d.mean()) - 1e-12))
    else:
        p = None
    return {"n_scenarios": n, "mean_diff": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_signflip": p, "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n_zero": int((d == 0).sum())}


def contrasts(rows):
    res = {}
    for key in METRICS:
        fam = {f: family_by_scenario(rows, f, key) for f in ("baseline", "A", "B")}
        nodet = key == "first_true_det_s"          # the baseline has no detector: a contrast against it would be meaningless
        res[key] = {"A-baseline": None if nodet else paired(fam["A"], fam["baseline"]),
                    "B-baseline": None if nodet else paired(fam["B"], fam["baseline"]),
                    "A-B": paired(fam["A"], fam["B"]),
                    "family_means": {f: (float(np.mean(list(v.values()))) if v else None) for f, v in fam.items()}}
    return res


def seed_table(rows):
    out = {}
    for key in METRICS:
        for arm in sorted({r["arm"] for r in rows}):
            v = [censored(r, key) for r in rows if r["arm"] == arm and (r.get(key) is not None or key in ("first_true_det_s", "ign_first_visit_s"))]
            if v:
                out.setdefault(key, {})[arm] = {"n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}
    return out


def figures(rows, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"baseline": "#6b6b6b", "A": "#1f77b4", "B": "#d95f02"}
    scs = sorted({r["scenario"] for r in rows})
    specs = [("ign_first_visit_s", "first visit to ignition cell (s, censored 600)"), ("visited_frac", "grid coverage at end"),
             ("inv_false_s", "false-priority diversion (s in INVESTIGATE)"), ("det_false", "false detections per run")]
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (key, lab) in zip(axs.ravel(), specs):
        for k, fam in enumerate(("baseline", "A", "B")):
            for r in rows:
                if r["family"] == fam:
                    ax.plot(scs.index(r["scenario"]) + (k - 1) * 0.2, censored(r, key), "o", color=col[fam], alpha=0.55, ms=5)
            fm = family_by_scenario(rows, fam, key)
            ax.plot([scs.index(s) + (k - 1) * 0.2 for s in sorted(fm)], [fm[s] for s in sorted(fm)], "-", color=col[fam], lw=1.2, label=fam)
        ax.set_xticks(range(len(scs)))
        ax.set_xticklabels([str(s) for s in scs], rotation=45, fontsize=7)
        ax.set_xlabel("scenario seed")
        ax.set_ylabel(lab, fontsize=8)
    axs[0, 0].legend(title="family (dots = runs, line = seed mean)", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_level3_scenarios.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dirs = sorted(d for p in a.runs for d in glob.glob(p) if glob.glob(os.path.join(d, "experiment_end_*.txt")))
    rows = [run_metrics(d) for d in dirs]
    with open(os.path.join(a.out, "per_run.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump({"n_runs": len(rows), "contrasts": contrasts(rows), "per_arm": seed_table(rows)},
              open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    figures(rows, a.out)
    print(f"{len(rows)} runs -> {a.out}")


if __name__ == "__main__":
    main()
