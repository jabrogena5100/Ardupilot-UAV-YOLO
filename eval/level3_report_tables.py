#!/usr/bin/env python3
"""
eval/level3_report_tables.py - report-ready tables and figure for the Level 3 final evaluation, computed only from recorded artifacts.

Inputs : docs/experiments/level3_final/per_run.csv (written by eval/level3_analyze.py), the run directories (fog CSVs, agent_A.log,
         run_meta.json), docs/experiments/boreal_v2/results/report.json and docs/experiments/level2_v2/seed_aggregate.json (Level 2).
Outputs: into --out (default docs/experiments/level3_final/):
  per_run_extra.csv        repeated-observation vs distinct-cell false detections, per run
  seed_summary.{csv,md}    arm rows and family rows: mean and SD over the 3 SEEDS (each seed value = mean over the 8 scenarios)
  cluster_stats.{json,md}  contrasts at the scenario level (descriptive) and at the ignition-CELL level (primary clusters), Holm-adjusted p
  censoring.{csv,md}       right-censoring (no observed first true detection / ignition-cell visit by 600 s) and coverage summaries
  linkage.{csv,md}         Level 2 (detector quality on the synthetic test set) next to Level 3 behaviour, per seed
  fig_trajectories.png     representative UAV cell paths, baseline vs A0 vs B2s0, scenarios 2001 and 2004
  provenance.{csv,md}      per-run harness era (git_dirty, parallel slot) and per-era summaries

Nothing here changes a run, a threshold or the frozen protocol. Bootstrap seed is fixed.

    python3 eval/level3_report_tables.py [--final local_data/level3/final] [--out docs/experiments/level3_final]
"""
import argparse
import collections
import csv
import glob
import itertools
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T_END = 600.0
ARMS = ["baseline", "A0", "A1", "A2", "B2s0", "B2s1", "B2s2"]
FAM = {"baseline": "baseline", "A0": "A", "A1": "A", "A2": "A", "B2s0": "B", "B2s1": "B", "B2s2": "B"}
CELL_OF = {2001: (3, 2), 2005: (3, 2), 2002: (2, 1), 2012: (2, 1), 2004: (1, 0), 2014: (1, 0), 2006: (0, 1), 2015: (0, 1)}
CENSORED = ("first_true_det_s", "ign_first_visit_s")
METRICS = ["first_true_det_s", "ign_first_visit_s", "visited_frac", "det_true", "det_false", "det_false_cells", "inv_true_s", "inv_false_s"]
LABEL = {"first_true_det_s": "first true detection (s)", "ign_first_visit_s": "first visit of ignition cell (s)",
         "visited_frac": "grid coverage at 600 s", "det_true": "true detection observations / run",
         "det_false": "false detection observations / run", "det_false_cells": "distinct cells with a false detection / run",
         "inv_true_s": "INVESTIGATE log lines on burned-adjacent targets (~s)", "inv_false_s": "INVESTIGATE log lines on false targets (~s)"}


def read_csv(p):
    return list(csv.DictReader(open(p)))


def num(v):
    return None if v in ("", None) else float(v)


def cval(r, k):
    """Value with right-censoring at T for the two time-to-event metrics (None -> 600). A LOWER BOUND on the true time, never an exact time."""
    v = num(r[k])
    if v is None and k in CENSORED:
        return T_END
    return v


def extra_rows(rows, results):
    out = []
    for r in rows:
        d = os.path.join(results, r["run"])
        dets = read_csv(glob.glob(os.path.join(d, "detections_*.csv"))[0]) if glob.glob(os.path.join(d, "detections_*.csv")) else []
        fp = [x for x in dets if x["correct"] == "0" and float(x["t"]) >= 0]
        tp = [x for x in dets if x["correct"] == "1" and float(x["t"]) >= 0]
        out.append({"run": r["run"], "det_false_obs": len(fp), "det_false_cells": len({(x["i"], x["j"]) for x in fp}),
                    "det_true_obs": len(tp), "det_true_cells": len({(x["i"], x["j"]) for x in tp})})
    return out


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, ps[i] * (len(ps) - rank)))
        adj[i] = run
    return adj


def flip_p(d):
    d = np.asarray(d, float)
    s = np.array(list(itertools.product((-1, 1), repeat=len(d))), float)
    return float(np.mean(np.abs((s * d).mean(1)) >= abs(d.mean()) - 1e-12))


def per_scenario(rows, fam, key):
    o = collections.defaultdict(list)
    for r in rows:
        if FAM[r["arm"]] == fam:
            o[int(r["scenario"])].append(cval(r, key))
    return {s: float(np.mean(v)) for s, v in o.items()}


def seed_summary(rows, extra, out):
    ex = {e["run"]: e for e in extra}
    for r in rows:
        r["det_false_cells"] = ex[r["run"]]["det_false_cells"]
    tab, csvrows = {}, []
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        d = {"arm": arm, "family": FAM[arm], "n_scenarios": len(rs)}
        for k in METRICS:
            v = [cval(r, k) for r in rs]
            d[k] = float(np.mean(v))
            if k in CENSORED:
                d[k + "_n_observed"] = sum(1 for r in rs if num(r[k]) is not None)
        tab[arm] = d
        csvrows.append(d)
    fam_rows = []
    for fam in ("A", "B"):
        arms = [a for a in ARMS if FAM[a] == fam]
        d = {"family": fam, "n_seeds": len(arms)}
        for k in METRICS:
            v = [tab[a][k] for a in arms]
            d[k + "_mean"], d[k + "_sd"] = float(np.mean(v)), float(np.std(v, ddof=1))
        fam_rows.append(d)
    with open(os.path.join(out, "seed_summary.csv"), "w", newline="") as fh:
        keys = list(csvrows[0]) + [k for k in csvrows[1] if k not in csvrows[0]]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(csvrows)
    L = ["# Seed-level summary (generated by `eval/level3_report_tables.py`)", "",
         "Each arm value is the mean over the 8 held-out scenarios (one run per scenario). Family rows give the mean and the sample SD (ddof = 1) **over the 3 seed values** (n = 3: the SD is itself very uncertain).",
         "Time-to-event metrics are right-censored at 600 s: the value shown is the mean with unobserved events set to 600 s, i.e. a **lower bound** on the true mean, not a measured mean. `obs` = runs (of 8) with an observed event.", "",
         "| arm | " + " | ".join(LABEL[k] for k in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for arm in ARMS:
        d = tab[arm]
        cells = []
        for k in METRICS:
            f = "%.3f" if k == "visited_frac" else "%.1f"
            s = f % d[k]
            if arm == "baseline" and k == "first_true_det_s":
                s = "n/a (no detector)"
            elif k in CENSORED:
                s += f" (obs {d[k + '_n_observed']}/8)" if k == "ign_first_visit_s" or d[k + "_n_observed"] < 8 else ""
            cells.append(s)
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L += ["", "**Family mean +- SD over 3 seeds**", "", "| family | " + " | ".join(LABEL[k] for k in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for d in fam_rows:
        cells = []
        for k in METRICS:
            f = "%.3f +- %.3f" if k == "visited_frac" else "%.1f +- %.1f"
            cells.append(f % (d[k + "_mean"], d[k + "_sd"]))
        L.append(f"| {d['family']} | " + " | ".join(cells) + " |")
    L += ["", "The baseline has one arm (no detector, no seeds). Its 'false' and detection columns are 0 by construction."]
    open(os.path.join(out, "seed_summary.md"), "w").write("\n".join(L) + "\n")
    return tab, fam_rows


def cluster_stats(rows, extra, out):
    ex = {e["run"]: e for e in extra}
    for r in rows:
        r["det_false_cells"] = ex[r["run"]]["det_false_cells"]
    rng = np.random.default_rng(0)
    res, plist = [], []
    for k in METRICS:
        for a, b in (("A", "B"), ("A", "baseline"), ("B", "baseline")):
            if k in ("first_true_det_s", "det_true", "det_false_cells", "det_false") and b == "baseline":
                continue            # the baseline has no detector; the contrast is undefined for detection metrics
            x, y = per_scenario(rows, a, k), per_scenario(rows, b, k)
            sd = {s: x[s] - y[s] for s in sorted(x)}
            cells = collections.defaultdict(list)
            for s, v in sd.items():
                cells[CELL_OF[s]].append(v)
            cd = {c: float(np.mean(v)) for c, v in sorted(cells.items())}
            cv = np.array(list(cd.values()))
            boots = cv[rng.integers(0, len(cv), size=(10000, len(cv)))].mean(1)
            p = flip_p(cv)
            res.append({"metric": k, "contrast": f"{a}-{b}", "scenario_diffs": {str(s): v for s, v in sd.items()},
                        "cell_diffs": {f"{c[0]},{c[1]}": v for c, v in cd.items()}, "mean_diff": float(cv.mean()),
                        "cell_cluster_boot_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                        "cells_pos": int((cv > 0).sum()), "cells_neg": int((cv < 0).sum()), "cells_zero": int((cv == 0).sum()),
                        "scenarios_pos": int(sum(v > 0 for v in sd.values())), "scenarios_neg": int(sum(v < 0 for v in sd.values())),
                        "p_signflip_cells_unadj": p,
                        "p_signflip_scenarios_unadj_for_reference_only": flip_p(list(sd.values()))})
            plist.append(p)
    for r, h in zip(res, holm(plist)):
        r["p_cells_holm"] = h
    json.dump({"n_tests": len(res), "cells": {f"{c[0]},{c[1]}": sorted(s for s in CELL_OF if CELL_OF[s] == c) for c in sorted(set(CELL_OF.values()))},
               "note": "primary clusters = 4 ignition cells (each used by 2 scenarios); with 4 clusters the smallest possible two-sided sign-flip p is 0.125, so no contrast can be significant at 0.05 before or after Holm; p values are shown only to make that explicit",
               "results": res}, open(os.path.join(out, "cluster_stats.json"), "w"), indent=1)
    L = ["# Contrasts with the four ignition cells as clusters (generated)", "",
         "Unit of analysis: the **ignition cell** (4 clusters, each flown in 2 scenarios with different fire seed and texture). For each contrast the scenario differences (family mean over seeds, per scenario) are averaged within a cell, giving 4 cell differences.",
         "`CI` is a percentile bootstrap that resamples the 4 cells (10,000 draws, seed 0); with 4 clusters it is a crude, unstable interval and is **descriptive only**. `p` is the exact sign-flip p over the 4 cell differences: its **minimum is 0.125**, so nothing can reach 0.05; Holm adjustment over all %d contrasts is shown for completeness. Time-to-event contrasts use values censored at 600 s (lower bounds), see `censoring.md`. **Direction counts (cells/scenarios with the same sign) are the supported evidence.**" % len(res), "",
         "| metric | contrast | mean cell diff | cell CI (descriptive) | cells + / - | scenarios + / - | p over cells | Holm p |", "|---|---|---|---|---|---|---|---|"]
    for r in res:
        f = "%.3f" if r["metric"] == "visited_frac" else "%.1f"
        L.append(f"| {LABEL[r['metric']]} | {r['contrast']} | {f % r['mean_diff']} | [{f % r['cell_cluster_boot_ci95'][0]}, {f % r['cell_cluster_boot_ci95'][1]}] | "
                 f"{r['cells_pos']}/{r['cells_neg']} | {r['scenarios_pos']}/{r['scenarios_neg']} | {r['p_signflip_cells_unadj']:.3f} | {r['p_cells_holm']:.3f} |")
    L += ["", "**Scenario-level differences (descriptive; scenarios sharing a cell are not independent)**", ""]
    scen = sorted(CELL_OF)
    L += ["| metric | contrast | " + " | ".join(str(s) for s in scen) + " |", "|---|---|" + "---|" * len(scen)]
    for r in res:
        f = "%.3f" if r["metric"] == "visited_frac" else "%.0f"
        L.append(f"| {LABEL[r['metric']]} | {r['contrast']} | " + " | ".join(f % r["scenario_diffs"][str(s)] for s in scen) + " |")
    # sensitivity: scenario 2005 is the only scenario whose runs span both harness eras (sequential vs 2-slot parallel); drop it and recount directions
    L += ["", "**Sensitivity: scenario 2005 excluded** (its B2s1/B2s2 runs used the parallel harness, its other five arms the sequential one). Direction counts over the remaining 7 scenarios (cell (3,2) then rests on scenario 2001 alone):", "",
          "| metric | contrast | scenarios + / - | cells + / - |", "|---|---|---|---|"]
    for r in res:
        sd = {int(k): v for k, v in r["scenario_diffs"].items() if int(k) != 2005}
        cc = collections.defaultdict(list)
        for k, v in sd.items():
            cc[CELL_OF[k]].append(v)
        cm = [float(np.mean(v)) for v in cc.values()]
        L.append(f"| {LABEL[r['metric']]} | {r['contrast']} | {sum(v > 0 for v in sd.values())}/{sum(v < 0 for v in sd.values())} | {sum(v > 0 for v in cm)}/{sum(v < 0 for v in cm)} |")
    L += ["", "Cells: " + "; ".join(f"({c}) = scenarios {v}" for c, v in json.load(open(os.path.join(out, 'cluster_stats.json')))["cells"].items())]
    open(os.path.join(out, "cluster_stats.md"), "w").write("\n".join(L) + "\n")


def censoring(rows, out):
    L = ["# Right-censoring and coverage (generated)", "",
         "A run is **censored** for a time-to-event metric when the event was not observed within the 600 s window. The true time is then only known to exceed 600 s; it is never an exact time. Means that substitute 600 s are lower bounds.", "",
         "| arm | runs | ignition cell visited by 600 s | no observed visit (censored) | observed-only median visit (s) | true detection observed | no true detection (censored) | coverage mean +- SD over scenarios |", "|---|---|---|---|---|---|---|---|"]
    csvrows = []
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        vis = [float(r["ign_first_visit_s"]) for r in rs if r["ign_first_visit_s"] != ""]
        det = [float(r["first_true_det_s"]) for r in rs if r["first_true_det_s"] != ""]
        cov = [float(r["visited_frac"]) for r in rs]
        c = {"arm": arm, "runs": len(rs), "ign_visit_observed": len(vis), "ign_visit_censored": len(rs) - len(vis),
             "ign_visit_observed_median_s": float(np.median(vis)) if vis else None, "true_det_observed": len(det),
             "true_det_censored": len(rs) - len(det) if arm != "baseline" else None, "coverage_mean": float(np.mean(cov)), "coverage_sd": float(np.std(cov, ddof=1))}
        csvrows.append(c)
        L.append(f"| {arm} | {len(rs)} | {len(vis)} | {len(rs) - len(vis)} | {('%.0f' % np.median(vis)) if vis else 'n/a'} | "
                 f"{len(det) if arm != 'baseline' else 'n/a (no detector)'} | {(len(rs) - len(det)) if arm != 'baseline' else 'n/a'} | {np.mean(cov):.3f} +- {np.std(cov, ddof=1):.3f} |")
    tot = sum(c["ign_visit_censored"] for c in csvrows)
    L += ["", f"Total runs with no observed ignition-cell visit by 600 s: **{tot} of {len(rows)}**. The observed-only median is conditional on a visit having occurred and is biased low for arms with many censored runs; do not read it as the arm's typical time.",
          "Censored runs by scenario (ignition-cell visit): " + ", ".join(f"{s}: " + str(sum(1 for r in rows if int(r['scenario']) == s and r['ign_first_visit_s'] == '')) for s in sorted(CELL_OF)) + ".",
          "Runs with no observed true detection (perception arms): " + (", ".join(r["run"] for r in rows if r["arm"] != "baseline" and r["first_true_det_s"] == "") or "none") + "."]
    L += ["", "Direction of the bias: substituting 600 s for an unobserved time understates that run's true time, and the arm with more censored runs (B for the ignition-cell visit) is understated more, so time contrasts between arms are lower bounds on the size of the difference, not estimates of it. No survival model was fitted (4 clusters, 8 scenarios)."]
    with open(os.path.join(out, "censoring.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csvrows[0]))
        w.writeheader()
        w.writerows(csvrows)
    open(os.path.join(out, "censoring.md"), "w").write("\n".join(L) + "\n")


def linkage(rows, tab, out, final):
    rep = json.load(open(os.path.join(ROOT, "docs/experiments/boreal_v2/results/report.json")))["tables"]
    arms = {"A": [("A0", "0"), ("A1", "1"), ("A2", "2")], "B": [("B2s0", "0"), ("B2s1", "1"), ("B2s2", "2")]}
    thr = json.load(open(os.path.join(ROOT, "eval/level3_thresholds.json")))["arms"]
    out_rows = []
    for fam, lst in arms.items():
        for arm, sd in lst:
            m = rep[f"{fam}|synth_test_pos"][sd]
            t = tab[arm]
            out_rows.append({"arm": arm, "level3_tau": thr[arm]["tau"], "L2_mAP50": m["mAP50"], "L2_recall_0.25": m["recall"], "L2_precision_0.25": m["precision"], "L2_fp_per_image_0.25": m["fp_per_image"],
                             "L3_first_true_det_lb_s": t["first_true_det_s"], "L3_ign_visit_lb_s": t["ign_first_visit_s"], "L3_ign_visit_censored": 8 - t["ign_first_visit_s_n_observed"],
                             "L3_det_false_obs": t["det_false"], "L3_det_false_cells": t["det_false_cells"], "L3_coverage": t["visited_frac"]})
    with open(os.path.join(out, "linkage.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    L = ["# Level 2 to Level 3 linkage (generated)", "",
         "Level 2 columns: each checkpoint on the **synthetic positive test set** (`synth_test_pos`, 701 frames), operating point conf 0.25, IoU 0.5, from `docs/experiments/boreal_v2/results/report.json` (Model B = the Boreal v2 clip-disjoint retrain used in Level 3, not the older v1 checkpoints). "
         "Level 3 columns: the same checkpoint flown in 8 scenarios at its own **per-arm threshold** `tau` (A: 0.30-0.45, B: 0.01-0.02), so Level 2 (0.25) and Level 3 operating points differ. Time columns are lower bounds (right-censored at 600 s).", "",
         "| arm | Level 3 tau | L2 mAP50 | L2 recall | L2 precision | L2 FP/img | L3 first true det (s, lb) | L3 ign-cell visit (s, lb) | ign-cell visit censored | L3 false det obs/run | L3 distinct false cells/run | L3 coverage |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out_rows:
        L.append(f"| {r['arm']} | {r['level3_tau']} | {r['L2_mAP50']:.3f} | {r['L2_recall_0.25']:.3f} | {r['L2_precision_0.25']:.3f} | {r['L2_fp_per_image_0.25']:.3f} | {r['L3_first_true_det_lb_s']:.0f} | "
                 f"{r['L3_ign_visit_lb_s']:.0f} | {r['L3_ign_visit_censored']}/8 | {r['L3_det_false_obs']:.1f} | {r['L3_det_false_cells']:.1f} | {r['L3_coverage']:.3f} |")
    L += ["", "Six checkpoints (3 per family) cannot support a regression of Level 3 on Level 2 quality; this table places the two sets of numbers side by side and makes no dose-response claim. Level 2 uses conf 0.25; Level 3 used per-arm thresholds, so the operating points differ."]
    open(os.path.join(out, "linkage.md"), "w").write("\n".join(L) + "\n")


def hud_path(run_dir):
    pts = []
    for line in open(os.path.join(run_dir, "agent_A.log")):
        m = re.search(r"mode=([A-Z]+)(?:→(\d+),(\d+))?.*? cell=(\d+),(\d+)", line)
        if m:
            pts.append((m.group(1), int(m.group(4)), int(m.group(5)), (int(m.group(2)), int(m.group(3))) if m.group(2) else None))
    return pts


def figure(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    scens, arms = (2001, 2004), ("baseline", "A0", "B2s0")
    fig, axs = plt.subplots(len(scens), len(arms), figsize=(11, 7.2))
    N = 10
    for ri, sc in enumerate(scens):
        for ci, arm in enumerate(arms):
            ax = axs[ri, ci]
            d = os.path.join(results, f"l3-{arm}-{sc}")
            fire = {(int(r["i"]), int(r["j"])) for r in read_csv(glob.glob(os.path.join(d, "fire_cells_*.csv"))[0])}
            cells = {(int(r["i"]), int(r["j"])): r for r in read_csv(glob.glob(os.path.join(d, "fog_cells_*.csv"))[0])}
            img = np.zeros((N, N))
            for (i, j), r in cells.items():
                img[i, j] = 0.35 if r["visited"] == "1" else 0.0
            for c in fire:
                img[c] = 0.7
            ax.imshow(img, cmap="Oranges", vmin=0, vmax=1, origin="upper", extent=(-0.5, N - 0.5, N - 0.5, -0.5))
            pts = hud_path(d)
            xs, ys = [p[2] for p in pts], [p[1] for p in pts]
            for k in range(1, len(pts)):
                col = "#d95f02" if pts[k][0] == "INVESTIGATE" else "#1f77b4"
                ax.plot(xs[k - 1:k + 1], ys[k - 1:k + 1], "-", color=col, lw=0.9, alpha=0.8)
            dets = read_csv(glob.glob(os.path.join(d, "detections_*.csv"))[0])
            cnt = collections.Counter((int(x["i"]), int(x["j"]), x["correct"]) for x in dets if float(x["t"]) >= 0)
            for (i, j, ok), n in cnt.items():
                ax.scatter([j], [i], s=12 + 6 * min(n, 30), facecolors="none", edgecolors="#1b9e77" if ok == "1" else "#cc0000", lw=1.1)
            ig = [r for r in read_csv(glob.glob(os.path.join(d, "fire_cells_*.csv"))[0]) if float(r["ignite_t"]) <= 0.5]
            if ig:
                ax.scatter([int(ig[0]["j"])], [int(ig[0]["i"])], marker="*", s=140, color="k", zorder=5)
            ax.set_xticks(range(N)); ax.set_yticks(range(N)); ax.tick_params(labelsize=6)
            ax.set_title(f"{arm}, scenario {sc}", fontsize=9)
    fig.legend(handles=[Line2D([], [], color="#1f77b4", label="patrol / approach path (cell resolution, ~1 Hz log)"), Line2D([], [], color="#d95f02", label="INVESTIGATE path"),
                        Line2D([], [], marker="o", mfc="none", mec="#1b9e77", ls="", label="true detections (size ~ count)"), Line2D([], [], marker="o", mfc="none", mec="#cc0000", ls="", label="false detections"),
                        Line2D([], [], marker="*", color="k", ls="", label="ignition cell"), Line2D([], [], marker="s", color="#fdd0a2", ls="", label="visited"), Line2D([], [], marker="s", color="#e6550d", ls="", label="burned by 600 s")],
               loc="lower center", ncol=4, fontsize=7)
    fig.suptitle("Illustrative flights (seed 0 of each family). Scenarios 2001 and 2004 were picked by the author after the aggregate analysis, one from each of two ignition cells; not a random or effect-based sample", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(os.path.join(out, "fig_trajectories.png"), dpi=150)
    plt.close(fig)


def provenance(final, out):
    rows = []
    for kind in ("results", "results_repeat"):
        for d in sorted(glob.glob(os.path.join(final, kind, "*"))):
            m = json.load(open(os.path.join(d, "run_meta.json")))
            rows.append({"set": kind, "run": os.path.basename(d), "start_utc": m["start_utc"], "git_commit": m["git_commit"][:7], "git_dirty": m["git_dirty"], "parallel_slot": m.get("slot"), "wall_s": m["wall_s"]})
    with open(os.path.join(out, "provenance.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    c = collections.Counter((r["set"], r["git_dirty"], r["parallel_slot"] is not None) for r in rows)
    L = ["# Run provenance by harness era (generated from run_meta.json)", "", "| set | git_dirty | parallel slot | runs |", "|---|---|---|---|"]
    for (s, dty, sl), n in sorted(c.items()):
        L.append(f"| {s} | {dty} | {sl} | {n} |")
    open(os.path.join(out, "provenance.md"), "w").write("\n".join(L) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=os.path.join(ROOT, "local_data/level3/final"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs/experiments/level3_final"))
    a = ap.parse_args()
    rows = read_csv(os.path.join(a.out, "per_run.csv"))
    results = os.path.join(a.final, "results")
    extra = extra_rows(rows, results)
    with open(os.path.join(a.out, "per_run_extra.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(extra[0]))
        w.writeheader()
        w.writerows(extra)
    tab, _ = seed_summary(rows, extra, a.out)
    cluster_stats(rows, extra, a.out)
    censoring(rows, a.out)
    linkage(rows, tab, a.out, a.final)
    provenance(a.final, a.out)
    figure(results, a.out)
    print("wrote tables to", a.out)


if __name__ == "__main__":
    main()
