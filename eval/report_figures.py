#!/usr/bin/env python3
"""Figures for docs/report/final_report.md, drawn only from committed result files (no model, no image).
  python3 eval/report_figures.py
Fig 1: 2x2 cross-domain matrix (Boreal v2 split), mAP50, per-seed points + mean. Fig 2: Model A vs B mAP50 across the evaluated datasets (pooled AP)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "report", "figures")
CA, CB = "#2b6cb0", "#c05621"      # A synthetic-trained, B real-trained (colour-blind-safe pair)


def jl(p):
    return json.load(open(os.path.join(ROOT, p)))


def fig1():
    cells = jl("docs/experiments/boreal_v2/results/level2/seed_aggregation.json")["cells"]
    spec = [("Synthetic test", "A|synth_test_pos", "B|synth_test_pos"), ("Boreal v2 test", "A|boreal_v2_test", "B|boreal_v2_test")]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, (lab, ka, kb) in enumerate(spec):
        for j, (k, col, name) in enumerate(((ka, CA, "Model A (trained synthetic)"), (kb, CB, "Model B (trained Boreal v2)"))):
            m = cells[k]["metrics"]["mAP50"]
            x = i + (j - 0.5) * 0.36
            ax.bar(x, m["mean"], 0.32, color=col, alpha=0.85, label=name if i == 0 else None)
            ax.scatter([x] * len(m["values"]), m["values"], color="black", s=12, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels([s[0] for s in spec]); ax.set_ylabel("mAP50 (pooled, conf floor 0.001)"); ax.set_ylim(0, 1.05)
    ax.set_title("Cross-domain matrix: bar = mean of 3 seeds, dots = seeds", fontsize=9); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_level2_matrix.png"), dpi=160); plt.close(fig)


def fig2():
    fm = jl("docs/external_eval/fireman_primary_v1_report.json"); fa = jl("docs/external_eval/fasdd_smoke_positive_v1_report.json")
    b2 = jl("docs/experiments/boreal_v2/results/level2/seed_aggregation.json")["cells"]
    g = fa["groups"]["platform_by_size"]
    rows = [("Boreal v2 test\n(8 clips)", b2["A|boreal_v2_test"]["metrics"]["mAP50"]["values"], b2["B|boreal_v2_test"]["metrics"]["mAP50"]["values"], False),
            ("FireMan primary\n(1 burn event)", fm["overall"]["A"]["mAP50"]["per_seed"], fm["overall"]["B"]["mAP50"]["per_seed"], True),
            ("FASDD pooled\n(mixed platforms)", fa["overall"]["A"]["mAP50"]["per_seed"], fa["overall"]["B"]["mAP50"]["per_seed"], True),
            ("FASDD 720p\n(descriptive only)", g["ground_or_low_video_720p"]["family"]["A"]["mAP50"]["per_seed"], g["ground_or_low_video_720p"]["family"]["B"]["mAP50"]["per_seed"], True),
            ("FASDD UAV nadir\n(306 images)", g["uav_nadir_1080p"]["family"]["A"]["mAP50"]["per_seed"], g["uav_nadir_1080p"]["family"]["B"]["mAP50"]["per_seed"], True),
            ("FASDD tower", g["tower_fixed_camera"]["family"]["A"]["mAP50"]["per_seed"], g["tower_fixed_camera"]["family"]["B"]["mAP50"]["per_seed"], True)]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (lab, a, b, ext) in enumerate(rows):
        for j, (v, col) in enumerate(((a, CA), (b, CB))):
            x = i + (j - 0.5) * 0.36
            ax.bar(x, sum(v) / len(v), 0.32, color=col, alpha=0.85, hatch="//" if "720p" in lab else None, label=("Model A (synthetic-trained)" if j == 0 else "Model B (Boreal v2-trained)") if i == 0 else None)
            ax.scatter([x] * len(v), v, color="black", s=10, zorder=3)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=8); ax.set_ylabel("pooled mAP50"); ax.set_ylim(0, 1.05)
    ax.set_title("The A-vs-B ordering changes across evaluated datasets (descriptive; dots = seeds; hatched = descriptive-only group; no inference drawn)", fontsize=8)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_ranking_across_datasets.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); fig1(); fig2(); print("ok")
