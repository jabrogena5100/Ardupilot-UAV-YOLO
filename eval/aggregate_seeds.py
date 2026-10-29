"""
eval/aggregate_seeds.py — per-seed values, mean, SD and in-domain vs cross-domain change for the 2x2.

    python3 -m eval.aggregate_seeds --cells DIR_WITH_cells --out report.md [--json report.json]

Input: the `cells/*.json` written by eval/run_matrix.py (model names `model_a`, `model_a_seed1`, ...; the family
and seed are read from each cell's recorded `model`). Seed-to-seed spread (this script) and image/video bootstrap
CIs (inside each cell) answer different questions and are reported in separate tables, never combined:
  * SD across seeds = variability from retraining with a different seed (n small; sample SD, ddof=1).
  * bootstrap CI    = variability from which test images were drawn, for one fixed checkpoint.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List

import numpy as np

METRICS = [("mAP50", lambda m: m["mAP50"]), ("mAP50-95", lambda m: m["mAP50-95"]),
           ("recall@0.25", lambda m: m["at_conf"]["0.25"]["recall"]),
           ("precision@0.25", lambda m: m["at_conf"]["0.25"]["precision"]),
           ("FP/img@0.25", lambda m: m["at_conf"]["0.25"]["fp_per_image"])]
IN_DOMAIN = {"A": "synth_test_pos", "B": "boreal_test"}
CROSS = {"A": "boreal_test", "B": "synth_test_pos"}
CELL_NAMES = {("A", "synth_test_pos"): "Synth -> Synth", ("A", "boreal_test"): "Synth -> Real",
              ("B", "synth_test_pos"): "Real -> Synth", ("B", "boreal_test"): "Real -> Real"}


def seed_of(model: str) -> int:
    m = re.search(r"seed(\d+)$", model)
    return int(m.group(1)) if m else 0


def stats(v: List[float]) -> Dict[str, Any]:
    a = np.asarray(v, float)
    return {"values": [float(x) for x in a], "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None,
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}


def load(cells_dir: str) -> Dict[Any, Dict[int, Dict]]:
    out: Dict[Any, Dict[int, Dict]] = {}
    for f in sorted(glob.glob(os.path.join(cells_dir, "*.json"))):
        c = json.load(open(f))
        fam = "A" if c["train_domain"] == "synthetic" else "B"
        out.setdefault((fam, c["dataset"]), {})[seed_of(c["model"])] = c
    return out


def aggregate(cells_dir: str) -> Dict[str, Any]:
    data = load(cells_dir)
    res: Dict[str, Any] = {"cells": {}, "deltas": {}, "checkpoints": {}}
    for (fam, ds), by_seed in sorted(data.items()):
        seeds = sorted(by_seed)
        cell = {"seeds": seeds, "metrics": {}}
        for name, fn in METRICS:
            vals = [fn(by_seed[s]["metrics"]) for s in seeds]
            if all(v is not None for v in vals):
                cell["metrics"][name] = stats(vals)
        cell["bootstrap"] = {str(s): {k: by_seed[s]["metrics"].get(k) for k in ("bootstrap", "bootstrap_cluster")} for s in seeds}
        res["cells"][f"{fam}|{ds}"] = cell
        for s in seeds:
            res["checkpoints"][f"{fam}|seed{s}"] = by_seed[s]["checkpoint_sha256"]
    for fam in "AB":
        i, x = data.get((fam, IN_DOMAIN[fam]), {}), data.get((fam, CROSS[fam]), {})
        common = sorted(set(i) & set(x))
        if not common:
            continue
        d = {}
        for name, fn in METRICS:
            diffs = [fn(x[s]["metrics"]) - fn(i[s]["metrics"]) for s in common]
            d[name] = stats(diffs)
        res["deltas"][fam] = {"seeds": common, "in_domain": IN_DOMAIN[fam], "cross_domain": CROSS[fam], "metrics": d}
    return res


def fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def render(res: Dict[str, Any]) -> str:
    L = ["# Level 2: seed aggregation", "",
         "Per-seed values come from independently trained checkpoints (seed 0 = level2_v1, seeds 1-2 = level2_v2) evaluated "
         "through the same harness. **SD is the sample standard deviation across seeds (n = number of seeds; with n=3 it is "
         "itself very uncertain). Bootstrap intervals below are a different quantity and are not seed uncertainty.**", ""]
    for mname, _ in METRICS[:3] + METRICS[4:]:
        L += [f"## {mname}", "", "| cell | " + " | ".join(f"seed {s}" for s in range(3)) + " | mean | SD | min-max |", "|---|---|---|---|---|---|---|"]
        for (fam, ds), lab in CELL_NAMES.items():
            c = res["cells"].get(f"{fam}|{ds}")
            if not c or mname not in c["metrics"]:
                continue
            st = c["metrics"][mname]
            per = {s: v for s, v in zip(c["seeds"], st["values"])}
            L.append(f"| {lab} | " + " | ".join(fmt(per.get(s)) for s in range(3)) + f" | {fmt(st['mean'])} | {fmt(st['sd'])} | {fmt(st['min'])}-{fmt(st['max'])} |")
        L.append("")
    L += ["## Cross-domain change (cross-domain minus in-domain, same family and seed)", "",
          "| model family | metric | per-seed change | mean | SD |", "|---|---|---|---|---|"]
    for fam, d in sorted(res["deltas"].items()):
        for mname in ("mAP50", "mAP50-95", "recall@0.25", "FP/img@0.25"):
            if mname in d["metrics"]:
                st = d["metrics"][mname]
                L.append(f"| {'A (synthetic-trained)' if fam == 'A' else 'B (real-trained)'}: {d['in_domain']} -> {d['cross_domain']} | {mname} | "
                         + ", ".join(f"{v:+.3f}" for v in st["values"]) + f" | {st['mean']:+.3f} | {fmt(st['sd'])} |")
    L += ["", "## Bootstrap 95% CIs per seed (test-set sampling only, fixed checkpoint)", "",
          "Image-level resampling treats frames as independent. Boreal frames from one video are correlated, so the "
          "video-cluster row (5 test videos) is the more honest one for Boreal; the percentile method is crude with so few clusters.", "",
          "| cell | seed | resampling | mAP50 | mAP50-95 | recall@0.25 |", "|---|---|---|---|---|---|"]
    for (fam, ds), lab in CELL_NAMES.items():
        c = res["cells"].get(f"{fam}|{ds}")
        if not c:
            continue
        for s in c["seeds"]:
            for key, rl in (("bootstrap", "images"), ("bootstrap_cluster", "videos")):
                b = c["bootstrap"][str(s)].get(key)
                if b:
                    f = lambda k: "n/a" if not b["ci"][k] else f"[{b['ci'][k][0]:.3f}, {b['ci'][k][1]:.3f}]"
                    L.append(f"| {lab} | {s} | {rl} | {f('mAP50')} | {f('mAP50-95')} | {f('recall')} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = aggregate(a.cells)
    open(a.out, "w").write(render(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
