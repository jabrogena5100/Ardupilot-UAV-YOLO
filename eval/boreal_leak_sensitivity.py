"""Sensitivity of the Boreal-test cells to the train/test source-clip overlap found by eval.boreal_split_audit.

Splits boreal_test into videos that share a source clip with Boreal train (affected) and videos that do not, and
re-evaluates saved predictions on each subset with the unchanged eval.metrics.evaluate. It never edits the frozen test
set; the subsets are reporting views only. Usage:
  python3 -m eval.boreal_leak_sensitivity --preds DIR --gts DIR --affected VIDEO[,VIDEO...] [--out OUT.json]
"""
import argparse
import json
import os

import numpy as np

from eval import metrics as M
from eval.boreal_split_audit import parse
from eval.run_matrix import load_gts, load_preds

MODELS = ["model_a", "model_a_seed1", "model_a_seed2", "model_b", "model_b_seed1", "model_b_seed2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True); ap.add_argument("--gts", required=True)
    ap.add_argument("--affected", required=True, help="comma-separated normalised video ids")
    ap.add_argument("--out")
    a = ap.parse_args()
    aff = set(a.affected.lower().split(","))
    gts = load_gts(os.path.join(a.gts, "boreal_test.json.gz"))
    views = {"all": sorted(gts), "affected": [n for n in sorted(gts) if parse(n)[0] in aff],
             "unaffected": [n for n in sorted(gts) if parse(n)[0] not in aff]}
    res = {"affected_videos": sorted(aff), "n_images": {k: len(v) for k, v in views.items()}, "models": {}}
    for m in MODELS:
        preds = load_preds(os.path.join(a.preds, f"{m}__boreal_test.json.gz"))
        res["models"][m] = {}
        for k, names in views.items():
            r = M.evaluate({n: preds.get(n, np.zeros((0, 6))) for n in names}, {n: gts[n] for n in names})
            c = next(iter(r["at_conf"].values()))
            res["models"][m][k] = {"mAP50": r["mAP50"], "mAP50-95": r["mAP50-95"], "recall@0.25": c.get("recall"),
                                   "n_gt_boxes": r["n_gt_boxes"], "n_images": r["n_images"]}
    s = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
