"""Texture-bank-clustered bootstrap for the synthetic test cells.

All 1,051 `synth_smoke_v1` test scenes are drawn from a texture bank of 6 slots (and 16 puff sprites) shared across the split,
so scenes are not independent. This resamples whole texture-bank slots (6 clusters) with eval.metrics.bootstrap_ci(clusters=...).
It measures test-set sampling variability with the texture bank treated as the sampling unit; with 6 clusters the percentile
interval is crude. Usage:
  python3 -m eval.synth_cluster_bootstrap --preds DIR --gts DIR --slots synth_test_texture_slot.json [--n 1000] [--out OUT.json]
"""
import argparse
import json
import os

from eval import metrics as M
from eval.run_matrix import load_gts, load_preds

MODELS = ["model_a", "model_a_seed1", "model_a_seed2", "model_b", "model_b_seed1", "model_b_seed2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True); ap.add_argument("--gts", required=True); ap.add_argument("--slots", required=True)
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--out")
    a = ap.parse_args()
    slots = json.load(open(a.slots))
    res = {"n_resamples": a.n, "unit": "texture-bank slot (6 clusters)", "models": {}}
    gts = load_gts(os.path.join(a.gts, "synth_test_pos.json.gz"))
    cl = {n: f"slot{slots[n]}" for n in gts}
    res["images_per_cluster"] = {c: sum(1 for v in cl.values() if v == c) for c in sorted(set(cl.values()))}
    for m in MODELS:
        preds = load_preds(os.path.join(a.preds, f"{m}__synth_test_pos.json.gz"))
        img = M.bootstrap_ci(preds, gts, n=a.n)
        clu = M.bootstrap_ci(preds, gts, n=a.n, clusters=cl)
        res["models"][m] = {"point_mAP50": img["point"]["mAP50"], "ci_mAP50_images": img["ci"]["mAP50"],
                            "ci_mAP50_texture_clusters": clu["ci"]["mAP50"]}
    s = json.dumps(res, indent=1, default=float)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
