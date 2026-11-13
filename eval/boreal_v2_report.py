"""Boreal v2 Level 2 report from harness cells (offline; no GPU, no images).

    python3 -m eval.boreal_v2_report --cells DIR/level2_boreal_v2/cells --out report.md [--json report.json] \
        [--preds DIR/level2_boreal_v2/predictions --gts DIR/level2_boreal_v2/gts --exclude-frames recheck.json]

Tables: 2x2 (mean +- sample SD over seeds, per-seed values), clip-clustered bootstrap CIs (test-set sampling only, 8 clusters), negative-scene diagnostics,
edge-touch recall and localisation, and an optional sensitivity analysis that drops test frames with a <=4-bit near-duplicate in train/val.
Seed variability and bootstrap intervals are different quantities and are never combined. Boreal v1 numbers are not used here.
"""
import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

OP = "0.25"


def seed_of(model: str) -> int:
    m = re.search(r"seed(\d+)$", model)
    return int(m.group(1)) if m else 0


def load_cells(d: str) -> Dict[str, Dict[int, Dict]]:
    out: Dict[str, Dict[int, Dict]] = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        c = json.load(open(f))
        fam = "A" if c["train_domain"] == "synthetic" else "B"
        out.setdefault(f"{fam}|{c['dataset']}", {})[seed_of(c["model"])] = c
    return out


def get(c, key):
    m = c["metrics"]
    if key == "mAP50":
        return m["mAP50"]
    if key == "mAP50-95":
        return m["mAP50-95"]
    a = m["at_conf"][OP]
    return {"recall": a["recall"], "precision": a["precision"], "fp_per_image": a["fp_per_image"], "fnr": a["false_negative_rate"],
            "empty_img_fp_rate": a["image_fp_rate_on_empty"], "mean_conf_tp": a["mean_conf_tp"], "mean_iou_tp": a["loc_mean_iou_tp"],
            "recall_edge": m["recall_by_edge_touch"]["edge_touching"]["recall"],
            "recall_nonedge": m["recall_by_edge_touch"]["not_edge_touching"]["recall"]}[key]


def stats(v: List[Optional[float]]) -> Dict[str, Any]:
    v = [x for x in v if x is not None]
    if not v:
        return {"values": [], "mean": None, "sd": None}
    a = np.asarray(v, float)
    return {"values": [float(x) for x in a], "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None}


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def cell_table(data, rows, cols, metric):
    L = [f"| cell | " + " | ".join(f"seed {s}" for s in range(3)) + " | mean | SD |", "|---|---|---|---|---|---|"]
    for lab, key in rows:
        by = data.get(key, {})
        vals = {s: get(c, metric) for s, c in by.items()}
        st = stats([vals.get(s) for s in sorted(vals)])
        L.append(f"| {lab} | " + " | ".join(f3(vals.get(s)) for s in range(3)) + f" | {f3(st['mean'])} | {f3(st['sd'])} |")
    return L


def ci_rows(data, rows, kind="bootstrap_cluster"):
    L = ["| cell | seed | unit | mAP50 [95% CI] | mAP50-95 [95% CI] | recall@0.25 [95% CI] | FP/img@0.25 [95% CI] |", "|---|---|---|---|---|---|---|"]
    for lab, key in rows:
        for s, c in sorted(data.get(key, {}).items()):
            b = c["metrics"].get(kind)
            if not b:
                continue
            f = lambda k: "n/a" if not b["ci"].get(k) else f"{b['point'][k]:.3f} [{b['ci'][k][0]:.3f}, {b['ci'][k][1]:.3f}]"
            L.append(f"| {lab} | {s} | {b['unit']} | {f('mAP50')} | {f('mAP50-95')} | {f('recall')} | {f('fp_per_image')} |")
    return L


def sensitivity(cells_data, preds_dir, gts_dir, boreal_ds, b_prefix, exclude):
    """Recompute Boreal metrics with `exclude` frames dropped (offline, from saved predictions and ground truth)."""
    from . import metrics as M
    from .run_matrix import load_gts, load_preds
    gts = load_gts(os.path.join(gts_dir, boreal_ds + ".json.gz"))
    keep = {n: g for n, g in gts.items() if n not in set(exclude)}
    out = {}
    for fam, pre in (("A", "model_a"), ("B", b_prefix)):
        for s, c in sorted(cells_data.get(f"{fam}|{boreal_ds}", {}).items()):
            p = load_preds(os.path.join(preds_dir, c["cell"] + ".json.gz"))
            full = M.evaluate({n: p[n] for n in gts}, gts, conf_thresholds=[float(OP)])
            sub = M.evaluate({n: p[n] for n in keep}, keep, conf_thresholds=[float(OP)])
            out[f"{fam}|seed{s}"] = {"n_full": len(gts), "n_kept": len(keep), "mAP50_full": full["mAP50"], "mAP50_kept": sub["mAP50"],
                                      "mAP50-95_full": full["mAP50-95"], "mAP50-95_kept": sub["mAP50-95"]}
    return out


def build(cells_dir, boreal_ds="boreal_v2_test", pos_ds="boreal_v2_test_pos", neg_ds="boreal_v2_test_neg", preds=None, gts=None,
          exclude_json=None, b_prefix="model_bv2"):
    data = load_cells(cells_dir)
    rows22 = [("Synth -> Synth (A on synth_test_pos)", "A|synth_test_pos"), ("Synth -> Real (A on " + boreal_ds + ")", f"A|{boreal_ds}"),
              ("Real -> Synth (B-v2 on synth_test_pos)", "B|synth_test_pos"), ("Real -> Real (B-v2 on " + boreal_ds + ")", f"B|{boreal_ds}")]
    L = ["# Boreal v2 Level 2 results (offline report)", "",
         "SD = sample standard deviation over independently trained seeds (n = 3: itself very uncertain). Bootstrap intervals (clip-clustered for Boreal, "
         "image-level for synthetic) describe test-set sampling for one fixed checkpoint only and are not seed uncertainty. "
         "**Boreal v2 estimates are not directly comparable with Boreal v1 estimates: train and test populations differ.**", ""]
    res: Dict[str, Any] = {"tables": {}}
    for metric, title in (("mAP50", "mAP50"), ("mAP50-95", "mAP50-95"), ("recall", "recall @ conf 0.25, IoU 0.5"), ("precision", "precision @ 0.25"),
                          ("fp_per_image", "false positives per image @ 0.25")):
        L += [f"## {title}", ""] + cell_table(data, rows22, None, metric) + [""]
    L += ["## Clustered uncertainty (Boreal: source-clip bootstrap, 8 clusters; synthetic: image bootstrap)", ""] + ci_rows(data, [rows22[1], rows22[3]]) + [""]
    L += ["Synthetic cells (image-level; texture-slot clustering is a separate offline step):", ""] + ci_rows(data, [rows22[0], rows22[2]], "bootstrap") + [""]
    neg_rows = [("Model A on held-out real negatives (" + neg_ds + ", 65 frames, 2 clips)", f"A|{neg_ds}"), ("Model B-v2 on held-out real negatives", f"B|{neg_ds}"),
                ("Model A on synthetic negatives (synth_test_all, 1051 incl. 350 empty)", "A|synth_test_all"), ("Model B-v2 on synthetic test_all", "B|synth_test_all")]
    L += ["## Negative-scene diagnostics", "",
          "Real negatives: 65 Empty-Images frames from `heinola_dji_0048` and `karkkila_dji_0003` (neither family trained on these clips). `empty_img_fp_rate` = fraction of empty images with at least one detection >= 0.25.", ""]
    L += ["| cell | seed | empty-image FP rate | FP/img |", "|---|---|---|---|"]
    for lab, key in neg_rows:
        for s, c in sorted(data.get(key, {}).items()):
            L.append(f"| {lab} | {s} | {f3(get(c, 'empty_img_fp_rate'))} | {f3(get(c, 'fp_per_image'))} |")
    L += ["", "## Edge-touch recall and localisation (Boreal v2 test, @0.25)", "", "| cell | seed | recall edge-touching | recall not edge-touching | mean IoU (TP) | mean conf (TP) |", "|---|---|---|---|---|---|"]
    for lab, key in (rows22[1], rows22[3]):
        for s, c in sorted(data.get(key, {}).items()):
            L.append(f"| {lab} | {s} | {f3(get(c, 'recall_edge'))} | {f3(get(c, 'recall_nonedge'))} | {f3(get(c, 'mean_iou_tp'))} | {f3(get(c, 'mean_conf_tp'))} |")
    if preds and gts and exclude_json:
        ex = json.load(open(exclude_json))["test_frames_with_train_or_val_neighbour_le4"]
        sens = sensitivity(data, preds, gts, boreal_ds, b_prefix, ex)
        res["sensitivity_excluding_near_duplicate_test_frames"] = sens
        L += ["", f"## Sensitivity: drop the {len(ex)} test frames with a <=4-bit near-duplicate in train/val (recomputed offline)", "",
              "| model | mAP50 full -> kept | mAP50-95 full -> kept |", "|---|---|---|"]
        for k, v in sens.items():
            L.append(f"| {k} | {v['mAP50_full']:.3f} -> {v['mAP50_kept']:.3f} | {v['mAP50-95_full']:.3f} -> {v['mAP50-95_kept']:.3f} |")
    for key, by in data.items():
        res["tables"][key] = {str(s): {k: get(c, k) for k in ("mAP50", "mAP50-95", "recall", "precision", "fp_per_image")} for s, c in sorted(by.items())}
    return "\n".join(L) + "\n", res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    ap.add_argument("--boreal-ds", default="boreal_v2_test")
    ap.add_argument("--neg-ds", default="boreal_v2_test_neg")
    ap.add_argument("--b-prefix", default="model_bv2")
    ap.add_argument("--preds")
    ap.add_argument("--gts")
    ap.add_argument("--exclude-frames", help="recheck.json with test_frames_with_train_or_val_neighbour_le4")
    a = ap.parse_args(argv)
    md, res = build(a.cells, a.boreal_ds, neg_ds=a.neg_ds, preds=a.preds, gts=a.gts, exclude_json=a.exclude_frames, b_prefix=a.b_prefix)
    open(a.out, "w").write(md)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
