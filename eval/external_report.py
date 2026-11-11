#!/usr/bin/env python3
"""Report for the external evaluation (eval/datasets_external.yaml): reads the saved predictions and ground truth of a run directory written by
eval/run_matrix.py and recomputes every number with `eval.metrics.evaluate` (the one metric code path). No model, no image.

  python3 -m eval.external_report --run-dir local_data/external_eval/runs/ext_v1_fireman_primary --dataset fireman_primary --out <dir>

Reading rules (fixed in eval/datasets_external.yaml, section analysis):
  * every number is DESCRIPTIVE unless the cluster gate is met: interval and replication reading need >= 20 independent clusters, an exact sign-flip test
    needs >= 8, below that no interval and no test are produced (the report states "descriptive only");
  * Model A = synthetic-trained (seeds 0-2), Model B = Boreal-v2-trained (seeds 0-2); a family value is the mean over its seeds, the seed SD is shown next to
    it and is never part of an interval; Model B minus Model A is the reported difference.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

try:
    from . import external_stats as S
    from . import metrics as M
except ImportError:                                   # python3 eval/external_report.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from eval import external_stats as S
    from eval import metrics as M

FAMILY = {"model_a": "A", "model_a_seed1": "A", "model_a_seed2": "A", "model_bv2": "B", "model_bv2_seed1": "B", "model_bv2_seed2": "B"}
SEED = {"model_a": 0, "model_a_seed1": 1, "model_a_seed2": 2, "model_bv2": 0, "model_bv2_seed1": 1, "model_bv2_seed2": 2}
KEYS = ["mAP50", "mAP50-95", "precision", "recall", "f1_max", "mean_conf_tp", "loc_mean_iou_tp", "detection_rate"]



def load_run(run_dir: str, dataset: str):
    """-> {model: preds}, gts for the dataset (both loaded the way run_matrix saved them)."""
    from . import run_matrix as RM
    preds = {}
    for m in FAMILY:
        p = os.path.join(run_dir, "predictions", f"{m}__{dataset}.json.gz")
        if os.path.exists(p):
            preds[m] = RM.load_preds(p)
    gts = RM.load_gts(os.path.join(run_dir, "gts", f"{dataset}.json.gz"))
    return preds, gts


def subset_metrics(preds: Dict[str, np.ndarray], gts: Dict[str, np.ndarray], names: List[str], conf: float = 0.25) -> Dict[str, Optional[float]]:
    """The headline numbers for one set of images, from the unchanged harness metrics."""
    r = M.evaluate({n: preds.get(n, np.zeros((0, 6))) for n in names}, {n: gts[n] for n in names}, conf_thresholds=[conf])
    a = r["at_conf"][f"{conf:g}"]
    mf = r.get("max_f1") or {}
    return {"n_images": r["n_images"], "n_gt": r["n_gt_boxes"], "mAP50": r["mAP50"], "mAP50-95": r["mAP50-95"], "precision": a["precision"], "recall": a["recall"],
            "f1_max": mf.get("f1"), "mean_conf_tp": a["mean_conf_tp"], "loc_mean_iou_tp": a["loc_mean_iou_tp"], "detection_rate": a["image_detection_rate_on_positive"]}


def family_table(per_model: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, dict]:
    """mean and SD (ddof 1) over seeds per family and metric; difference B - A of the means."""
    out = {}
    for fam in ("A", "B"):
        vals = [per_model[m] for m in per_model if FAMILY[m] == fam]
        out[fam] = {"n_seeds": len(vals)}
        for k in KEYS:
            x = [v[k] for v in vals if v.get(k) is not None]
            out[fam][k] = {"mean": float(np.mean(x)) if x else None, "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None, "per_seed": x}
    out["B_minus_A"] = {k: (out["B"][k]["mean"] - out["A"][k]["mean"]) if out["A"][k]["mean"] is not None and out["B"][k]["mean"] is not None else None for k in KEYS}
    return out


def gate(n_clusters: int, min_interval: int = 20, min_test: int = 8) -> str:
    return "interval_and_test" if n_clusters >= min_interval else "exact_test_only" if n_clusters >= min_test else "descriptive_only"


def cluster_analysis(preds: Dict[str, Dict[str, np.ndarray]], gts: Dict[str, np.ndarray], cluster_of: Dict[str, str], names: List[str],
                     n_boot: int = 10000, seed: int = 20261111) -> dict:
    """Per-cluster mAP50 per model -> paired cluster bootstrap of B minus A (seeds averaged inside each replicate), sign-flip test, leave-one-cluster-out.
    Only the part allowed by the cluster gate is computed."""
    sub_g = {n: gts[n] for n in names}
    clusters = sorted({cluster_of[n] for n in names})
    g = gate(len(clusters))
    res = {"n_clusters": len(clusters), "gate": g}
    if g == "descriptive_only":
        return res
    sc = {}
    for m, p in preds.items():
        s = S.cluster_scores(p, sub_g, {n: cluster_of[n] for n in names}, metric="mAP50")
        sc[m] = np.array([s.get(c, np.nan) for c in clusters])
    fam = {f: np.array([sc[m] for m in sorted(sc) if FAMILY[m] == f]) for f in ("A", "B")}
    res["clusters_with_smoke"] = int(np.isfinite(fam["A"]).any(axis=0).sum())
    diffs = S.paired_cluster_differences(fam["A"], fam["B"])
    res["paired_diff_mean"] = float(diffs.mean()) if len(diffs) else None
    res["sign_flip"] = S.sign_flip_test(diffs, seed=seed)
    res["leave_one_cluster_out"] = S.leave_one_cluster_out(diffs)
    if g == "interval_and_test":
        res["bootstrap"] = S.cluster_bootstrap(fam, diff=("A", "B"), n_boot=n_boot, seed=seed)
    return res


def md_family(title: str, ft: Dict[str, dict]) -> str:
    def c(x, nd=3):
        return "n/a" if x is None else f"{x:.{nd}f}"
    lines = [f"**{title}**", "", "| metric | Model A mean (SD) | Model B mean (SD) | B - A |", "|---|---|---|---|"]
    for k in KEYS:
        a, b = ft["A"][k], ft["B"][k]
        lines.append(f"| {k} | {c(a['mean'])} ({c(a['sd'])}) | {c(b['mean'])} ({c(b['sd'])}) | {c(ft['B_minus_A'][k])} |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="eval/datasets_external.yaml")
    ap.add_argument("--out", required=True, help="new directory for report.json and report.md")
    ap.add_argument("--clusters-file", help="FASDD cluster/platform json (default: the one pinned in the config)")
    ap.add_argument("--xml-dir", default="local_data/external_eval/fireman/annotations")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args(argv)
    import yaml
    cfg = yaml.safe_load(open(a.config))
    preds, gts = load_run(a.run_dir, a.dataset)
    if len(preds) != 6:
        raise SystemExit(f"expected 6 models, found {sorted(preds)}")
    names = sorted(gts)
    os.makedirs(a.out, exist_ok=False)
    report = {"dataset": a.dataset, "n_images": len(names), "run_dir": a.run_dir, "models": sorted(preds)}
    md = [f"# External evaluation report: {a.dataset}", "", f"{len(names)} images; six frozen checkpoints; operating point conf 0.25, IoU 0.5; Model A = synthetic-trained, Model B = Boreal-v2-trained.", ""]
    per_model = {m: subset_metrics(preds[m], gts, names) for m in preds}
    report["per_model"] = per_model
    report["overall"] = family_table(per_model)
    md += [md_family("Overall (pooled images)", report["overall"]), ""]
    groups: Dict[str, Dict[str, List[str]]] = {}
    if a.dataset.startswith("fireman"):
        groups["video"] = {}
        for n in names:
            groups["video"].setdefault(n.split("__frame_")[0].replace("__", "/"), []).append(n)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from eval import fireman_prep as P
        near, static = [], {}
        kd_cache = {}
        for n in names:
            vid, fid = n.split("__frame_")[0].replace("__", "/"), int(os.path.splitext(n)[0].rsplit("_", 1)[1])
            if vid not in kd_cache:
                s, v = vid.split("/")
                kd_cache[vid] = P.keyframe_distances(P.parse_cvat(os.path.join(a.xml_dir, f"{s}__{v}__annotations.xml")))
            d = kd_cache[vid][fid]
            if d is not None and d <= 2:
                near.append(n)
        groups["sensitivity"] = {"frames_within_2_of_a_keyframe": near}
        report["keyframe_near_n"] = len(near)
    else:
        cf = a.clusters_file or cfg["datasets"][a.dataset]["clusters"]
        cl = json.load(open(cf))["images"]
        groups["platform_by_size"] = {}
        for n in names:
            groups["platform_by_size"].setdefault(cl[n]["platform_by_size"], []).append(n)
    report["groups"] = {}
    for gname, gd in groups.items():
        report["groups"][gname] = {}
        md += [f"## By {gname}", ""]
        for key, ns in sorted(gd.items()):
            if not ns:
                continue
            pm = {m: subset_metrics(preds[m], gts, ns) for m in preds}
            ft = family_table(pm)
            report["groups"][gname][key] = {"n_images": len(ns), "family": ft}
            md += [md_family(f"{key} ({len(ns)} images)", ft), ""]
    if not a.dataset.startswith("fireman"):
        cl = json.load(open(a.clusters_file or cfg["datasets"][a.dataset]["clusters"]))["images"]
        report["cluster_analysis"] = {}
        md += ["## Cluster-level uncertainty (dHash64 scene clusters; paired bootstrap of Model B minus Model A on per-cluster mAP50)", ""]
        subsets = {"all": names, **{k: v for k, v in groups["platform_by_size"].items()}}
        for rad in ("cluster_r6", "cluster_r4", "cluster_r2"):
            for sname, ns in subsets.items():
                co = {n: f"{sname}:{cl[n][rad]}" for n in ns}
                res = cluster_analysis(preds, gts, co, ns, n_boot=a.n_boot)
                key = f"{rad}/{sname}"
                report["cluster_analysis"][key] = res
                if rad == "cluster_r6" or sname == "all":
                    line = f"- {key}: {res['n_clusters']} clusters, gate {res['gate']}"
                    if "bootstrap" in res:
                        d = res["bootstrap"]["difference"]
                        line += f"; B - A = {d['point']:.3f}, 95% interval [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}], P(diff>0) {d['p_greater_than_zero']:.3f}; sign-flip p {res['sign_flip']['p']:.4f}; leave-one-cluster-out [{res['leave_one_cluster_out']['min']:.3f}, {res['leave_one_cluster_out']['max']:.3f}]"
                    md.append(line)
        md.append("")
    else:
        md += ["FireMan primary is one burn event: descriptive only (no interval, no test, no replication claim).", ""]
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1, default=float)
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
