#!/usr/bin/env python3
"""Report for the external evaluation (eval/datasets_external.yaml): reads the saved predictions and ground truth of a run directory written by
eval/run_matrix.py and recomputes every number with `eval.metrics.evaluate` (the one metric code path). No model, no image.

  python3 -m eval.external_report --run-dir local_data/external_eval/runs/ext_v1_fireman_primary --dataset fireman_primary --out <dir>

Reading rules (fixed in eval/datasets_external.yaml, section analysis):
  * every number is DESCRIPTIVE. The cluster bootstrap and sign-flip test are CONDITIONAL SENSITIVITY ANALYSES: they assume the dHash groups behave as
    independent units, which is not demonstrated (similarity components are not videos, cameras or burn events). They are not population-level inference.
    They are produced only where the cluster-count gate (>= 20 intervals, >= 8 exact test) is met AND the config does not restrict the subset to
    descriptive reporting (analysis.gates.platform_groups; a subset that contains a restricted group is restricted too);
  * two different estimands are reported side by side and are never mixed: POOLED AP (all images ranked together; the headline, mean over seeds) and
    CLUSTER-MACRO AP (AP computed inside each cluster, then averaged over clusters). AP is nonlinear, so they differ in weighting and in the confidence
    ranking. The bootstrap interval and the sign-flip p belong to the cluster-macro estimand only; no interval is computed for pooled AP;
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
                     n_boot: int = 10000, seed: int = 20261111, allow_inference: bool = True) -> dict:
    """ESTIMAND: cluster-macro mAP50 (AP inside each cluster, averaged over clusters), not pooled AP. Per-cluster mAP50 per model -> point values per family
    (always, when there is at least one cluster with smoke) and, only where the gate AND allow_inference permit, a paired cluster bootstrap of B minus A (seeds
    averaged inside each replicate), a sign-flip test and leave-one-cluster-out. These are conditional on the grouping being independent units."""
    sub_g = {n: gts[n] for n in names}
    clusters = sorted({cluster_of[n] for n in names})
    g = gate(len(clusters))
    res = {"estimand": "cluster_macro_mAP50", "n_clusters": len(clusters), "gate": g, "inference_allowed_by_config": bool(allow_inference)}
    if not allow_inference:
        res["gate"] = "descriptive_only_by_config"
    sc = {}
    for m, p in preds.items():
        s = S.cluster_scores(p, sub_g, {n: cluster_of[n] for n in names}, metric="mAP50")
        sc[m] = np.array([s.get(c, np.nan) for c in clusters])
    fam = {f: np.array([sc[m] for m in sorted(sc) if FAMILY[m] == f]) for f in ("A", "B")}
    full = np.arange(len(clusters))
    res["cluster_macro_mAP50"] = {f: S._score(fam[f], full) for f in ("A", "B")}
    res["cluster_macro_mAP50"]["B_minus_A"] = res["cluster_macro_mAP50"]["B"] - res["cluster_macro_mAP50"]["A"]
    res["clusters_with_smoke"] = int(np.isfinite(fam["A"]).any(axis=0).sum())
    if g == "descriptive_only" or not allow_inference:
        return res
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
        groups["sensitivity"] = {"frames_within_2_of_an_inferred_annotation_keyframe": near}   # keyframes are inferred from coordinate slope changes, not read from explicit flags
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
        restricted = {k for k, v in ((cfg.get("analysis") or {}).get("gates") or {}).get("platform_groups", {}).items()
                      if str(v.get("inference", "allowed")).startswith("descriptive")}
        plat = groups["platform_by_size"]
        subsets = {"all": (names, not restricted),            # a subset containing a descriptive-only group is descriptive-only too
                   **{k: (v, k not in restricted) for k, v in plat.items()}}
        if restricted:
            subsets["all_excluding_" + "_".join(sorted(restricted))] = ([n for k, v in plat.items() if k not in restricted for n in v], True)
        md += ["## Two estimands (do not mix them)", "",
               "- **Pooled AP** (headline above): all images of a subset ranked together, mean over seeds. No interval is computed for it.",
               "- **Cluster-macro AP**: AP inside each dHash64 scene cluster, then the unweighted mean over clusters. Weighting and confidence ranking differ from pooled AP (AP is nonlinear), so its value is a different number, not a refinement of the pooled one.",
               "- The bootstrap interval and sign-flip p below belong to the cluster-macro estimand only, are **conditional sensitivity analyses** (they assume the dHash groups behave as independent units, which is not demonstrated: similarity components are not videos, cameras or burn events), and are **not population-level inference**. Subsets the config restricts to descriptive reporting get point values only.",
               "- Bootstrap fraction = share of resamples with B - A > 0; it is neither a posterior probability nor the sign-flip p-value.", "",
               "| subset (cluster radius) | n clusters | pooled mAP50 A / B / B-A | cluster-macro mAP50 A / B / B-A | conditional interval (cluster-macro, percentile) | bootstrap B-A>0 | sign-flip |", "|---|---|---|---|---|---|---|"]
        def pooled(ns):
            ft = family_table({m: subset_metrics(preds[m], gts, ns) for m in preds})
            return ft["A"]["mAP50"]["mean"], ft["B"]["mAP50"]["mean"], ft["B_minus_A"]["mAP50"]
        pooled_cache = {sname: pooled(ns) for sname, (ns, _) in subsets.items()}
        for rad in ("cluster_r6", "cluster_r4", "cluster_r2"):
            for sname, (ns, allow) in subsets.items():
                co = {n: f"{sname}:{cl[n][rad]}" for n in ns}
                res = cluster_analysis(preds, gts, co, ns, n_boot=a.n_boot, allow_inference=allow)
                res["pooled_mAP50"] = dict(zip(("A", "B", "B_minus_A"), pooled_cache[sname]))
                key = f"{rad}/{sname}"
                report["cluster_analysis"][key] = res
                pa, pb, pd = pooled_cache[sname]; cm = res["cluster_macro_mAP50"]
                row = f"| {sname} ({rad[-2:]}) | {res['n_clusters']} | {pa:.3f} / {pb:.3f} / {pd:+.3f} | {cm['A']:.3f} / {cm['B']:.3f} / {cm['B_minus_A']:+.3f} | "
                if "bootstrap" in res:
                    d = res["bootstrap"]["difference"]
                    row += f"[{d['ci'][0]:.3f}, {d['ci'][1]:.3f}] | {round(d['p_greater_than_zero'] * res['bootstrap']['n_boot'])}/{res['bootstrap']['n_boot']} | {S.format_p(res['sign_flip'], short=True)} |"
                else:
                    row += ("descriptive only (config restriction) | - | - |" if not allow else "gate not met | - | - |")
                md.append(row)
        md.append("")
    else:
        md += ["FireMan primary is one burn event: descriptive only (no interval, no test, no replication claim).", ""]
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1, default=float)
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
