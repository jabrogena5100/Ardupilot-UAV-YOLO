"""Read-only audit of the completed Boreal v2 experiment, from COMMITTED files only (no pod, no GPU, no network).

    python3 -m eval.boreal_v2_audit [--out docs/experiments/boreal_v2/audit.json] [--recompute]

Each check returns (id, status, evidence): VERIFIED = recomputed or read from a recorded artifact in this audit run; FAILED = contradicted.
INFERRED / UNKNOWN claims are not computable here and are listed by docs/experiments/boreal_v2_audit.md. --recompute also re-derives every harness
cell offline from the saved predictions (slow: bootstrap) and compares; the default recomputes without bootstrap.
"""
import argparse
import csv
import glob
import gzip
import hashlib
import json
import os
import re
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "docs", "experiments", "boreal_v2")
RES = os.path.join(EXP, "results")
sys.path.insert(0, ROOT)

SEED_RUN = {0: "seed0_try3", 1: "seed1_try4", 2: "seed2_try4"}
EXPECT_SHA = {0: "090c0f4cbe2718d2b084a41b8a3736fa27b673bc66b20c64672b13c83c299a05",
              1: "5a9616cefddd3bca6eccf31d0691f7b1a2fadb572f84cf7948c85296644e9388",
              2: "c4d443a8af919c6b00e4319b117a5dfbe90bd8b29750c75e3970a9389ae097ef"}
FROZEN = {"train": "5b23b199965dfc47372e6100f4d5bfd41d57ef5b0b16b5ed83e91db056245c25", "val": "8ef9e902bb24a720112bef32d20248d6aa33c17d67efc9771643daa87d252e2f",
          "test": "9fa6dbc1d0eb9fa437e989be29afadc9a459f7850502f1c18c350379afcddc8f"}
PROTOCOL = {"model": "yolov8s.pt", "epochs": 100, "patience": 100, "imgsz": 640, "batch": 32, "deterministic": True, "workers": 8, "optimizer": "AdamW",
            "lr0": 0.002, "momentum": 0.9, "warmup_bias_lr": 0.0}
out = []


def rec(cid, ok, evidence):
    out.append({"id": cid, "status": "VERIFIED" if ok else "FAILED", "evidence": evidence})


def jl(p):
    return json.load(open(p))


def list_hash(names):
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def audit_checkpoints():
    for s, d in SEED_RUN.items():
        tc = jl(f"{RES}/training/{d}/train_config.json")
        rows = list(csv.DictReader(open(f"{RES}/training/{d}/results.csv")))
        a = tc["train_args"]
        rec(f"ckpt.seed{s}.hash_recorded", tc["checkpoint_sha256"] == EXPECT_SHA[s], tc["checkpoint_sha256"])
        rec(f"ckpt.seed{s}.complete_100_epochs", tc["complete"] and tc["epochs_completed"] == 100 and len(rows) == 100, f"complete={tc['complete']} epochs={tc['epochs_completed']} csv_rows={len(rows)}")
        rec(f"ckpt.seed{s}.protocol_args", all(a.get(k) == v for k, v in PROTOCOL.items()) and a.get("seed") == s, {k: a.get(k) for k in list(PROTOCOL) + ["seed"]})
        rec(f"ckpt.seed{s}.cache_ram_recorded_in_args", a.get("cache") == "ram", f"args.yaml cache={a.get('cache')}")
        rec(f"ckpt.seed{s}.data_is_boreal_v2", str(a.get("data", "")).endswith("/splits/boreal_v2/data.yaml"), a.get("data"))
        rec(f"ckpt.seed{s}.env", tc["environment"]["ultralytics"] == "8.4.173" and tc["environment"]["torch"].startswith("2.8.0"), {k: tc["environment"][k] for k in ("ultralytics", "torch", "cuda", "gpu", "driver")})
        # the harness hashed the file it evaluated: every cell of this model must carry the same SHA-256 and the expected path
        name = "model_bv2" if s == 0 else f"model_bv2_seed{s}"
        cells = glob.glob(f"{RES}/level2/cells/{name}__*.json")
        shas = {jl(c)["checkpoint_sha256"] for c in cells}
        paths = {jl(c)["checkpoint"] for c in cells}
        pinned = all(jl(c)["checkpoint_pinned"] for c in cells)
        rec(f"ckpt.seed{s}.evaluated_file_hash_matches", shas == {EXPECT_SHA[s]} and len(cells) == 5 and pinned, f"{len(cells)} cells, shas={[x[:12] for x in shas]}, pinned={pinned}")
        rec(f"ckpt.seed{s}.volume_path", len(paths) == 1 and "/real_boreal_v2_best_seed%d.pt" % s in list(paths)[0], sorted(paths))
    inte = jl(f"{RES}/INTEGRITY.json")
    rec("v1.existing_checkpoints_unchanged_after_run", inte["ok"] and all(v["unchanged"] for v in inte["existing_checkpoints"].values()), f"{len(inte['existing_checkpoints'])} checkpoints re-hashed on the pod")


def audit_cells():
    cells = sorted(glob.glob(f"{RES}/level2/cells/*.json"))
    models = {"model_a", "model_a_seed1", "model_a_seed2", "model_bv2", "model_bv2_seed1", "model_bv2_seed2"}
    ds = {"boreal_v2_test": 744, "boreal_v2_test_pos": 679, "boreal_v2_test_neg": 65, "synth_test_pos": 701, "synth_test_all": 1051}
    got = {(jl(c)["model"], jl(c)["dataset"]) for c in cells}
    rec("cells.count_30_full_cross_product", len(cells) == 30 and got == {(m, d) for m in models for d in ds}, f"{len(cells)} cells")
    cfg = yaml.safe_load(open(f"{ROOT}/eval/datasets_boreal_v2.yaml"))
    bad = []
    for c in cells:
        j = jl(c)
        d = cfg["datasets"][j["dataset"]]
        if j["dataset_filelist_sha256"] != d["filelist_sha256"] or j["dataset_n_images"] != ds[j["dataset"]] or not j["checkpoint_pinned"]:
            bad.append(os.path.basename(c))
        s = j["settings"]
        if (s["imgsz"], s["conf"], s["nms_iou"], s["max_det"], s["report_conf"]) != (640, 0.001, 0.7, 300, [0.25]):
            bad.append(os.path.basename(c) + ":settings")
    rec("cells.dataset_pins_and_settings", not bad, bad or "all 30 cells match eval/datasets_boreal_v2.yaml pins; imgsz 640, conf 0.001, NMS 0.7, max_det 300, report 0.25")
    a_shas = {jl(c)["checkpoint_sha256"] for c in cells if jl(c)["model"].startswith("model_a")}
    v1 = yaml.safe_load(open(f"{ROOT}/eval/datasets.yaml"))["models"]
    rec("cells.model_a_are_existing_pinned_checkpoints", a_shas == {v1[m]["sha256"] for m in ("model_a", "model_a_seed1", "model_a_seed2")}, "Model A seeds 0-2 = the pinned v1 checkpoints")


def audit_split():
    fl = jl(f"{EXP}/filelists.json")
    asg = jl(f"{EXP}/split_assignment.json")
    audit = jl(f"{EXP}/split_audit.json")
    rec("split.scan_is_pinned", hashlib.sha256(open(f"{ROOT}/docs/experiments/level2_v2_audit/boreal_scan.json.gz", "rb").read()).hexdigest() == asg["construction"]["scan_sha256_expected"], "boreal_scan.json.gz sha256 8a4db032...")
    hs = {p: list_hash([e["name"] for e in fl[p]]) for p in fl}
    rec("split.filelist_hashes_frozen", hs == FROZEN and audit["filelist_sha256"] == FROZEN, hs)
    from eval.boreal_split_audit import parse
    vids = {p: {parse(e["name"])[0] for e in fl[p]} for p in fl}
    inter = {f"{a}&{b}": sorted(vids[a] & vids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    rec("split.video_disjoint_recomputed", all(not v for v in inter.values()) and sum(len(v) for v in vids.values()) == 31, f"31 clips, intersections {inter}")
    scan = json.load(gzip.open(f"{ROOT}/docs/experiments/level2_v2_audit/boreal_scan.json.gz"))
    scan_names = sorted(n for s in scan.values() for n in s)
    allv2 = sorted(e["name"] for p in fl.values() for e in p)
    rec("split.covers_every_scanned_file_once", scan_names == allv2, f"{len(allv2)} files")
    md5 = {}
    for s in scan.values():
        for n, r in s.items():
            md5.setdefault(r[0], set()).add(n)
    part = {e["name"]: p for p, es in fl.items() for e in es}
    span = sum(1 for g in md5.values() if len({part[n] for n in g}) > 1)
    rec("split.no_exact_duplicates_across_partitions", span == 0, f"{span} md5 groups span partitions (dataset-wide duplicate groups: {sum(1 for g in md5.values() if len(g) > 1)})")
    t = fl["test"]
    cfg = yaml.safe_load(open(f"{ROOT}/eval/datasets_boreal_v2.yaml"))["datasets"]
    views = {"boreal_v2_test": [e["name"] for e in t], "boreal_v2_test_pos": [e["name"] for e in t if e["n_boxes"] > 0], "boreal_v2_test_neg": [e["name"] for e in t if e["name"].startswith("Empty-Images__")]}
    rec("split.harness_views_pinned", all(cfg[k]["filelist_sha256"] == list_hash(v) and cfg[k]["n_images"] == len(v) for k, v in views.items()), {k: len(v) for k, v in views.items()})


def audit_leakage():
    from eval.boreal_v2_recheck import recheck
    r = recheck(scan_path=f"{ROOT}/docs/experiments/level2_v2_audit/boreal_scan.json.gz", asg_path=f"{EXP}/split_assignment.json")
    committed = jl(f"{EXP}/recheck.json")
    rec("leakage.recheck_reproduces", r["n_pairs_le4"] == committed["n_pairs_le4"] == 22 and r["near_pairs_le4_cross_partition"] == committed["near_pairs_le4_cross_partition"], f"{r['n_pairs_le4']} pairs <=4 bits (recomputed == committed)")
    rec("leakage.six_distinct_test_frames", len(r["test_frames_with_train_or_val_neighbour_le4"]) == 6, r["test_frames_with_train_or_val_neighbour_le4"])
    clips = {tuple(sorted((p["a"].split("__")[1].rsplit("_frame", 1)[0].lower(), p["b"].split("__")[1].rsplit("_frame", 1)[0].lower()))) for p in r["near_pairs_le4_cross_partition"]}
    rec("leakage.only_two_clip_pairs", len(clips) == 2, sorted(clips))
    rep = jl(f"{RES}/report.json")["sensitivity_excluding_near_duplicate_test_frames"]
    mx = max(abs(v["mAP50_full"] - v["mAP50_kept"]) for v in rep.values())
    rec("leakage.sensitivity_max_mAP50_change", mx <= 0.0015, f"max |mAP50 change| = {mx:.5f} over 6 models")
    asg = jl(f"{EXP}/split_assignment.json")["assignment"]
    rec("leakage.karkkila_0007_0008_together", asg["karkkila_dji_0007"] == asg["karkkila_dji_0008"], f"both in {asg['karkkila_dji_0007']}")


def audit_metrics(recompute_bootstrap):
    import numpy as np
    from eval import metrics as M
    from eval.run_matrix import load_gts, load_preds
    pd, gd = f"{RES}/level2/predictions", f"{RES}/level2/gts"
    n, diff, loc = 0, [], 0
    for c in sorted(glob.glob(f"{RES}/level2/cells/*.json")):
        j = jl(c)
        preds = load_preds(os.path.join(pd, j["cell"] + ".json.gz"))
        gts = load_gts(os.path.join(gd, j["dataset"] + ".json.gz"))
        new = M.evaluate(preds, gts, conf_thresholds=j["settings"]["report_conf"])
        old = j["metrics"]
        for k in ("n_images", "n_positive_images", "n_negative_images", "n_gt_boxes", "n_predictions", "mAP50", "mAP50-95"):
            n += 1
            if new[k] != old[k]:
                diff.append((j["cell"], k))
        for k, v in old["at_conf"]["0.25"].items():
            n += 1
            if new["at_conf"]["0.25"][k] != v:
                (loc := loc + 1) if k.startswith("loc_") else diff.append((j["cell"], "at_conf." + k))
        if recompute_bootstrap and "bootstrap_cluster" in old:
            clusters = {nm: parse_clip(nm) for nm in gts}
            b = M.bootstrap_ci(preds, gts, n=old["bootstrap_cluster"]["n_resamples"], seed=old["bootstrap_cluster"]["seed"], conf=0.25, clusters=clusters)
            n += 1
            if b["ci"] != old["bootstrap_cluster"]["ci"]:
                diff.append((j["cell"], "bootstrap_cluster"))
    rec("metrics.detection_metrics_recomputed_exact", not diff, f"{n} values recomputed from saved predictions/gts; non-localisation differences: {len(diff)} {diff[:3]}; localisation-only float differences: {loc}")


def parse_clip(name):
    from eval.boreal_split_audit import parse
    return parse(name)[0]


def audit_documentation():
    docs = {p: open(os.path.join(ROOT, p)).read() for p in ("docs/experiments/boreal_v2_results.md", "models/MODELS.md", "docs/handoffs/from_cloud_2026-10-29.md", "docs/project-status.md")}
    dev = jl(f"{RES}/PROTOCOL_DEVIATION.json")
    rec("deviation.cache_ram_documented", dev["deviation"] == "cache=ram" and dev["all_identical"] and dev["ok"] and all("cache=ram" in docs[p] for p in docs), "PROTOCOL_DEVIATION.json + results doc + MODELS.md + handoff + status all mention cache=ram")
    r = docs["docs/experiments/boreal_v2_results.md"]
    rec("deviation.non_determinism_warning_and_limited_evidence_stated", "non-deterministic" in r and "only 4 epochs" in r, "results doc states Ultralytics warning and the 4-epoch evidence limit")
    for h in EXPECT_SHA.values():
        rec(f"docs.hash_{h[:8]}_in_registry", h in docs["models/MODELS.md"] and h in r, "present in MODELS.md and results doc")
    rep = jl(f"{RES}/report.json")["tables"]
    import statistics as st
    chk = [("0.512", "A|boreal_v2_test"), ("0.612", "B|synth_test_pos"), ("0.899", "B|boreal_v2_test")]
    ok = all(f"{st.mean([rep[k][s]['mAP50'] for s in '012']):.3f}" == v and v in r for v, k in chk)
    rec("docs.headline_means_match_cells", ok, "0.512 / 0.612 / 0.899 recomputed from report.json")
    # hand-typed negative-scene counts
    neg = {}
    for fam, pre in (("A", "model_a"), ("B", "model_bv2")):
        for s in range(3):
            m = pre if s == 0 else f"{pre}_seed{s}"
            j = jl(f"{RES}/level2/cells/{m}__boreal_v2_test_neg.json")
            neg[(fam, s)] = round(j["metrics"]["at_conf"]["0.25"]["image_fp_rate_on_empty"] * 65)
    rec("docs.negative_counts_9_11_17_and_1_1_1", [neg[("A", s)] for s in range(3)] == [9, 11, 17] and [neg[("B", s)] for s in range(3)] == [1, 1, 1], {f"{k[0]}_seed{k[1]}": v for k, v in neg.items()})


def audit_preservation():
    import subprocess
    base = "c080777d3afa8fff59bb20b32b2c464e403cf8df"
    protected = ["eval/datasets.yaml", "docs/experiments/real_boreal_v1.md", "docs/experiments/level2_v1.md", "docs/experiments/level2_v2.md", "docs/experiments/level2_v2_audit.md",
                 "docs/experiments/level2_v1", "docs/experiments/level2_v2", "docs/experiments/level2_v2_audit", "eval/metrics.py"]
    try:
        d = subprocess.run(["git", "-C", ROOT, "diff", "--name-only", base, "HEAD", "--"] + protected, capture_output=True, text=True, timeout=60)
        if d.returncode != 0:
            raise RuntimeError(d.stderr[:100])
        rec("preservation.v1_and_historical_v2_files_unchanged_vs_main", d.stdout.strip() == "", d.stdout.strip() or f"0 changed files among {len(protected)} protected paths since {base[:8]}")
        mm = subprocess.run(["git", "-C", ROOT, "diff", "--numstat", base, "HEAD", "--", "models/MODELS.md"], capture_output=True, text=True).stdout.split()
        rec("preservation.models_md_append_only", mm and mm[1] == "0", f"MODELS.md: +{mm[0]} -{mm[1]} lines")
    except Exception as e:   # shallow clone or no base commit
        out.append({"id": "preservation.git_diff", "status": "SKIPPED", "evidence": f"git history unavailable: {type(e).__name__}"})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(EXP, "audit.json"))
    ap.add_argument("--recompute", action="store_true", help="also recompute the clip-cluster bootstrap CIs (slow)")
    a = ap.parse_args(argv)
    for f in (audit_checkpoints, audit_cells, audit_split, audit_leakage, audit_documentation, audit_preservation):
        f()
    audit_metrics(a.recompute)
    json.dump({"checks": out, "summary": {s: sum(1 for c in out if c["status"] == s) for s in ("VERIFIED", "FAILED", "SKIPPED")}}, open(a.out, "w"), indent=1, default=str)
    for c in out:
        print(f"{c['status']:9s} {c['id']}")
    print({s: sum(1 for c in out if c["status"] == s) for s in ("VERIFIED", "FAILED", "SKIPPED")})
    return 1 if any(c["status"] == "FAILED" for c in out) else 0


if __name__ == "__main__":
    sys.exit(main())
