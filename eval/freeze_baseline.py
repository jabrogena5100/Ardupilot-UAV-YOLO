#!/usr/bin/env python3
"""Freeze / verify the Level 1-3 verified baseline.

  python3 eval/freeze_baseline.py --write [--code-commit SHA]   writes docs/baseline/baseline_v1.json
  python3 eval/freeze_baseline.py --verify                       recomputes every hash and compares; exit 1 on any mismatch

Records: git commit, SHA-256 of scientific configuration files, checkpoints (cross-checked against models/MODELS.md),
dataset pins, result artifacts under docs/experiments, and the raw Level 3 per-run artifacts (gitignored, 2.8 GB; only the
files the analysis reads are hashed). Nothing here modifies a checkpoint, dataset or result.
"""
import argparse, glob, hashlib, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs/baseline/baseline_v1.json")

CONFIG_FILES = [
    "eval/datasets.yaml", "eval/datasets_boreal_v2.yaml", "eval/level3_scenarios.json", "eval/level3_fire.json",
    "eval/level3_thresholds.json", "eval/metrics.py", "eval/run_matrix.py", "eval/level3_analyze.py", "eval/level3_report_tables.py",
    "exp_swarm_fire.yaml", "exp_swarm_perception_single.yaml", "synthgen/configs/domain_rand.yaml",
    "docs/experiments/level2_v1/datasets_pinned.yaml", "docs/experiments/level2_v2/datasets_pinned_on_pod.yaml",
    "models/MODELS.md",
]
CHECKPOINTS = {  # name -> local backup path (hashes are not stored here; they are computed and cross-checked against MODELS.md)
    "A_seed0 synth_smoke_best_v1": "local_data/backups/synth_smoke_a_v1/synth_smoke_a_v1/synth_smoke_best_v1.pt",
    "A_seed1 synth_smoke_best_v1_seed1": "local_data/backups/level2_v2_seeds/synth_smoke_a_v1_seed1/synth_smoke_best_v1_seed1.pt",
    "A_seed2 synth_smoke_best_v1_seed2": "local_data/backups/level2_v2_seeds/synth_smoke_a_v1_seed2/synth_smoke_best_v1_seed2.pt",
    "B_v1_seed0 real_boreal_best_v1": "local_data/backups/real_boreal_v1/real_boreal_v1_backup/real_boreal_best_v1.pt",
    "B_v1_seed1 real_boreal_best_v1_seed1": "local_data/backups/level2_v2_seeds/real_boreal_seed1/real_boreal_best_v1_seed1.pt",
    "B_v1_seed2 real_boreal_best_v1_seed2": "local_data/backups/level2_v2_seeds/real_boreal_seed2/real_boreal_best_v1_seed2.pt",
    "B_v2_seed0 real_boreal_v2_best_seed0": "local_data/backups/boreal_v2/real_boreal_v2_best_seed0.pt",
    "B_v2_seed1 real_boreal_v2_best_seed1": "local_data/backups/boreal_v2/real_boreal_v2_best_seed1.pt",
    "B_v2_seed2 real_boreal_v2_best_seed2": "local_data/backups/boreal_v2/real_boreal_v2_best_seed2.pt",
}
RESULT_DIRS = ["docs/experiments/level2_v1", "docs/experiments/level2_v2", "docs/experiments/boreal_v2",
               "docs/experiments/level3_final"]
RESULT_FILES = ["docs/experiments/level2_v1.md", "docs/experiments/level2_v2.md", "docs/experiments/boreal_v2_results.md",
                "docs/experiments/level3_final.md", "docs/experiments/level3_methodology.md", "docs/datasets/synth_smoke_v1.md"]
DATASET_FILES = ["local_data/datasets/synth_smoke_v1/manifest.json", "local_data/datasets/synth_smoke_v1/data.yaml"]
RAW_L3 = "local_data/level3/final"
RAW_PATTERNS = ["manifest.json", "run_meta.json", "experiment_end_*.txt", "detections_*.csv", "fire_cells_*.csv"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, ROOT)


def tree(paths):
    return {rel(p): sha(p) for p in sorted(paths) if os.path.isfile(p)}


def digest(d):
    return hashlib.sha256("\n".join(f"{k} {v}" for k, v in sorted(d.items())).encode()).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def collect(code_commit):
    reg = open(os.path.join(ROOT, "models/MODELS.md")).read()
    ck = {}
    for name, p in CHECKPOINTS.items():
        full = os.path.join(ROOT, p)
        h = sha(full)
        ck[name] = {"path": p, "sha256": h, "bytes": os.path.getsize(full), "in_MODELS_md": h in reg}
    cfg = tree(os.path.join(ROOT, f) for f in CONFIG_FILES)
    res = tree([os.path.join(ROOT, f) for f in RESULT_FILES] +
               [os.path.join(dp, f) for d in RESULT_DIRS for dp, _, fs in os.walk(os.path.join(ROOT, d)) for f in fs])
    l3cfg = tree(glob.glob(os.path.join(ROOT, RAW_L3, "l3-*.yaml")))
    raw = tree(p for r in ("results", "results_repeat") for pat in RAW_PATTERNS
               for p in glob.glob(os.path.join(ROOT, RAW_L3, r, "*", pat)))
    ds = tree(os.path.join(ROOT, f) for f in DATASET_FILES)
    return {
        "frozen_on": "2026-11-09",
        "code_commit": code_commit,
        "note": "code_commit is the commit that contains the final analysis code and regenerated Level 3 outputs. This file is committed afterwards; the commit that adds it carries the git tag baseline-L1-3-v1.",
        "checkpoints": ck,
        "level3_checkpoint_families": {"A": "synth_smoke_best_v1{,_seed1,_seed2} (model_tag synth_smoke_v1_seed0..2)",
                                       "B": "real_boreal_v2_best_seed0..2 (model_tag real_boreal_v2_seed0..2), NOT the v1 Boreal checkpoints"},
        "config_files": cfg, "config_digest": digest(cfg),
        "level3_run_configs": {"n": len(l3cfg), "digest": digest(l3cfg), "files": l3cfg},
        "datasets": ds,
        "result_artifacts": res, "result_digest": digest(res),
        "level3_raw_artifacts": {"n": len(raw), "digest": digest(raw), "files": raw},
    }


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--verify", action="store_true")
    ap.add_argument("--code-commit", default=None)
    a = ap.parse_args()
    if a.write:
        cur = collect(a.code_commit or git("rev-parse", "HEAD"))
        bad = [k for k, v in cur["checkpoints"].items() if not v["in_MODELS_md"]]
        if bad:
            sys.exit(f"checkpoint hash not in models/MODELS.md: {bad}")
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(cur, open(OUT, "w"), indent=1, sort_keys=True)
        print("wrote", rel(OUT), "| checkpoints", len(cur["checkpoints"]), "| config", len(cur["config_files"]),
              "| results", len(cur["result_artifacts"]), "| raw L3", cur["level3_raw_artifacts"]["n"])
        return
    ref = json.load(open(OUT))
    cur = collect(ref["code_commit"])
    problems = []
    for sec in ("checkpoints", "config_files", "datasets", "result_artifacts"):
        for k in sorted(set(ref[sec]) | set(cur[sec])):
            r, c = ref[sec].get(k), cur[sec].get(k)
            r, c = (r["sha256"] if isinstance(r, dict) else r), (c["sha256"] if isinstance(c, dict) else c)
            if r != c:
                problems.append(f"{sec}: {k}: frozen={r} now={c}")
    for sec in ("level3_run_configs", "level3_raw_artifacts"):
        if ref[sec]["digest"] != cur[sec]["digest"]:
            problems.append(f"{sec}: digest differs")
    try:
        anc = subprocess.run(["git", "merge-base", "--is-ancestor", ref["code_commit"], "HEAD"], cwd=ROOT).returncode == 0
    except OSError:
        anc = False
    if not anc:
        problems.append("code_commit is not an ancestor of HEAD")
    print("\n".join(problems) or "baseline verified: all hashes match")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
