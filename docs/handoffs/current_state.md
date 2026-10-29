# Handoff: current state (reusable snapshot)

Snapshot taken 2026-10-29 at base commit `cbf0195` (branch `main`). It is a snapshot, not a source of truth: the repo wins.
Before acting, run `git status`, `git log --oneline -10`, and compare with `docs/project-status.md` and `models/MODELS.md`.
This file holds no credentials. It names where credentials live, never their values.

Legend: **VERIFIED** = recorded in a committed artifact or re-checked this session; **UNKNOWN** = not recorded anywhere; do not guess.

## 1. Project in one paragraph

ECE 496 wildfire monitoring. Question: does the training domain (synthetic vs real) change smoke-detection quality, and does it
propagate to UAV monitoring in ArduPilot SITL? Level 1 = Model A (synthetic) vs Model B (real Boreal). Level 2 = 2x2 cross-domain
matrix through one harness (`eval/`). Level 3 = closed loop; **not started** (gated on Level 2 review). Single class `0: smoke`.
Rules: `CLAUDE.md`. Auditor role: Codex (`AGENTS.md`, read-only).

## 2. Models (authoritative: `models/MODELS.md`)

Checkpoints (`*.pt`) are never in git. Copies: RunPod volume and `local_data/backups/` on the owner's laptop (neither visible to a cloud sandbox).

| Role | File | Seed | SHA-256 | Status |
|---|---|---|---|---|
| Model B | `real_boreal_best_v1.pt` | 0 | `1dc35ef293b6b35bb2dadaf5b6d5887ece89ca9d55a445bfdde258286c8355b6` | VERIFIED |
| Model B | `real_boreal_best_v1_seed1.pt` | 1 | `00c0e02fa2411993e160af4724ca9af6b3a17c2d25ac8e84761d4163175366c2` | VERIFIED |
| Model B | `real_boreal_best_v1_seed2.pt` | 2 | `fe53f08d414a8b09d799d4130a5cb1af62af82a5dc899339dd5a0571f0ab51d7` | VERIFIED |
| Model A | `synth_smoke_best_v1.pt` | 0 | `1da10c4814a905d92deec2d9c36185f872eabea447d47c389b9b065468009564` | VERIFIED |
| Model A | `synth_smoke_best_v1_seed1.pt` | 1 | `31c4528e37c54f779cdbfbf02d8e325877108696f3608391c0490b22ec34a454` | VERIFIED |
| Model A | `synth_smoke_best_v1_seed2.pt` | 2 | `3d5a31689ab212434b206949090c948a4de0b4b653abf8d0099619529146556a` | VERIFIED |

Model B seed 0 recipe (VERIFIED, `models/MODELS.md`, `eval/runpod/boreal_stage2_train_eval.sh`): YOLOv8s from COCO `yolov8s.pt`,
100 epochs, `patience=100`, imgsz 640, batch 32, seed 0, `deterministic=True`, workers 8, `optimizer=auto` (Ultralytics resolved it to
AdamW), other args Ultralytics 8.4.173 defaults. Env: ultralytics 8.4.173, torch 2.8.0+cu128, CUDA 12.8, cuDNN 9.10.02,
Python 3.12.3, driver 570.195.03, 1x RTX 4090 (secure, US-IL-1). Trained 2026-10-04, wall 2268 s, best epoch 37 by val fitness.
Seeds 1-2 set AdamW explicitly (`lr0=0.002 momentum=0.9 warmup_bias_lr=0.0`), wall 2180 s each, best epochs 59 and 91.

Not registry entries, never cite as Model A or B: `models/real_boreal_bestv1.pt` (git history, 5,442,970 B, sha `05ef7d46...47a54`,
commit `73dab87`) and the 0-byte `models/synthetic_best.pt`. UNKNOWN: what the 5.4 MB file was (epochs, data, architecture);
the "3-epoch demo checkpoint" in older docs is unconfirmed.

## 3. Level 2 results (3 seeds each; `docs/experiments/level2_v2.md`)

| train -> test | mAP50 mean +- SD | per seed |
|---|---|---|
| Synth -> Synth | 0.995 +- 0.000 | 0.995 x3 (saturated, cannot discriminate) |
| Synth -> Real | 0.375 +- 0.059 | 0.429, 0.384, 0.313 |
| Real -> Synth | 0.759 +- 0.107 | 0.753, 0.656, 0.869 |
| Real -> Real | 0.954 +- 0.003 | 0.951, 0.956, 0.953 |

Harness reproduces Model B's Ultralytics test mAP50 0.9442 as 0.9511 (VERIFIED, within tolerance, not identical).
A fresh offline recompute reproduces all 1,602 committed metric values exactly (VERIFIED, `eval/`, `docs/experiments/level2_v2/cells/`).

## 4. Known methodological findings (Codex REVISE, accepted 2026-10-29; `docs/experiments/level2_v2_audit.md`)

- **Boreal split is NOT video-disjoint** (VERIFIED by pod scan, `docs/experiments/level2_v2_audit/`): after dropping the site prefix,
  3 clips are shared train/test and 2 train/val, all through 197 background-only `Empty-Images` train frames. 0 exact (MD5) duplicates;
  no near-duplicate pair within a shared clip (64-bit dHash, Hamming <= 4). Identity rests on filenames and frame adjacency, not pixels.
  Test-subset sensitivity shows no consistent inflation of Model B (+0.034 / -0.026 / -0.015 mAP50 by seed).
  UNKNOWN: whether the clips are truly the same recordings at pixel level; the original dataset's authors' split rationale.
- Boreal test has no usable negatives (1 unlabeled frame). Real-domain specificity is unmeasured. The train-split `Empty-Images`
  diagnostic is valid for Model A only (Model B trained on it).
- "Optimizer steps" = minibatches: 109 x 100 = 10,900. Derived optimizer updates: 5,532 per run (gradient accumulation 2, ramped in
  warmup; derived from trainer source, not logged). `synth_smoke_v1_spec.md` line 25 still says 10,900 (not fixed; owner has uncommitted edits there).
- Synthetic test scenes share 6 texture slots and 16 puff sprites, so they are not independent. Texture-cluster bootstrap widens
  Real->Synth intervals 2-5x. "Seed variability dominates" was withdrawn.
- Edge-touch mismatch synthetic ~0.76 vs Boreal ~0.30: a stated limitation of Model A (gate cleared for the baseline only).
- Pod inference provenance: cells record `git.commit: null`; the exact `predict()` source for saved predictions is UNKNOWN.
- Training and pod inference are not shown bit-reproducible. Only the offline recompute from saved predictions is.
- `tilt_deg` / `fov_deg` conventions in `sim/camera.py` are unconfirmed (UNKNOWN) before Level 3 use.

## 5. Storage locations

RunPod network volume `wildfire-datasets`, id `sej40ez3ia`, 50 GB STANDARD, data center US-IL-1 (VERIFIED, ledger). Mounted at `/workspace`.

| Path on volume | Content |
|---|---|
| `/workspace/boreal/boreal_yolo` | Boreal source (read-only by convention) |
| `/workspace/splits/boreal_v1` | frozen split: symlinks to the source, `manifest.json`, `data.yaml`, `caveats.json` |
| `/workspace/synth_smoke_v1` | frozen synthetic dataset (10,616 files, `data_rw.yaml` is the path-rewritten YAML) |
| `/workspace/runs/real_boreal`, `real_boreal_seed1`, `real_boreal_seed2` | Model B runs and checkpoints |
| `/workspace/runs/synth_smoke_a_v1`, `..._seed1`, `..._seed2` | Model A runs and checkpoints |
| `/workspace/runs/real_boreal_eval` | seed-0 Model B test eval outputs |
| `/workspace/runs/level2_v2` | harness cells and pinned config |
| `/workspace/incoming` | operator uploads (repo tarball, `synth_smoke_v1.SHA256SUMS`) |
| `/workspace/integrity/boreal_scan.json.gz` | 2026-10-29 split scan (sha256 `8a4db032e44c4040872d6b4863af9efefe7b2e5ac3b1f6e48b6f993a3e400e8e`) |
| `/workspace/repo_v2` | non-git copy of the repo used for v2 inference |

Owner laptop only (not reachable from a cloud sandbox): `local_data/backups/` (checkpoints with `SHA256SUMS`),
`local_data/datasets/synth_smoke_v1/`, `local_data/integrity/`.
In git: configs, per-run `train_config.json`, `args.yaml`, `results.csv`, harness cells, audit artifacts, scripts under `eval/runpod/`.

## 6. Spend (ledger: `docs/project-status.md`, "Autonomous RunPod budget ledger")

Billed October total at last read $9.08 (storage $2.92); conservative cumulative about $9.3 including an unposted $0.25 scan-pod allowance;
remaining autonomous allowance about $90.7. Billing lags by hours: re-read before launching. Storage accrues about $0.12/day. No pods running at snapshot (`list-pods` empty).

## 7. Uncommitted owner work (do not touch, do not stage)

` M .gitignore`, ` M docs/synthetic/boreal_visual_analysis.md`, ` M docs/synthetic/synth_smoke_v1_spec.md`, ` M synthgen/generate_dataset.py`, `?? AGENTS.md`.
These exist only in the owner's working tree; a cloud checkout will not have them.

## 8. Open owner decisions

1. Report Level 2 with the documented overlap and sensitivity, or build a clip-level Boreal v2 split (new versioned split; frozen `boreal_v1` is never edited) and retrain Model B.
2. Real-domain negatives and cell-level truth for Level 3 (`docs/experiments/level3_observation_contract.md`, proposal only).
3. Second Codex review of `docs/experiments/level2_v2_audit.md` and `docs/experiments/level2_v2_audit/` is pending.
4. No probability-map code (`mapping/probability_map.py`) until the contract's section 5 quantities are measured.
