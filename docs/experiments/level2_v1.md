# Experiment: level2_v1 (Model A baseline training and the first 2x2 matrix)

Date: 2026-10-27. Status: **complete**. Single run, single seed (0). Nothing here is a final result: the synthetic
dataset is `synth_smoke_v1` with a known edge-touch mismatch, and Model A is a *baseline*, not evidence that the synthetic
domain is optimal.

## Purpose

1. Train Model A (YOLOv8s on `synth_smoke_v1`) with Model B's recipe (differences pinned in
   `docs/synthetic/synth_smoke_v1_spec.md` section 5).
2. Run the first Level 2 matrix through the unified harness (`eval/run_matrix.py` -> `eval/metrics.py`), after first
   reproducing Model B's one-off Ultralytics test numbers through the same harness.

## Setup

| Item | Value |
|---|---|
| Pod | RunPod secure, 1x RTX 4090, US-IL-1, `wildfire-datasets` volume at `/workspace`, $0.89/hr, pod `h88h3d09647945` (terminated) |
| Image / software | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; ultralytics 8.4.173, torch 2.8.0+cu128, CUDA 12.8, cuDNN 9.10.02, Python 3.12.3 (identical to Model B); driver 595.91.07 (Model B: 570.195.03) |
| Repo revision | harness committed at `6c9a5de`; the pod ran `eval/runpod/level2_modelA_train_and_matrix.sh` as committed at `5ca3e74` (uploaded as `repo_eval.tar` from `git archive HEAD`) |
| Model A | `synth_smoke_best_v1.pt`, SHA-256 `1da10c4814a905d92deec2d9c36185f872eabea447d47c389b9b065468009564` |
| Model A recipe | YOLOv8s COCO-pretrained, 100 epochs, imgsz 640, batch 32, seed 0, deterministic, patience 100, AdamW lr0 0.002 momentum 0.9 warmup_bias_lr 0.0 (the optimizer Model B resolved to); all other arguments identical to Model B (`model_a_training/args.yaml`, `train_config_v1.json`) |
| Model A training data | `synth_smoke_v1` train 3,479 images (3,219 smoke + 260 background); checkpoint chosen by best fitness on the synthetic val split; Model A never saw Boreal train/val/test |
| Dataset integrity | tarball SHA-256 and all 10,616 file hashes verified on the pod before training and again after the run ("DATASET STILL INTACT"). A path-rewritten `data_rw.yaml` copy was used; hashed originals untouched |
| Eval settings | conf floor 0.001, NMS IoU 0.7, imgsz 640, max_det 300, batch 32, seed 0; operating point IoU 0.5 at conf >= 0.25 (`datasets_pinned.yaml`) |
| Test sets (pinned by count and file-list hash) | `boreal_test` 701 (700 labelled + 1 background); `synth_test_pos` 701; `synth_test_all` 1,051 (701 + 350 background) |
| Training wall time | 996 s (Model B: 2,268 s on the same GPU type; not investigated) |

Also run: `attempt1_failed_pip` — the first launch aborted before any training because the image enforces PEP 668
(`pip install` refused); dataset and Model B pins had already passed. The script was fixed (`--break-system-packages`,
abort on failed import) and rerun from a clean state. No GPU time was spent on the failed attempt beyond the idle pod.

## Results (measured)

`matrix.md` / `matrix.csv` / `cells/*.json` in this directory; every number comes from the same code path.

| train -> test | images | mAP50 | mAP50-95 | P@0.25 | R@0.25 | FNR@0.25 | FP/img@0.25 | mean conf (TP) | mean IoU (TP) |
|---|---|---|---|---|---|---|---|---|---|
| Synth -> Synth (`test_pos`) | 701 | 0.995 | 0.985 | 1.000 | 1.000 | 0.000 | 0.000 | 0.968 | 0.975 |
| Synth -> Real (`boreal_test`) | 701 | 0.429 | 0.216 | 0.579 | 0.572 | 0.428 | 0.424 | 0.818 | 0.761 |
| Real -> Synth (`test_pos`) | 701 | 0.753 | 0.389 | 0.900 | 0.568 | 0.432 | 0.063 | 0.695 | 0.767 |
| Real -> Real (`boreal_test`) | 701 | 0.951 | 0.673 | 0.888 | 0.945 | 0.055 | 0.121 | 0.803 | 0.843 |

Empty-scene false positives (`synth_test_all`, 350 background images): Model A 0 of 350 images have a detection >= 0.25;
Model B 1.1%. The Boreal test set has a single background image (both models fire on it), so no Boreal empty-scene rate
is meaningful.

Pre-declared edge-touch diagnostic (spec section 4): recall at conf 0.25, GT boxes touching the image border vs not.

| cell | edge-touching (n) | not edge-touching (n) |
|---|---|---|
| Model A on `boreal_test` | 0.873 (142) | 0.497 (571) |
| Model B on `boreal_test` | 0.993 (142) | 0.933 (571) |
| Model B on `synth_test_pos` | 0.594 (542) | 0.478 (159) |
| Model A on `synth_test_pos` | 1.000 (542) | 1.000 (159) |

Model B reproduction through the harness (reported only, never used to adjust results): mAP50 0.9511 vs 0.9442
(+0.0069), mAP50-95 0.6730 vs 0.6688 (+0.0042); both within the pre-set tolerances (0.01 / 0.02). The small positive offset
is expected from the harness' own greedy matching and `predict` letterboxing versus Ultralytics `val`.
Model A validation curve (`model_a_training/results.csv`): val mAP50 0.923 at epoch 1, 0.994 by epoch 5, 0.995 from epoch
~10; val mAP50-95 0.669 (epoch 1) -> 0.983 (epoch 100); best epoch 99.

## Interpretation (what the evidence does and does not support)

Supported by the measurements:
- Same-domain performance is high for both models, but the synthetic in-domain task is **saturated**: Model A reaches the
  metric ceiling (0.995 mAP50) within about 5 epochs, so the Synth -> Synth cell cannot discriminate between
  training choices, and checkpoint selection on the synthetic val split is nearly arbitrary (best epoch 99 differs from
  neighbours by tiny amounts).
- Cross-domain transfer is asymmetric in this baseline. On the Boreal test set the synthetic-trained model scores 0.429
  mAP50 against 0.951 for the real-trained model (gap 0.522); on the synthetic test set the real-trained model scores 0.753
  against 0.995 (gap 0.242).
- Model A on Boreal misses far more non-edge-touching boxes (recall 0.497) than edge-touching ones (0.873), consistent
  with the documented edge-touch mismatch in its training data (about 0.76 touching vs Boreal 0.30). This is a
  *consistency*, not proof: edge-touching status is confounded with box size, scene content and appearance, and the groups
  are unequal (142 vs 571).

Not supported / not claimed:
- That the synthetic domain is worse in general or that the edge-touch mismatch is the cause. One generator
  configuration, one seed, one run; no variance estimate; no ablation.
- Any Level 3 conclusion. These numbers are the inputs for the later sensitivity/specificity estimates, not a UAV result.

## Limitations (state in any write-up)

- `synth_smoke_v1` edge-touch rate 0.76 vs Boreal 0.30 (documented, deliberately uncorrected); smoke is a flat billboard
  with no 3D occlusion.
- Boreal test: no usable negatives, site mix differs across splits (Heinola absent from test), 1 unlabeled image.
- Both models were trained with an identical, un-tuned recipe; neither was tuned, and Model B's own val curve plateaued early.
- The harness' metrics follow Ultralytics conventions closely but are not byte-identical (see reproduction check). A
  perfect detector scores 0.995.
- Training time differed by 2.3x between Model A and Model B on the same GPU type; not investigated (likely data loading or
  host differences), no effect on the recipe.

## Artifacts

- In git: this directory (`matrix.*`, `cells/*.json`, `datasets_pinned.yaml`, `model_a_training/`, driver logs).
- Not in git, local backup with `SHA256SUMS`: `local_data/backups/synth_smoke_a_v1/` (checkpoint `synth_smoke_a_v1/synth_smoke_best_v1.pt`,
  last/best weights, plots, `level2_v1/predictions/*.json.gz` so every metric can be recomputed with
  `python3 -m eval.run_matrix --preds-from ...`, full training log).
- On the RunPod volume: `/workspace/runs/synth_smoke_a_v1/`, `/workspace/runs/level2_v1/`, `/workspace/synth_smoke_v1/`.
  Model B's checkpoint was not touched (SHA-256 re-checked after the run: `1dc35ef2...8355b6`).

## Spend

Pod about 30 min x $0.89/hr, estimated about $0.45 (billing had not posted the pod when this was written; reconcile in
`docs/project-status.md`).
