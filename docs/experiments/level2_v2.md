# Experiment: level2_v2 (seed replication, protocol audit and uncertainty for the 2x2)

Date: 2026-10-27/28. Status: **complete**. Supersedes the single-seed reading of `level2_v1.md` (whose numbers are seed 0 of this series).
Nothing here is a final result: the synthetic dataset is `synth_smoke_v1` with a known edge-touch mismatch, n = 3 seeds per
family, one training recipe, no tuning.

## 1. Pre-GPU audits

### 1.1 Spend reconciliation
Billing was re-read before launching (storage $0.11-0.12/day; no pod charge yet posted for the previous run) and again after.
After this phase RunPod reports **$8.21 billed for October 2026**: pods $5.41 (GPU $5.23, CPU $0.15, disk $0.03) and
network-volume storage $2.80. Billing lags the pods: the three v2 pods ran about 33 + 9 + 120 minutes at $0.89/hr (about **$2.4**, from creation/termination
times) while billing had posted only about $1.6 of pod charges for 2026-10-27/28 at reading time. Using the conservative time-based
figure the cumulative spend is about **$9.1** and the remaining autonomous allowance **about $90.9** (billing-only: $91.79).
The level2_v1 pod estimate (about $0.45) is consistent with the billed 2026-10-27 GPU charge ($0.51, which also covers the first
minutes of the next pod). Ledger and the pending re-read: `docs/project-status.md`.

### 1.2 Are Model A and Model B training runs comparable?
Audited from `args.yaml`, `results.csv` and `train_config*.json` of all six runs (seed 0 = level2_v1 / `real_boreal_v1`):

| run | epochs done / set | patience | train images | iters/epoch | optimizer steps | final lr | wall (s) | median epoch (s) | best epoch |
|---|---|---|---|---|---|---|---|---|---|
| A seed 0 | 100 / 100 | 100 | 3,479 | 109 | 10,900 | 3.98e-5 | 996 | 9.4 | 99 |
| A seed 1 | 100 / 100 | 100 | 3,479 | 109 | 10,900 | 3.98e-5 | 1,000 | 9.6 | 97 |
| A seed 2 | 100 / 100 | 100 | 3,479 | 109 | 10,900 | 3.98e-5 | 1,000 | 9.6 | 99 |
| B seed 0 | 100 / 100 | 100 | 3,479 | 109 | 10,900 | 3.98e-5 | 2,268 | 21.9 | 37 |
| B seed 1 | 100 / 100 | 100 | 3,479 | 109 | 10,900 | 3.98e-5 | 2,180 | 21.2 | 59 |
| B seed 2 | 100 / 100 | 100 | 3,479 | 109 | 10,900 | 3.98e-5 | 2,180 | 21.5 | 91 |

All runs: YOLOv8s COCO-pretrained, imgsz 640, batch 32, AMP, deterministic, workers 8, close_mosaic 10, no early stopping
(patience 100; every run used all 100 epochs). Both families see the same number of training images (3,479, each including
260 background images) and the same number of optimizer steps. The only nominal argument difference across the six runs is
B seed 0's `optimizer=auto` (lr0 0.01, momentum 0.937, warmup_bias_lr 0.1 in `args.yaml`) versus the explicit AdamW settings
(lr0 0.002, momentum 0.9, warmup_bias_lr 0.0) used for the other five. Ultralytics resolved `auto` to exactly those AdamW values
for B seed 0 (spec section 5, verified from `results.csv` and the Ultralytics source), and the identical final learning rate
(3.98e-5) in all six runs corroborates it. This is a documented difference in recorded arguments, not in effective training.

**Runtime.** Wall time differed 2.2-2.3x between families on the same kind of host (A about 9.6 s/epoch, B about 21.5 s/epoch).
Measurements in this phase explain it as a data effect, not a protocol effect:
- Boreal training images are 4096x2160 and average **2,927 KB**; synthetic images are 1280x720 and average **142 KB** (probe run
  on the pod, 300 files each): about 20x the bytes and 6x the pixels to decode and resize per sample, with CPU-bound dataloader
  workers (GPU utilization about 21% in a slow-host run, workers at about 90% of a core).
- The same Model A recipe ran at 9.4-9.6 s/epoch on two hosts and **45 s/epoch** on a third machine (observed twice, 35 epochs
  and an aborted 5-epoch check; same GPU model, same dataset), so wall time is strongly host-dependent and says nothing about
  the training budget. Those two slow-host attempts were stopped (about $0.7 of GPU time, of which the second was mostly setup) and none of their output is
  used; their logs are in `training/aborted_slow_host/`.
- On a single fast host, B/A epoch-time ratio was 2.2.
Conclusion: the runtime difference does not affect comparability of training budget (epochs, optimizer steps, schedule and
precision are identical). It would matter only if wall time were itself a budget, which it is not in this protocol.

### 1.3 Bootstrap verification
`eval.metrics.bootstrap_ci` resamples **test images with replacement** (or, for Boreal, whole **source videos**) and, for every
resample, recomputes the complete detection metric from the pooled predictions: precision-recall curves, 101-point AP at IoU 0.5 and
at the ten thresholds 0.50:0.05:0.95, and precision/recall/FP-per-image at conf 0.25. Only the within-image greedy matching is
cached, which is exact because matching never crosses images. Tests (`tests/test_eval_metrics.py`) assert that the unresampled point
estimate equals the full `evaluate()` result to 9 decimals, that a resample (with duplicated images) equals a full recompute on the
resampled set, that the seeded procedure is deterministic, and that cluster resampling is wider than image resampling when units
are correlated. Assumptions and limits (also in the docstring):
- It measures **test-set sampling variability for one fixed checkpoint and fixed operating threshold**. It is **not training-seed
  uncertainty**; that is the across-seed SD below, and the two are never combined.
- Images are treated as independent; Boreal frames from the same video are correlated, so image-level Boreal intervals are optimistic.
  The video-cluster interval resamples the 5 test videos; with only 5 clusters it is very wide and the percentile method is crude.
- Near a metric ceiling (Synth -> Synth, mAP50 0.995) the intervals collapse to zero width and say nothing about real precision.
- Synthetic test images have no cluster structure beyond independent scenes drawn from the generator.

### 1.4 Do the evaluation splits contain background-only images?
| split | background-only images | notes |
|---|---|---|
| Boreal train | 260 (256 `Empty-Images__*` true negatives + 4 raw frames with no label file) | Model B trained on them |
| Boreal val | 0 | |
| **Boreal test** | **1** | an unlabeled raw frame (recorded caveat), **not a verified negative**; effectively **no usable negatives** |
| `synth_smoke_v1` train | 260 | |
| `synth_smoke_v1` val | 0 | |
| `synth_smoke_v1` test | **350** (view `synth_test_all` = 701 positive + 350 background) | true negatives by construction |

Consequences: precision measured on positive-only sets (`boreal_test`, `synth_test_pos`) is not a false-positive assessment.
Empty-scene false-positive rates are reported separately: on the 350 synthetic negatives, and on a **diagnostic** set of the 256
true Boreal `Empty-Images` negatives from the Boreal *train* split (`boreal_empty_train_diag`, file-list SHA-256
`acd1b1aec93b989aef8282f8b62e06d236681ad7e183ea5fc1debf69bbbd3743`). That diagnostic is not a frozen test set; it is a valid probe for
Model A (never trained on Boreal) but **Model B trained on those images, so its numbers there are optimistic** and must not be
compared with Model A's.

## 2. Setup
- Four new independent training runs, seeds 1 and 2 for each family, identical protocol (above), distinct names:
  `synth_smoke_best_v1_seed1.pt`, `synth_smoke_best_v1_seed2.pt`, `real_boreal_best_v1_seed1.pt`, `real_boreal_best_v1_seed2.pt`
  (SHA-256 in `models/MODELS.md`). Existing seed-0 checkpoints were hash-verified before and after and not modified.
- Frozen datasets untouched (10,616 synthetic file hashes verified before and after; Boreal split pins verified by the harness).
- All six checkpoints evaluated through the same harness on `boreal_test` (701), `synth_test_pos` (701), `synth_test_all` (1,051),
  `boreal_empty_train_diag` (256): 24 cells; inference on the pod, then metrics and 1,000-resample bootstraps recomputed offline from
  the saved predictions and ground truth (no GPU).
- Environment (training and inference): RunPod secure 1x RTX 4090, ultralytics 8.4.173, torch 2.8.0+cu128, CUDA 12.8, cuDNN 9.10.02,
  Python 3.12.3 (Model B seed 0: driver 570.195.03; the new runs: host driver 595.91.07).

## 3. Deviations and incidents
1. Model B seeds 1-2 use explicit AdamW arguments, B seed 0 used `optimizer=auto` (resolved identically; see 1.2).
2. Seeds 1-2 ran on a different physical host and pod than seeds 0 (different driver, CPU). GPU model, software versions and the
   recipe are identical; PyTorch is not guaranteed bit-reproducible across hosts, which is part of what the seed spread measures.
3. Two launch attempts landed on a slow host (45 s/epoch for Model A) and were aborted by hand and then by a speed guard added to
   `eval/runpod/level2_seeds_train_and_matrix.sh` after the first; none of their results are used. Total pod spend for the phase
   was about $2.4 (three pods, time-based) instead of the planned $2.0; the per-run timeout in the script was raised from 6,000 s to 9,000 s.
4. The harness' inference path (`predict`) is not Ultralytics `val`: on Model B seed 0 it reproduces the one-off test numbers within
   the pre-set tolerance (mAP50 0.9511 vs 0.9442).
5. `eval_sim_synthgen_dirty: true` in the cell records comes from an unrelated uncommitted docstring edit in
   `synthgen/generate_dataset.py` (path text only), not from evaluation code.
6. Checkpoint selection uses each family's own-domain val split; Model A's saturated synthetic val makes its best epoch nearly
   arbitrary (97-99), Model B's best epochs differ widely (37, 59, 91) on a noisy val curve.

## 4. Results (measured)
Source: `seed_aggregate.md` / `.json`, `matrix.md` / `.csv`, `cells/*.json` in `docs/experiments/level2_v2/`. Operating point for P/R/FP:
IoU 0.5, conf >= 0.25. **SD = sample SD across the 3 seeds (n = 3, itself very uncertain); the bootstrap intervals are a different
quantity.**

### 4.1 mAP50 per seed
| cell | seed 0 | seed 1 | seed 2 | mean | SD | min-max |
|---|---|---|---|---|---|---|
| Synth -> Synth | 0.995 | 0.995 | 0.995 | 0.995 | 0.000 | 0.995-0.995 |
| Synth -> Real | 0.429 | 0.384 | 0.313 | 0.375 | 0.059 | 0.313-0.429 |
| Real -> Synth | 0.753 | 0.656 | 0.869 | 0.759 | 0.107 | 0.656-0.869 |
| Real -> Real | 0.951 | 0.956 | 0.953 | 0.954 | 0.003 | 0.951-0.956 |

mAP50-95 (mean +- SD): Synth -> Synth 0.986 +- 0.001; Synth -> Real 0.164 +- 0.050; Real -> Synth 0.390 +- 0.076; Real -> Real
0.675 +- 0.004. Recall at 0.25: 1.000, 0.519 +- 0.046, 0.627 +- 0.105, 0.947 +- 0.002. FP per image at 0.25: 0.000, 0.436 +- 0.072,
0.093 +- 0.058, 0.070 +- 0.044. Full tables including per-seed values: `seed_aggregate.md`.

### 4.2 In-domain versus cross-domain change (cross minus in-domain, same family, same seed)
| family | mAP50 per seed | mean | SD | mAP50-95 mean | recall@0.25 mean |
|---|---|---|---|---|---|
| A: Synth -> Real minus Synth -> Synth | -0.566, -0.611, -0.682 | -0.620 | 0.059 | -0.822 | -0.481 |
| B: Real -> Synth minus Real -> Real | -0.198, -0.301, -0.084 | -0.194 | 0.108 | -0.284 | -0.320 |

Absolute cross-domain levels (mean mAP50): on Boreal test, real-trained 0.954 vs synthetic-trained 0.375; on the synthetic test, synthetic-trained
0.995 vs real-trained 0.759. Because Model A's in-domain score is at the metric ceiling, the "drop" for A is measured from a saturated
baseline and the two drops are not on a common scale; prefer the absolute levels.

### 4.3 Test-set sampling uncertainty (bootstrap, 1,000 resamples, 95%, fixed checkpoint)
Per-seed intervals are in `seed_aggregate.md`. Examples: Real -> Real mAP50 images [0.925-0.930, 0.973-0.975], videos
[0.845-0.862, 0.994-0.995]; Synth -> Real mAP50 images [0.280, 0.474] across seeds, videos [0.094, 0.969]; Real -> Synth
mAP50 images [0.618, 0.892] across seeds (no video structure). Image-level intervals (+-0.02-0.05) are much narrower than the
across-seed spread for Synth -> Real and Real -> Synth (SD 0.059, 0.107): **seed variability, not test-set sampling, dominates**
those cells. The Boreal video-cluster intervals are enormous (5 videos) and show that image-level Boreal intervals understate the
uncertainty about generalizing to new videos.

### 4.4 Background-only (empty-scene) false-positive rate, image level at conf >= 0.25
| model | 350 synthetic negatives | 256 Boreal `Empty-Images` (train split; diagnostic) |
|---|---|---|
| A seed 0 / 1 / 2 | 0.000 / 0.000 / 0.000 | **0.379 / 0.141 / 0.496** |
| B seed 0 / 1 / 2 | 0.011 / 0.043 / 0.040 | 0.004 / 0.004 / 0.039 (optimistic: B trained on these) |

Model A produces no detections on synthetic empty scenes but fires on 14-50% of real empty forest scenes it never saw. Synthetic
negatives alone would therefore give a misleadingly good specificity for Model A.

### 4.5 Pre-declared edge-touch diagnostic (recall at conf 0.25; GT boxes touching the border vs not)
| model | Boreal edge-touching (n=142) | Boreal not edge-touching (n=571) |
|---|---|---|
| A seed 0 / 1 / 2 | 0.873 / 0.746 / 0.746 | 0.497 / 0.434 / 0.424 |
| B seed 0 / 1 / 2 | 0.993 / 0.993 / 1.000 | 0.933 / 0.933 / 0.937 |

## 5. Interpretation (what the evidence supports)
Supported, with n = 3 and one recipe:
- Real -> Real is high and stable across seeds (mAP50 0.951-0.956). Synth -> Synth is saturated and cannot discriminate.
- Synth -> Real is low in every seed (0.313-0.429), far below Real -> Real in every seed (smallest gap 0.52). Synth -> Real
  variation across seeds (SD 0.059) is large compared with its image bootstrap interval.
- Real -> Synth is intermediate on average (0.759) but **highly seed-dependent** (0.656-0.869); a single seed (as in `level2_v1`) would
  have misrepresented it. Model A's cross-domain mAP50 drop is larger than Model B's in all three seed pairs, but the drops are
  measured from different baselines (see 4.2) and with n = 3 no significance test is claimed.
- The synthetic-trained model misses non-edge-touching real boxes far more than edge-touching ones (0.42-0.50 vs 0.75-0.87) in all seeds;
  the real-trained model shows a much smaller gap (0.06). This is consistent with, not proof of, the documented edge-touch mismatch
  (synthetic train ~0.76 vs Boreal ~0.30) contributing; edge status is confounded with box size, context and appearance.
- Synthetic empty scenes underestimate false-positive risk for the synthetic-trained model on real empty scenes (4.4).

Not supported: that the synthetic domain is worse in general or that edge-touch is the cause; any ranking of recipes; any
Level 3 claim; any statement about statistical significance; the Boreal-negatives result for Model B (trained on them).

## 6. Limitations
Edge-touch mismatch in `synth_smoke_v1`; flat-billboard smoke, no 3D occlusion; Boreal test has no usable negatives, 5 test videos only, site mix
differs across splits; three seeds per family on one recipe; the real-trained model's val curve is noisy (best epochs 37/59/91), so
checkpoint choice adds variance; both families are un-tuned; seeds 1-2 ran on a different host than seed 0; the harness follows but is not identical to
Ultralytics `val`; mAP of a perfect detector is 0.995; the `boreal_empty_train_diag` set is a train-split diagnostic, not a test set.

## 7. Artifacts and reproduction
- In git (`docs/experiments/level2_v2/`): `cells/*.json` (24 cells, bootstrap included, offline recompute), `pod_cells/*.json` (24 cells as computed on the pod, with
  inference provenance), `seed_aggregate.{md,json}`, `matrix.{md,csv}`, `datasets_pinned_on_pod.yaml`, `training/<run>/{train_config.json,args.yaml,results.csv}`,
  driver logs, `training/aborted_slow_host/`.
- Local backup with `SHA256SUMS` (not in git): `local_data/backups/level2_v2_seeds/` (4 checkpoints, plots, full training logs, `level2_v2/predictions/*.json.gz` and
  `gts/*.json.gz`, so every number is recomputable with no GPU).
- RunPod volume: `/workspace/runs/{synth_smoke_a_v1_seed1,synth_smoke_a_v1_seed2,real_boreal_seed1,real_boreal_seed2,level2_v2}`; aborted-attempt directories are kept under
  `aborted_slowhost*` names.
- Recompute everything offline:
  `python3 -m eval.run_matrix --config eval/datasets.yaml --out OUT --var BOREAL_SPLIT=/x --var SYNTH_ROOT=/x --var RUNS=/x --preds-from PRED_DIR --gts-from GT_DIR --bootstrap 1000`
  then `python3 -m eval.aggregate_seeds --cells OUT/cells --out OUT/seed_aggregate.md`.
- Pod script: `eval/runpod/level2_seeds_train_and_matrix.sh`. Harness: `eval/metrics.py`, `eval/run_matrix.py`, `eval/aggregate_seeds.py`; tests: `tests/test_eval_metrics.py`, `tests/test_run_matrix.py`, `tests/test_aggregate_seeds.py`.

## 8. Spend
Phase pods about $2.4 by running time (about $1.6 posted by billing so far). Billed October total $8.21; conservative cumulative about $9.1;
remaining autonomous allowance about $90.9. No pod is running (`list-pods` empty). Re-read billing next session and replace the estimate.
