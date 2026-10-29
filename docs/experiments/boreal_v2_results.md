# Boreal v2: Level 2 results (clip-disjoint split, three Model B seeds)

Date: 2026-10-29. Status: **complete** (3 Model B v2 seeds trained, 6 models x 5 datasets evaluated through the unified harness, results recomputed offline). Split construction and limitations: `docs/experiments/boreal_v2.md`. Boreal v1 and the Level 2 v2 experiment are untouched and remain the historical baseline. **Boreal v1 and v2 numbers are not directly comparable: the Boreal train and test populations differ (v2 test = 8 other clips, 744 frames incl. 65 true negatives).** Not implemented: Level 3 probability calibration.

## 1. Headline: the revised 2x2 (mean over 3 seeds, sample SD in the last column, per-seed values shown)

Boreal column = `boreal_v2_test` (744 frames). Synthetic column = `synth_test_pos` (701). Operating point @0.25, IoU 0.5. SD with n=3 is itself very uncertain.

### mAP50

| cell | seed 0 | seed 1 | seed 2 | mean | SD |
|---|---|---|---|---|---|
| Synth -> Synth (Model A on synth_test_pos) | 0.995 | 0.995 | 0.995 | **0.995** | 0.000 |
| Synth -> Real (Model A on boreal_v2_test) | 0.602 | 0.522 | 0.413 | **0.512** | 0.095 |
| Real -> Synth (Model B-v2 on synth_test_pos) | 0.587 | 0.522 | 0.726 | **0.612** | 0.104 |
| Real -> Real (Model B-v2 on boreal_v2_test) | 0.923 | 0.903 | 0.870 | **0.899** | 0.027 |

### mAP50-95

| cell | seed 0 | seed 1 | seed 2 | mean | SD |
|---|---|---|---|---|---|
| Synth -> Synth (Model A on synth_test_pos) | 0.985 | 0.987 | 0.986 | **0.986** | 0.001 |
| Synth -> Real (Model A on boreal_v2_test) | 0.283 | 0.244 | 0.186 | **0.238** | 0.049 |
| Real -> Synth (Model B-v2 on synth_test_pos) | 0.274 | 0.245 | 0.348 | **0.289** | 0.053 |
| Real -> Real (Model B-v2 on boreal_v2_test) | 0.547 | 0.539 | 0.508 | **0.531** | 0.021 |

### Recall @0.25

| cell | seed 0 | seed 1 | seed 2 | mean | SD |
|---|---|---|---|---|---|
| Synth -> Synth (Model A on synth_test_pos) | 1.000 | 1.000 | 1.000 | **1.000** | 0.000 |
| Synth -> Real (Model A on boreal_v2_test) | 0.570 | 0.514 | 0.506 | **0.530** | 0.035 |
| Real -> Synth (Model B-v2 on synth_test_pos) | 0.495 | 0.405 | 0.589 | **0.496** | 0.092 |
| Real -> Real (Model B-v2 on boreal_v2_test) | 0.920 | 0.906 | 0.871 | **0.899** | 0.025 |

### Precision @0.25

| cell | seed 0 | seed 1 | seed 2 | mean | SD |
|---|---|---|---|---|---|
| Synth -> Synth (Model A on synth_test_pos) | 1.000 | 1.000 | 0.999 | **1.000** | 0.001 |
| Synth -> Real (Model A on boreal_v2_test) | 0.702 | 0.616 | 0.519 | **0.612** | 0.091 |
| Real -> Synth (Model B-v2 on synth_test_pos) | 0.853 | 0.816 | 0.939 | **0.869** | 0.063 |
| Real -> Real (Model B-v2 on boreal_v2_test) | 0.927 | 0.882 | 0.797 | **0.869** | 0.066 |

### False positives per image @0.25

| cell | seed 0 | seed 1 | seed 2 | mean | SD |
|---|---|---|---|---|---|
| Synth -> Synth (Model A on synth_test_pos) | 0.000 | 0.000 | 0.001 | **0.000** | 0.001 |
| Synth -> Real (Model A on boreal_v2_test) | 0.224 | 0.297 | 0.434 | **0.319** | 0.106 |
| Real -> Synth (Model B-v2 on synth_test_pos) | 0.086 | 0.091 | 0.039 | **0.072** | 0.029 |
| Real -> Real (Model B-v2 on boreal_v2_test) | 0.067 | 0.113 | 0.206 | **0.129** | 0.071 |

Reading it (descriptive, not a significance claim): the in-domain real cell is far above the cross-domain cells; synthetic-trained models reach near-perfect scores on synthetic test scenes (saturated, cannot discriminate) but recover only about half of Boreal smoke (recall 0.53 @0.25, mAP50 0.51); Boreal-trained models reach only mAP50 0.61 and recall 0.50 on synthetic smoke. The qualitative pattern matches Level 2 v2 on the v1 split; the magnitudes are on different populations and are not compared.

## 2. Uncertainty (kept separate: seed variability vs test-set sampling)

- **Seed variability** (above): SD over 3 independently trained seeds. Largest for the cross-domain cells (mAP50 SD 0.095 Synth->Real, 0.104 Real->Synth) and smallest in-domain real (0.027).
- **Clip-clustered bootstrap** (Boreal v2 test = **8 source clips**, 1000 resamples, percentile; test-set sampling only, per fixed checkpoint): Real->Real mAP50 0.923 [0.838, 0.966], 0.903 [0.807, 0.967], 0.870 [0.699, 0.940] for seeds 0/1/2; Synth->Real 0.602 [0.175, 0.810], 0.522 [0.087, 0.756], 0.413 [0.036, 0.716]. With 8 clusters the intervals are very wide and the Real->Real and Synth->Real ranges approach each other at their edges (0.838 vs 0.810 for seed 0), so **the size of the domain gap is not tightly estimated** even though every seed shows the same ordering. The percentile method is crude with so few clusters.
- Image-level bootstrap (frames treated as independent) is in the cells and is optimistic for Boreal; it is not used for conclusions.
- **Synthetic test cells** use image-level and texture-slot (6 clusters) bootstrap: Model B-v2 on synth_test_pos mAP50 0.587 [0.436, 0.822], 0.522 [0.375, 0.730], 0.726 [0.638, 0.823] (`synth_texture_cluster_bootstrap.json`); Model A is saturated at 0.995 with zero-width intervals.
- Full tables: `results/report.md`, `results/level2/matrix.md`, `results/level2/seed_aggregation.md`.

## 3. Negative-scene diagnostics (empty-scene false positives)

Held-out real negatives: 65 Empty-Images frames from `heinola_dji_0048` (47) and `karkkila_dji_0003` (18); neither family trained on these clips. Fraction of empty images with at least one detection >= 0.25:

| family | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Model A (synthetic-trained) on real negatives | 0.138 (9/65) | 0.169 (11/65) | 0.262 (17/65) |
| Model B-v2 (real-trained) on real negatives | 0.015 (1/65) | 0.015 (1/65) | 0.015 (1/65) |
| Model A on the 350 synthetic empty scenes | 0.000 | 0.000 | 0.000 |
| Model B-v2 on the 350 synthetic empty scenes | 0.069 | 0.089 | 0.014 |

The synthetic-trained model, which never produces a false positive on synthetic empty scenes, fires on 14-26% of real empty forest frames; the real-trained model fires on 1 of 65. Caveats: 65 frames from only 2 clips; both clips are pre-smoke background of clips that also contain smoke; Model B-v2's training set contains 132 Empty-Images frames from other clips.

## 4. Other diagnostics (Boreal v2 test, @0.25)

- **Edge-touching smoke**: Model A recall 0.892 / 0.703 / 0.689 on boxes touching the image border versus 0.482 / 0.463 / 0.456 elsewhere; Model B-v2 0.966 / 0.939 / 0.946 vs 0.908 / 0.897 / 0.851. The documented synthetic edge-touch mismatch (0.76 vs 0.30) is a candidate explanation for Model A's weak non-edge recall; not tested here.
- **Localisation** (mean IoU of true positives): Model A 0.74, Model B-v2 0.79-0.80. Mean confidence of TPs: Model A 0.84-0.90, Model B-v2 0.78-0.81 (Model A is more confident and less right).
- **Near-duplicate sensitivity**: dropping the 6 test frames with a <=4-bit near-duplicate in train/val changes no model's mAP50 by more than 0.001 (`results/report.md`).

## 5. Experiment inventory

### 5.1 Dataset (frozen before any training)
Boreal v2 split (31 normalised clips; construction, constraints, design history and audit in `boreal_v2.md`): train 3462 / val 748 / test 744 frames. File-list SHA-256: train `5b23b199965dfc47372e6100f4d5bfd41d57ef5b0b16b5ed83e91db056245c25`, val `8ef9e902bb24a720112bef32d20248d6aa33c17d67efc9771643daa87d252e2f`, test `9fa6dbc1d0eb9fa437e989be29afadc9a459f7850502f1c18c350379afcddc8f` (re-verified by the pod before every run and recorded in `results/INTEGRITY.json`). Harness views: `boreal_v2_test` 744, `boreal_v2_test_pos` 679, `boreal_v2_test_neg` 65, plus the pinned `synth_test_pos` (701) and `synth_test_all` (1051).

### 5.2 Training (Model B v2; Model A not retrained)

Protocol (unchanged from Level 2 v2): YOLOv8s from COCO `yolov8s.pt`, 100 epochs, patience 100, imgsz 640, batch 32, deterministic, workers 8, AdamW lr0 0.002 momentum 0.9 warmup_bias_lr 0.0, ultralytics 8.4.173, torch 2.8.0+cu128, CUDA 12.8, 1x RTX 4090 (secure, US-IL-1), driver 595.91.07. Checkpoint = Ultralytics `best.pt` (fitness on the v2 val split); test never used for selection.

**Documented protocol deviation: `cache=ram`** for all three seeds (Ultralytics decodes and resizes each image once and keeps it in RAM). Reason: two successive RunPod hosts decoded the 4K frames about 6x slower than the v1 pod (130-150 s/epoch). Checks before training (`results/PROTOCOL_DEVIATION.json`, `eval/runpod/cache_equivalence_check.py`, probe pod `hvpedopu4i1ics`): cached and uncached loads are byte-identical on 24/24 probe images and 16/16 in-job images (shape 338x640x3 uint8); RAM need 2.76 GB for train+val vs 41-83 GB available; `cache=disk` was pixel-identical too but would need about 111 GB (rejected, volume is 50 GB). v1 Model B was trained without a cache. Ultralytics prints that `cache=ram` 'may produce non-deterministic training results'; evidence here: seed 1's first 4 epochs trained on two different pods matched exactly (44 values, max abs diff 0), but only 4 epochs were comparable.

| seed | run (attempt) | epochs | train wall (s) | median epoch (s) | best epoch (val mAP50-95) | val mAP50 / mAP50-95 at best | checkpoint SHA-256 |
|---|---|---|---|---|---|---|---|
| 0 | `boreal_v2_try3` | 100 | 4983 | 45.1 | 86 | 0.951 / 0.620 | `090c0f4cbe2718d2b084a41b8a3736fa27b673bc66b20c64672b13c83c299a05` |
| 1 | `boreal_v2_try4` | 100 | 1021 | 9.5 | 66 | 0.959 / 0.625 | `5a9616cefddd3bca6eccf31d0691f7b1a2fadb572f84cf7948c85296644e9388` |
| 2 | `boreal_v2_try4` | 100 | 1020 | 9.4 | 23 | 0.970 / 0.624 | `c4d443a8af919c6b00e4319b117a5dfbe90bd8b29750c75e3970a9389ae097ef` |

Hosts differed strongly in speed (shared hosts, load 9-31): seed 0 ran on a busier host than seeds 1-2. The pinned existing checkpoints (Model A seeds 0-2, Model B v1 seeds 0-2) were re-hashed before and after the run: all unchanged (`results/INTEGRITY.json`).

### 5.3 Attempt log (all preserved; nothing overwritten)

| # | tag / pod | outcome | est. cost |
|---|---|---|---|
| 1 | `boreal_v2` / `kv3imy4lnltlea` | aborted by speed guard at epoch 4 (153 s/epoch, shared host); pod could not self-stop (REST 403), stopped by hand | $0.42 |
| 2 | `boreal_v2_try2` / `g9pmq8ek0cx0au` | aborted by speed guard at epoch 4 (about 130 s/epoch); pod self-stopped (GraphQL) | $0.24 |
| 3 | `boreal_v2_try3` / `vi2o551fcsho16` | **seed 0 complete**; seed 1 aborted at epoch 4 by the 45 s guard (57 s/epoch, host got busier); pod self-stopped | $1.46 |
| 4 | `boreal_v2_try4` / `lh2zdraedw4bpq` | **seeds 1-2 complete** and the full 6-model x 5-dataset evaluation; results fetched via the proxy; pod self-stopped | $0.72 |
| - | diagnostic/probe pods (7: create probes, diag x2, stop probe, cache check) | host/IO diagnosis, stop-method and cache feasibility | about $0.09 |

**Estimated GPU spend for the whole experiment: $2.93** (RunPod pod start/stop timestamps x $0.89/h; billing had posted $11.27 total at last read and lags by hours; container-disk charges for stopped pods are small and not itemised). Within the $20 per-experiment limit; cumulative conservative spend about $12.3 of the $100 autonomous allowance.

### 5.4 Artifact locations

- **RunPod volume `wildfire-datasets` (`sej40ez3ia`)**: checkpoints `/workspace/runs/boreal_v2_try3/real_boreal_v2_seed0/real_boreal_v2_best_seed0.pt`, `/workspace/runs/boreal_v2_try4/real_boreal_v2_seed{1,2}/real_boreal_v2_best_seed{1,2}.pt`; full run dirs, logs, plots, `results_bundle.tgz` per run; supervisor records `/workspace/runs/boreal_v2*_train_eval_orch/`; split `/workspace/splits/boreal_v2/`. **Checkpoints are not in git and no local backup exists yet** (owner step; hashes above).
- **Git (`science/boreal-v2`, local, not pushed)**: `docs/experiments/boreal_v2/results/` = `level2/` (30 cells, `matrix.{md,csv}`, `seed_aggregation.{md,json}`, saved `predictions/` and `gts/`, pinned config), `training/` (args, results.csv, train_config.json per seed; aborted-attempt records), `report.{md,json}`, `synth_texture_cluster_bootstrap.json`, `recompute_check.json`, `INTEGRITY.json`, `PROTOCOL_DEVIATION.json`, `checkpoint_hashes_try3.txt`. Code: `eval/boreal_v2_split.py`, `eval/boreal_v2_recheck.py`, `eval/boreal_v2_report.py`, `eval/datasets_boreal_v2.yaml`, `eval/runpod/boreal_v2_train_eval.sh`.

### 5.5 Reproduction and review notes

- **Offline recompute**: all 30 cells recomputed from the saved predictions and ground truth on a different machine (`recompute_check.json`): every detection metric (mAP, P/R, TP/FP/FN, bootstrap intervals) is identical; 107 localisation statistics (mean/median IoU of TPs, centre error) differ by at most 4.2e-6 relative (NumPy 2.1.2 on the pod vs 2.5.3 here). Reproduction is exact for detection metrics, not bit-identical for localisation means.
- Training reproduction on a fresh pod has been checked only for seed 1's first 4 epochs (identical). Full-run bit reproducibility is not demonstrated; `cache=ram` carries Ultralytics' non-determinism warning.
- Pod inference provenance: the pods ran a bundled copy of the repo; `git_commit` is recorded (`6d73d89d...` seed 0, `0b242fb4...` seeds 1-2) but those commits are local and unpushed (by instruction); the bundle hashes are in each `train_config.json`.

## 6. Limitations

- **Few clips**: 31 clips overall, 8 in test; one defensible assignment among many near-equal ones (`boreal_v2.md` section 3). Clustered intervals are very wide.
- **Residual leakage risk** (documented, not proven independence): 22 train/test frame pairs within 4 dHash bits (6 distinct test frames, same-site sessions `heinola_dji_0028`/`0048`, `0048`/`ruokolahti_dji_0089`); consecutive-numbered same-site clips in different partitions (4 pairs, e.g. `karkkila_dji_0003`/`0004`, `0006`/`0007`); the `karkkila_dji_0007`/`0008` ambiguity (kept together in val). Sensitivity to the 6 frames is nil (<= 0.001 mAP50).
- **Test composition**: no Evo clip in test (Evo is in val and train); real negatives only 65 frames from 2 mixed clips; 5 unlabeled non-Empty frames sit in train (kept as v1 did).
- **Model A is a baseline** on `synth_smoke_v1` (edge-touch 0.76 vs Boreal 0.30; flat-billboard smoke); not evidence that the synthetic domain is optimal.
- **Seeds**: n=3; seed 0 trained in a different run and on a different host than seeds 1-2; `cache=ram` for all (deviation above); speed-guard thresholds differed (45 s vs 65 s) but do not affect training.
- **Checkpoint selection** uses the v2 val split (5 clips); val scores (0.95-0.97 mAP50) exceed test scores (0.87-0.92): different clips and site mix.

## 7. Open questions for review
1. Is a 3-seed, 8-clip test enough to support a domain-gap claim, or should a second clip-disjoint assignment (alternative split from the recorded near-optimal set) be run as a robustness check?
2. Pixel-level identity of consecutive same-site clips (UNKNOWN).
3. Does matching the edge-touch rate in synthetic data (a versioned `synth_smoke_v2`) close the Synth->Real gap? (post-hoc dataset improvement; not done.)
4. Level 3 observation contract quantities (sensitivity/specificity) can now use Boreal v2 negatives; no calibration code written.

