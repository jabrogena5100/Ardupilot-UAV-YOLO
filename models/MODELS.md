# Model registry

Checkpoints are never committed to git (`*.pt` is gitignored). The primary copy lives on the RunPod network volume `wildfire-datasets` (`sej40ez3ia`, US-IL-1); a verified backup is kept on the local Ubuntu machine under `local_data/backups/`. Each entry below lists both locations and the SHA-256.

Naming: **Model B** (real / Boreal) is `real_boreal_best_v1.pt` below. **Model A** (synthetic) is `synth_smoke_best_v1.pt` below
(dataset `synth_smoke_v1`, recipe in `docs/synthetic/synth_smoke_v1_spec.md` section 5). Current project progress is in `docs/project-status.md`.

## synth_smoke_best_v1.pt — Model A (synthetic), v1 baseline

- Experiment: `docs/experiments/level2_v1.md` (Model A baseline on `synth_smoke_v1`; limitation: edge-touch 0.76 vs Boreal 0.30)
- Architecture: YOLOv8s (COCO-pretrained `yolov8s.pt`); AdamW, lr0 0.002, momentum 0.9, warmup_bias_lr 0.0; otherwise Model B's recipe
- Training: 100 epochs, imgsz 640, batch 32, seed 0, deterministic; best epoch 99 (by synthetic val fitness)
- Dataset: `synth_smoke_v1` (tag `synth-v1`), train 3,479 images; Model A never saw Boreal train/val/test
- SHA-256: `1da10c4814a905d92deec2d9c36185f872eabea447d47c389b9b065468009564`
- RunPod copy: `/workspace/runs/synth_smoke_a_v1/synth_smoke_best_v1.pt` (copy of `weights/best.pt`)
- Local backup: `local_data/backups/synth_smoke_a_v1/synth_smoke_a_v1/synth_smoke_best_v1.pt` (hash verified after transfer; `SHA256SUMS` alongside)
- Git status: excluded by `.gitignore` (`*.pt`); not tracked
- Full config + environment (pip freeze): `docs/experiments/level2_v1/model_a_training/train_config_v1.json` (in git)
- Trained: 2026-10-27, RunPod secure 1x RTX 4090, US-IL-1, wall time 996 s; ultralytics 8.4.173, torch 2.8.0+cu128
- Harness results: synthetic val/test_pos mAP50 0.995 (saturated); Boreal test mAP50 0.429, mAP50-95 0.216 (`docs/experiments/level2_v1/matrix.md`)

### Seed replicates (level2_v2, 2026-10-28)

Independent retrainings with the same documented protocol (100 epochs, imgsz 640, batch 32, deterministic, AdamW lr0 0.002, momentum 0.9,
warmup_bias_lr 0.0, best checkpoint by own-domain val). Seed 0 = the entries above/below. Experiment: `docs/experiments/level2_v2.md`.
Checkpoints are not in git (`*.pt`); RunPod copy under `/workspace/runs/<run>/`; local backup with `SHA256SUMS` under
`local_data/backups/level2_v2_seeds/<run>/`; per-run config/args/results in `docs/experiments/level2_v2/training/<run>/`.

| Checkpoint | Family | Seed | Run dir (RunPod) | Best epoch | Wall (s) | SHA-256 |
|---|---|---|---|---|---|---|
| `synth_smoke_best_v1_seed1.pt` | A (synthetic) | 1 | `synth_smoke_a_v1_seed1` | 97 | 1,000 | `31c4528e37c54f779cdbfbf02d8e325877108696f3608391c0490b22ec34a454` |
| `synth_smoke_best_v1_seed2.pt` | A (synthetic) | 2 | `synth_smoke_a_v1_seed2` | 99 | 1,000 | `3d5a31689ab212434b206949090c948a4de0b4b653abf8d0099619529146556a` |
| `real_boreal_best_v1_seed1.pt` | B (real) | 1 | `real_boreal_seed1` | 59 | 2,180 | `00c0e02fa2411993e160af4724ca9af6b3a17c2d25ac8e84761d4163175366c2` |
| `real_boreal_best_v1_seed2.pt` | B (real) | 2 | `real_boreal_seed2` | 91 | 2,180 | `fe53f08d414a8b09d799d4130a5cb1af62af82a5dc899339dd5a0571f0ab51d7` |

Seed 0 (`synth_smoke_best_v1.pt`, `real_boreal_best_v1.pt`) remains the reference pair used by `level2_v1`. Model B seed 0 used
`optimizer=auto`, which Ultralytics resolved to the same AdamW settings; the new runs set them explicitly. Aborted slow-host
attempts (`aborted_slowhost*` directories on the volume) produced no checkpoints used anywhere.

## Legacy artifacts (not registry entries)

Present only in git history, in `models/`; both deleted in `23dc32a` ("renamed models .pt files", 2026-10-03):

| Path | Size | SHA-256 | Notes |
|---|---|---|---|
| `models/real_boreal_bestv1.pt` (commit `73dab87`) | 5,442,970 B | `05ef7d467a638a5022e4d85e2089acb29a099530372e1b602dd5237fd8247a54` | Not Model B v1 (that file is 22,501,034 B with a different hash). Training provenance unrecorded; plausibly the "3-epoch demo checkpoint" mentioned in older docs, but nothing in the repo confirms it. |
| `models/synthetic_best.pt` | 0 B | n/a | Empty placeholder; never a trained model. `perception/base.py`'s docstring example still names this path. |

Do not cite either as Model A, Model B, or an experimental result.

## real_boreal_best_v1.pt — Model B (real / Boreal), v1

- Experiment: Real Boreal smoke detector (summary: `docs/experiments/real_boreal_v1.md`)
- Architecture: YOLOv8s
- Training: 100 epochs, imgsz 640, batch 32, seed 0
- Test mAP50: 0.944
- Test mAP50-95: 0.669
- SHA-256: `1dc35ef293b6b35bb2dadaf5b6d5887ece89ca9d55a445bfdde258286c8355b6`
- RunPod copy: `/workspace/runs/real_boreal/real_boreal_best_v1.pt` (copy of `weights/best.pt`)
- Local backup: `local_data/backups/real_boreal_v1/real_boreal_v1_backup/real_boreal_best_v1.pt` (verified against the SHA-256 above on 2026-10-05, together with `train_config_v1.json`, `test_metrics_v1.json`, `args.yaml`, `results.csv`, `manifest.json` and a `SHA256SUMS` file)
- Git status: checkpoint intentionally excluded by `.gitignore` (`*.pt`); not tracked
- Full config + environment (pip freeze): `/workspace/runs/real_boreal/train_config_v1.json`
- Trained: 2026-10-04, RunPod secure 1x RTX 4090, US-IL-1, wall time 2268 s (100 epochs, ~22 s/epoch)

### Training recipe (reuse unchanged for Model A)

| Item | Value |
|---|---|
| Model | `yolov8s.pt` (COCO-pretrained) |
| Epochs | 100, `patience=100` (no early stopping in effect) |
| imgsz / batch | 640 / 32 |
| seed / deterministic | 0 / True |
| workers | 8 |
| Other args | Ultralytics 8.4.173 defaults (full list in `args.yaml`) |
| Software | ultralytics 8.4.173, torch 2.8.0+cu128, CUDA 12.8, cuDNN 9.10.02, Python 3.12.3, driver 570.195.03 |
| Class list | `0: smoke` (single class; no fire class in Boreal) |

### Data

Frozen split `/workspace/splits/boreal_v1/` (symlinks to read-only `/workspace/boreal/boreal_yolo`):
the dataset's pre-existing split, verified video-disjoint with 0 cross-split md5 duplicates.

| Split | Images | Background | Videos |
|---|---|---|---|
| train | 3479 | 260 | 26 |
| val | 774 | 0 | 5 |
| test | 701 | 1 | 5 |

Per-split file-list SHA-256 values are in `manifest.json`; `train_config_v1.json` records them.

Caveats:
- 5 raw images have no label file and were deliberately NOT fixed (source is read-only):
  `Evo-Images/evoDJI_0001_frame23`, `Karkkila-Images/karkkila_DJI_0004_frame132`,
  `Ruokolahti-Images/ruokolahti_DJI_0087_frame{249,259,46}`. See `caveats.json`.
- Test set contains no explicit negative images (all 256 `Empty` negatives are in train), so test
  cannot measure false positives on smoke-free scenes.
- Site mix differs across splits (Heinola absent from test; Evo mostly in test).

### Results

Test (frozen test split, evaluated once, conf=0.001, iou=0.7; P/R at max-F1 operating point):

| P | R | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.937 | 0.933 | 0.944 | 0.669 |

Validation (best epoch 37): mAP50 0.876, mAP50-95 0.591. Final epoch 100 val mAP50-95: 0.561.
Test artifacts: `/workspace/runs/real_boreal_eval/{test_v1,test_predictions_v1}`.

Notes: `best.pt` is chosen by val fitness at epoch 37 of 100; val mAP50-95 fluctuated ~0.53-0.59 from
epoch ~30 onward while train box loss kept falling (0.54 at epoch 100 vs val box loss ~1.27),
i.e. no further gain from more epochs is evident on val. Test scores exceed val scores, which
likely reflects the different site/box-size mix rather than better generalization.

Scripts: `eval/runpod/boreal_stage1_freeze_smoke.sh`, `eval/runpod/boreal_stage2_train_eval.sh`.
