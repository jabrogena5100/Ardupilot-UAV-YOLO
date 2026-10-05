# Experiment: real_boreal_v1 (Model B, real / Boreal)

Date: 2026-10-04. Status: **complete** (100/100 epochs, exit code 0, test evaluation run once).

## Setup

| Item | Value |
|---|---|
| Model | YOLOv8s, `yolov8s.pt` (COCO-pretrained), 11,125,971 params (fused), 28.4 GFLOPs |
| Epochs | 100 (`patience=100`, no early stopping triggered) |
| imgsz / batch | 640 / 32 |
| Seed / deterministic | 0 / True |
| Workers | 8 |
| Other args | Ultralytics 8.4.173 defaults; full list in `args.yaml` on the volume |
| GPU | 1x NVIDIA GeForce RTX 4090 (24 GB), RunPod secure cloud, US-IL-1, driver 570.195.03 |
| Software | ultralytics 8.4.173, torch 2.8.0+cu128, CUDA 12.8, cuDNN 9.10.02, Python 3.12.3 |
| Training wall time | 2268 s (about 38 min, about 22 s/epoch) |
| Cost | about $0.52 for the training pod (about 42 min at $0.74/hr, estimated from pod run time, not from billing); about $0.08 for Stage 1 inspection and smoke test; about $0.60 total |

## Dataset and frozen split

Source: Boreal Forest dataset, `/workspace/boreal/boreal_yolo` (read-only, unmodified). Single class: `0: smoke`.
YOLO label format `cls cx cy w h`, normalized; all 4,862 boxes validated (none malformed or out of range).

The frozen split is the dataset's pre-existing `images/{train,val,test}` split, not rebuilt. It was verified
video-disjoint (0 shared videos between any two splits) with 0 cross-split exact (md5) duplicates.
It is materialised as a symlink farm at `/workspace/splits/boreal_v1/` (with `manifest.json`, `data.yaml`,
`caveats.json`). The split hashes were re-checked and matched before training began.

| Split | Images | With label file | Background | Videos | Boxes | Sites |
|---|---|---|---|---|---|---|
| train | 3479 | 3219 | 260 | 26 | 3304 | Empty 256, Evo 603, Heinola 735, Karkkila 765, Ruokolahti 1120 |
| val | 774 | 774 | 0 | 5 | 845 | Heinola 171, Karkkila 134, Ruokolahti 469 |
| test | 701 | 700 | 1 | 5 | 713 | Evo 328, Karkkila 197, Ruokolahti 176 |

File-list SHA-256 (sorted image names): train `9ed3ca0d...f9c4`, val `c1e3cabf...7763`, test `d2a610de...c41b`
(full values in `manifest.json`).

## Results

Final test evaluation (frozen test split, run once on `real_boreal_best_v1.pt`; conf=0.001, IoU=0.7;
P and R are the Ultralytics mean values at the max-F1 confidence point):

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.937 | 0.933 | 0.944 | 0.669 |

Inference speed on the 4090: about 0.8 ms inference per image (about 0.6 ms pre, 0.8 ms post).

Validation (selection only; the test set was not used for training or checkpoint choice):
best epoch 37 (val mAP50 0.876, mAP50-95 0.591); final epoch 100 val mAP50-95 0.561.

Training dynamics: val mAP50-95 varied between about 0.53 and 0.59 from epoch 30 onward while train box
loss kept falling (0.54 at epoch 100 against a val box loss of about 1.27). There is no evidence on val that
more epochs would help. Test scores are higher than val scores; this most likely reflects the different
site and box-size mix (median box area fraction: train 0.34, val 0.51, test 0.32), not stronger
generalization.

The test confusion matrix and 12 sample predictions were saved but have not been inspected by a human yet.
In the 12 sample predictions the model output exactly one smoke box per image.

## Artifacts (persistent, RunPod network volume `wildfire-datasets`, id `sej40ez3ia`, US-IL-1)

| Artifact | Path |
|---|---|
| **Checkpoint** (not in Git) | `/workspace/runs/real_boreal/real_boreal_best_v1.pt` (copy of `weights/best.pt`), SHA-256 `1dc35ef293b6b35bb2dadaf5b6d5887ece89ca9d55a445bfdde258286c8355b6` |
| Last-epoch weights | `/workspace/runs/real_boreal/weights/last.pt` |
| Run directory | `/workspace/runs/real_boreal/` |
| Per-epoch metrics (CSV) | `/workspace/runs/real_boreal/results.csv` |
| Training args | `/workspace/runs/real_boreal/args.yaml` |
| Full config + environment (JSON, includes pip freeze) | `/workspace/runs/real_boreal/train_config_v1.json` |
| Test metrics (JSON) | `/workspace/runs/real_boreal/test_metrics_v1.json` |
| Training curves and plots | `/workspace/runs/real_boreal/{results.png,BoxF1_curve.png,BoxPR_curve.png,BoxP_curve.png,BoxR_curve.png,confusion_matrix*.png,labels.jpg}` |
| Test eval outputs and plots | `/workspace/runs/real_boreal_eval/test_v1/` (confusion matrices, PR/F1/P/R curves, val_batch*_pred.jpg) |
| Test sample predictions | `/workspace/runs/real_boreal_eval/test_predictions_v1/` (12 images, seed 0, conf 0.25) |
| Full training log | `/workspace/runs/real_boreal_v1_train.log` |
| Frozen split | `/workspace/splits/boreal_v1/` |

Ultralytics did not write a separate JSON for these runs beyond the above (`save_json` was off); the
machine-readable records are `results.csv`, `args.yaml`, `test_metrics_v1.json` and `train_config_v1.json`.

In this repository: `models/MODELS.md` (model registry), `eval/runpod/boreal_stage1_freeze_smoke.sh` and
`eval/runpod/boreal_stage2_train_eval.sh` (the exact scripts used), and this file. `*.pt` is gitignored and
no weights are tracked.

The checkpoint exists only on the network volume. If the volume is deleted, the weights are lost, so keep a
copy elsewhere (for example via the RunPod console or `runpodctl`).

## Known caveats

- **No fire class.** Boreal labels smoke only, so this model cannot be compared on fire.
- **No negatives in test.** All 256 `Empty` background images are in train; test has 700 labeled images and
  1 unlabeled one. False positives on smoke-free scenes cannot be measured on this test set.
- **Site mix differs across splits.** Heinola is absent from test; Evo is mostly in test; val and test box-size
  distributions differ.
- **5 raw images have no label file** and were deliberately not fixed (source is read-only):
  `Evo-Images/evoDJI_0001_frame23`, `Karkkila-Images/karkkila_DJI_0004_frame132`,
  `Ruokolahti-Images/ruokolahti_DJI_0087_frame249`, `..._frame259`, `..._frame46`. They may be unlabeled
  smoke. Some of them may fall in train (4 non-Empty train images and 1 test image lack label files).
- **Single run, single seed.** Differences from Model A smaller than seed-to-seed variance should not be interpreted.
- The best checkpoint was selected by Ultralytics' fitness on val (epoch 37 of 100).
- This is a one-off RunPod launch, not yet the config-driven eval harness; the four-cell matrix must go
  through a single `(checkpoint, test-set)` code path.

## Reproduce for Model A

Use the same model, epochs, imgsz, batch, seed, deterministic setting, workers and Ultralytics version
(8.4.173); only the dataset changes. Keep the same fixed-split discipline.
