# Model registry

Checkpoints are never committed to git (`*.pt` is gitignored). The primary copy lives on the RunPod network volume `wildfire-datasets` (`sej40ez3ia`, US-IL-1); a verified backup is kept on the local Ubuntu machine under `~/backups/`. Each entry below lists both locations and the SHA-256.

## real_boreal_best_v1.pt — Model B (real / Boreal), v1

- Experiment: Real Boreal smoke detector (summary: `docs/experiments/real_boreal_v1.md`)
- Architecture: YOLOv8s
- Training: 100 epochs, imgsz 640, batch 32, seed 0
- Test mAP50: 0.944
- Test mAP50-95: 0.669
- SHA-256: `1dc35ef293b6b35bb2dadaf5b6d5887ece89ca9d55a445bfdde258286c8355b6`
- RunPod copy: `/workspace/runs/real_boreal/real_boreal_best_v1.pt` (copy of `weights/best.pt`)
- Local backup: `~/backups/real_boreal_v1/real_boreal_v1_backup/real_boreal_best_v1.pt` (verified against the SHA-256 above on 2026-10-05, together with `train_config_v1.json`, `test_metrics_v1.json`, `args.yaml`, `results.csv`, `manifest.json` and a `SHA256SUMS` file)
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
