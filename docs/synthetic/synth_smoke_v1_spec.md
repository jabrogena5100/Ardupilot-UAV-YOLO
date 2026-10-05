# synth_smoke_v1: specification (candidate, NOT yet frozen)

Condition name: **`Boreal-prior-matched`**. Boreal train/val statistics informed the synthetic box-geometry priors
(`docs/synthetic/boreal_visual_analysis.md`); Boreal **test** was never used. No Boreal pixel is used anywhere.

Status (2026-10-05, updated at end of day): **frozen at `9cb0f1a` (tag `synth-v1`); full dataset GENERATED and
validated** (see `docs/datasets/synth_smoke_v1.md`). Model A has not been trained, and the dataset has not been
uploaded to the RunPod volume; both need separate approval. The text below describes the plan as written before
generation (including "Nothing at full scale has been generated", which is now outdated). POC v2 (110 images, `~/synth_poc_v2`, not committed)
validates and is byte-reproducible.

## 1. Approved dataset design

| Split | Smoke | Background | Total | Notes |
|---|---|---|---|---|
| train | 3,219 | 260 | 3,479 | exactly Boreal's train composition |
| val | 774 | 0 | 774 | Boreal val composition (all smoke); same checkpoint-selection pressure |
| test | 701 | 350 | 1,051 | two views, reported separately: `test_pos` (701 smoke) and `test_all` (701 + 350 background) |

Explicit counts on the command line: `--counts train=3219:260,val=774:0,test=701:350 --seed 1001 --dataset-id synth_smoke_v1`.
`generate_dataset` writes `lists/test_pos.txt`, `lists/test_all.txt` and `data_test_pos.yaml` / `data_test_all.yaml`;
the validator checks that `test_pos` has no background image and is a subset of `test_all`.

Size is matched to Boreal so that Model A and Model B get the same number of gradient updates (109 iterations per
epoch, 10,900 in 100 epochs). Boreal's frames come from 26 videos and are correlated, so equal nominal counts still
give Model A more independent scenes than Model B has; this is a known, accepted asymmetry.

## 2. Seeds and versioning

- Base seed **1001**. Streams (all `numpy.random.SeedSequence`): scene `[1001, split_id, index, attempt]`;
  image-level draws (archetype, box size, edge decision) `[1001, split_id, index, 777000]`; texture bank
  `[1001, split_id, 10000+slot]`; smoke-sprite bank `[1001, split_id, 20000]`; background selection
  `[1001, split_id, 424242]`; split ids train 0, val 1, test 2. The POC used base seed 0 (different streams).
- `manifest.json`: dataset id, requested counts, config sha256, base seed and stream definitions, bank sizes, test views
  (with list hashes), per-split file-list and label hashes, `metadata_sha256` (without wall-clock fields), library
  versions, git commit and a dirty flag.
- Code plus config reproduce the data byte-for-byte in a pinned environment. Verified: two runs of the same command
  gave 110/110 byte-identical images and labels and identical hashes. Across machines, library versions (OpenCV
  JPEG encoder, NumPy FFT) may differ; pin versions and compare the manifest.
- Generated images are never committed. Only config, code, hashes and docs are.

## 3. Implemented pre-scaling changes (approved items 1, 2, 4, 5)

1. **Split-specific, larger smoke-sprite banks** (48 / 16 / 16, seeded per split). Confirmed independent: the smallest
   mean absolute difference between any sprite in one split and any in another is 0.10-0.11 (within a split 0.11-0.15;
   an identical sprite would be 0.00). Texture banks are also per-split.
2. **Plume drift capped and decoupled.** Drift (along-wind lean) is drawn from its own per-archetype range, independent
   of the box-width target, and capped at 2x plume height; width now comes from the cross-section and puff size.
   Before: drift/height P50 1.37, P95 12.5, 40% above 2. After: P50 0.70, P95 1.37, max 1.50, none above 2.
3. **Alternative box conventions recorded** per positive image in `metadata.jsonl` (`box_variants`: tight 0.30/0%,
   default 0.08/2%, haze 0.02/5%, padded 0.08/10%). The labels on disk use `default`.
4. **Bigger texture banks** (16 / 6 / 6) with per-texture palette jitter (brightness, red and blue vs green).

Also fixed while doing this: the archetype, box size and edge decision are now per-image draws, independent of the
retry attempt. A first version of the edge-touch work re-drew them on every retry and silently biased the dataset
(curtain share 11% instead of 36%); the corrected sampler keeps the intended mix (0.227 / 0.204 / 0.215 / 0.354 over
4,000 sampled scenes, intended 0.23 / 0.20 / 0.21 / 0.36) and the box-area marginal (P5/P50/P95 0.127 / 0.324 / 0.771 vs
Boreal 0.121 / 0.338 / 0.815).

## 4. Known limitation of synth_smoke_v1: edge-touching (Option A, approved)

**Decision (frozen before Model A is trained and before any Synth->Real result is seen): accept and document the
mismatch. It is a known synthetic-to-real domain gap of `synth_smoke_v1`, not something to be optimised away.**

| Quantity | Boreal train | synth_smoke_v1 (POC v2) |
|---|---|---|
| Fraction of boxes touching an image edge | **about 0.304** (train-val range 0.30-0.43) | **about 0.804** (97 smoke images) |
| Box area / image, P5 / P50 / P95 | 0.121 / 0.338 / 0.815 | 0.121 / 0.329 / 0.806 |
| Smoke images with 1 box | 99.6% | 100% |
| Background fraction (train) | 0.075 | 0.075 (260 / 3,479) |

The box-area distribution matches closely. Forcing the edge-touch frequency toward Boreal's would have distorted that
well-matched area distribution or required increasingly Boreal-specific generator tuning, so it is deliberately not
done. Options B (shrink column/curtain size priors) and C (joint position-and-size calibration) are **not** implemented
for v1.

Evidence (POC metadata): the target-level touch policy does not control labels (36 of 53 non-touching targets gave a
touching label; the rendered smoke extends a median 0.10 of the image height below its target box, P90 0.29); even the
tight convention (alpha 0.30, no padding) touches 69% of the time; geometry alone forces about 44% of boxes to touch
under the size priors taken from Boreal percentiles. The label-level gate exists in the code but is **off by default**
(`edge_policy.enforce_on_labels: false`); plume alignment was tried and reverted.

### Pre-declared diagnostic (analysis only, fixed now)

After Model A exists, evaluate it on the Boreal test set grouped into **edge-touching** and **non-edge-touching**
ground-truth boxes (a Boreal box touches when it is within 0.2% of the image border, the same definition used for the
statistics above) using the same evaluation code path, and report mAP50, mAP50-95, precision and recall for each group,
plus the false-positive/false-negative breakdown by group. This is a description of the domain gap only. **The result
will not be used to modify, re-tune or regenerate `synth_smoke_v1`**; any later generator change is a new version
(v2) reported as post-hoc. The same grouping may also be applied to Model B for reference.

## 5. Model A training specification (matched to Model B)

Verified from local artifacts and the Ultralytics 8.4.173 source: Model B used `optimizer: auto`, which resolved at run
time to **AdamW, lr0 0.002, betas (0.9, 0.999), warmup bias lr 0.0** (the `auto` rule gives AdamW when
`ceil(N/64) x epochs <= 10,000`, otherwise MuSGD; for 3,479 images that is 5,500). The learning rate in `results.csv`
implies lr0 = 0.002 exactly (epoch 100: 3.98e-5 = 0.002 x 0.0199). Model A has the same train size, so `auto` would give
the same result, but the optimizer is pinned explicitly anyway.

Exact differences between the Model B arguments (`args.yaml`) and the Model A arguments; **everything else is identical**
(`imgsz 640`, `epochs 100`, `batch 32`, `patience 100`, `seed 0`, `deterministic True`, `workers 8`, pretrained
`yolov8s.pt`, `single_cls False` with one class `0 = smoke`, `lrf 0.01`, `weight_decay 0.0005`, `warmup_epochs 3`,
`close_mosaic 10`, `nbs 64`, `amp`, mosaic 1.0, hsv 0.015/0.7/0.4, translate 0.1, scale 0.5, fliplr 0.5, randaugment,
erasing 0.4, `cos_lr False`, `rect False`):

| Argument | Model B (args.yaml, nominal) | Model B (effective at run time) | Model A (explicit) |
|---|---|---|---|
| data | `/workspace/splits/boreal_v1/data.yaml` | same | `/workspace/synth_smoke_v1/data.yaml` |
| optimizer | auto | AdamW | **AdamW** |
| lr0 | 0.01 | 0.002 | **0.002** |
| momentum (AdamW beta1) | 0.937 | 0.9 | **0.9** |
| warmup_bias_lr | 0.1 | 0.0 | **0.0** |
| run name | real_boreal | | synth_smoke_a_v1 |

Planned command (to be run on the pod only after separate approval of cost, GPU, dataset, model, epochs and output location):

```
yolo detect train model=yolov8s.pt data=/workspace/synth_smoke_v1/data.yaml epochs=100 patience=100 imgsz=640 \
  batch=32 seed=0 deterministic=True workers=8 plots=True optimizer=AdamW lr0=0.002 momentum=0.9 \
  warmup_bias_lr=0.0 project=/workspace/runs name=synth_smoke_a_v1 exist_ok=False
```

Checkpoint selection uses the synthetic **val** split (best fitness, like Model B on Boreal val). The checkpoint is
exported as `synth_smoke_best_v1.pt` and evaluated once per cell.

## 6. Pre-declared evaluation

- Same code path for all four cells; `conf=0.001, iou=0.7`.
- Report **mAP50 (primary, more robust to the human-versus-mask box convention gap)**, mAP50-95, precision, recall,
  confusion matrix, and the false-positive rate on `test_all` background images. Emphasis is fixed now and will not
  change after seeing Model A results.
- Synthetic test views stay separate in all reporting: `test_pos` for the primary cross-domain comparison, `test_all`
  for false-positive analysis. The real test has no usable background images, so the empty-scene false-positive rate is
  measurable only on the synthetic side.
- Checkpoint selection uses each model's own-domain val split; Model A never sees Boreal val or test.

## 7. Cost (measured on this VM)

- A/B with identical conditions: the new code costs about 16% more per smoke image than the first POC (mean 4.72 s vs
  4.08 s on a degraded machine; medians equal).
- The VM currently runs about 4x slower than during the first POC (0.22 img/s vs 0.96 img/s; the runs show 82% system
  time, i.e. memory-allocation overhead, not CPU steal). At the healthy speed the 5,304-image dataset is about
  1.7-1.8 hours; at today's speed it would be about 6-7 hours. Re-measure before the full run.
- Storage about 0.8 GB; peak memory about 0.54 GB.
- Slowdown check at freeze preparation (idle machine): a 3M-iteration pure-Python loop takes about 1.35 s against about
  0.3 s expected on a healthy core, with no competing process. Recommendation: restart the VM and re-run that check
  before the full generation; proceed on the current VM only if the restart does not restore normal speed, in which case
  the run needs about 6-7 hours and should be launched in a persistent session (tmux or nohup), because the generator
  is not resumable.

## 8. Freeze and generation plan (each step needs approval)

1. Full test suite passes (41 tests) and POC v2 validates (see below). Done at freeze preparation.
2. Commit the spec with a clean tree; that commit defines the frozen generator and config. Tag name: **`synth-v1`**
   (annotated). The tag is created only after approval; the exact commit hash is reported before tagging.
3. Generate from the tag, in a clean checkout of that tag:

```
python3 -m synthgen.generate_dataset --config synthgen/configs/domain_rand.yaml \
  --out ~/datasets/synth_smoke_v1 --counts train=3219:260,val=774:0,test=701:350 \
  --seed 1001 --dataset-id synth_smoke_v1 --contact-sheet 24
```

   Destination: `~/datasets/synth_smoke_v1` on the local VM (outside the repository; never committed). The generator
   refuses to overwrite an existing `manifest.json`. Expected output: 5,304 images (about 0.8 GB), `data.yaml`,
   `data_test_pos.yaml`, `data_test_all.yaml`, `lists/`, `metadata.jsonl`, `manifest.json`.
4. Run the validator on `data.yaml`, `data_test_pos.yaml` and `data_test_all.yaml`; compare against the reference
   statistics; record the manifest hashes in `docs/datasets/synth_smoke_v1.md`.
5. Upload to the RunPod volume only with separate approval (cost stated first); then train Model A, then evaluate.
6. After that, any generator change is v2 and must be reported as post-hoc.

Generation caveat: the generator is not resumable; a crash or a stopped terminal loses the run. Per-image generation is
deterministic and independent, so a skip-existing resume option would not change any output, but it would be a code
change and, if wanted, should be made and tested **before** the freeze commit.
