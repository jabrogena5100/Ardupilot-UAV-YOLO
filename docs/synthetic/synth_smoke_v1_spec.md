# synth_smoke_v1: specification (candidate, NOT yet frozen)

Condition name: **`Boreal-prior-matched`**. Boreal train/val statistics informed the synthetic box-geometry priors
(`docs/synthetic/boreal_visual_analysis.md`); Boreal **test** was never used. No Boreal pixel is used anywhere.

Status (2026-10-05): approved Phase 5 changes implemented and validated on a small POC (`~/synth_poc_v2`, 110 images,
not committed). **Not frozen**: freezing is a separate step awaiting approval (see "Freeze procedure"). **Nothing at full
scale has been generated; Model A has not been trained.**

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

## 4. Edge-touching: NOT corrected (open item)

Target: Boreal train 30.4% of boxes touch an image edge (train-val range 0.30-0.43; per-scene-type rates unknown).

| Set (smoke images) | n | edge-touch | area P5 / P50 / P95 | w50 | h50 | cx50 | cy50 |
|---|---|---|---|---|---|---|---|
| Boreal train (reference) | 3,219 | 0.304 | 0.121 / 0.338 / 0.815 | 0.62 | 0.62 | 0.58 | 0.40 |
| Before: POC v1 | 167 | 0.814 | 0.133 / 0.336 / 0.776 | 0.56 | 0.67 | 0.58 | 0.43 |
| After: POC v2 (final config) | 97 | 0.804 | 0.121 / 0.329 / 0.806 | 0.56 | 0.77 | 0.59 | 0.50 |

What was tried and measured (POC metadata, no extra renders unless stated):
- **Target-level policy** (a box touches only when geometry forces it): the intended decision rate is 0.44-0.46, but of 53
  scenes whose target did not touch, 36 still produced a touching label. The rendered smoke extends well past its target
  box (median 0.10 of the image height below the target bottom, P90 0.29), so target placement does not control the label.
- **Label-level gate** (calibrate the plume, reject scenes whose predicted label disagrees, before any expensive
  rendering): label touch fell only to 0.69, fell back on 23% of images after 10 attempts each, and cost about 4 attempts
  per positive. Kept in the code, **off by default** (`edge_policy.enforce_on_labels: false`).
- **Plume alignment** (shift the smoke base so the rendered box centre matches the target; tested on 30 scenes):
  centre error roughly halved with vertical moves but some plumes collapsed (area ratio P10 0.13); horizontal-only gave
  little. **Reverted**; not in the code.
- **Label convention** does not rescue it: even the tight convention (0.30, no padding) touches 69% (default 80%).
- **Margins and area scaling** only trade edge-touch against the box-area median (it fell to 0.23-0.28 when touching was
  pushed down). Geometry alone forces about 44% of boxes to touch because the column and curtain size priors (taken from
  Boreal percentiles) are large relative to the frame.

Options, none applied (all need your approval):
- A. **Accept and document** (recommended for v1): label-touch 0.80 vs Boreal 0.30. Effect is on box regression near
  frame borders; it can be examined afterwards by evaluating Model A on Boreal test boxes stratified by touching/not
  touching (analysis only, no tuning).
- B. Shrink the column/curtain size priors until labels reach about 0.30-0.45. This moves the box-area marginal away from
  Boreal's (a different mismatch).
- C. A proper joint position-and-size calibration in image space. Two simple attempts failed; a robust version is a real
  piece of work with uncertain payoff.

## 5. Model A training specification (to match Model B)

Verified from local artifacts and the installed-version source for Ultralytics 8.4.173: Model B used `optimizer: auto`,
which resolved to **AdamW, lr0 = 0.002, betas (0.9, 0.999)** (the `auto` rule gives AdamW when
`ceil(N/64) x epochs <= 10,000`, otherwise MuSGD; for 3,479 images that is 5,500 iterations). Evidence: the learning rate
in `results.csv` implies lr0 = 0.002 exactly (epoch 100: 3.98e-5 = 0.002 x 0.0199). Training log with the optimizer line
is on the volume only and was not needed.

Pin for Model A (explicit, not `auto`): `model=yolov8s.pt imgsz=640 batch=32 epochs=100 patience=100 seed=0
deterministic=True workers=8 optimizer=AdamW lr0=0.002 momentum=0.9 warmup_bias_lr=0.0 weight_decay=0.0005 lrf=0.01
warmup_epochs=3 close_mosaic=10` with every other argument at the Ultralytics 8.4.173 defaults recorded in
`~/backups/real_boreal_v1/.../args.yaml` (mosaic 1.0, hsv 0.015/0.7/0.4, translate 0.1, scale 0.5, fliplr 0.5,
randaugment, erasing 0.4, amp). Explicit `AdamW` takes `lr0` and `momentum` from the arguments, so they must be set as above.

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

## 8. Freeze procedure (needs approval)

1. Decide the edge-touch option above.
2. Full tests pass; small POC validates; contact sheet reviewed.
3. Commit with a clean tree, tag `synth-v1`, record the commit and config hash.
4. Generate the full dataset from that tag; run the validator; record the manifest hashes in a short
   `docs/datasets/synth_smoke_v1.md`; only then evaluate Model A on Boreal test.
5. After that, any generator change is v2 and must be reported as post-hoc.
