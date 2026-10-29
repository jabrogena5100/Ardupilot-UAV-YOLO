# Level 2 v2 methodological correction phase (response to Codex REVISE)

Date: 2026-10-29. Status: **complete; evidence package ready for a second independent review.** No model was retrained, no
checkpoint, dataset, split or earlier artifact was modified. One short paid pod (read-only scan, section 1.1) was necessary
because the Boreal train/val file lists exist only on the RunPod volume.
The Codex report itself is not in the repository; the findings below follow the owner's summary of it (split integrity,
Level 3 observation definition, optimizer count, bootstrap claims, reproducibility wording, stale manifest hash, inference
provenance). Corrected text lives in `level2_v2.md`; this file holds the new evidence.

## 1. Boreal split integrity (Priority 1)

### 1.1 What was done
`/workspace/boreal/boreal_yolo` (source of the frozen split; read-only) was scanned on a RunPod pod: all 4,954 images got
MD5, size, pixel dimensions, a 64-bit difference hash (dHash, 9x8 grayscale), and label-line count. Pod `n5kg8sii9fcrp5`
(RTX 4090, $0.89/h, US-IL-1, python:3.12-slim, volume mounted; no CPU pod was available in that data center), scan time
about 20 s; terminated, `list-pods` empty. Billing not yet re-read (estimate in `project-status.md` ledger). The result was
returned through the pod log as gzip+base64, reassembled and checked against the pod's printed hashes:
`boreal_scan.json.gz` sha256 `8a4db032e44c4040872d6b4863af9efefe7b2e5ac3b1f6e48b6f993a3e400e8e` (raw JSON
`4f67ef85c6518bd220e867ec644516ea6c4c7762653cf66352a25c0268625527`). A copy was also written to
`/workspace/integrity/boreal_scan.json.gz` on the volume (new file; nothing existing touched).

Verification that the scan is the frozen split: the sorted file names per split reproduce the three pinned file-list SHA-256
values in the Model B `manifest.json` exactly (train `9ed3ca0d...`, val `c1e3cabf...`, test `d2a610de...`); label-less
images (260 / 0 / 1) and box counts (3,304 / 845 / 713) match the manifest.

Code: `eval/boreal_split_audit.py` (identity normalisation, overlaps, MD5/dHash), `tests/test_boreal_split_audit.py`.
Outputs: `level2_v2_audit/boreal_split_audit.json`, `level2_v2_audit/boreal_scan.json.gz`.

### 1.2 Finding: the "video-disjoint" claim was wrong
The original check keyed a video on the file name **including the site prefix** (`Empty-Images__heinola_DJI_0028`,
`Heinola-Images__heinola_DJI_0028`), so a clip filed under the `Empty-Images` pseudo-site and the same clip under its real
site were different videos. Normalising identity to the clip name (prefix and `_frameN` removed, case-folded):

| pair | shared source clips (original prefixed definition) | shared source clips (normalised) |
|---|---|---|
| train & test | 0 | **3** |
| train & val | 0 | **2** |
| val & test | 0 | 0 |

Videos per split are unchanged (26 / 5 / 5 under both definitions); the overlap exists because `Empty-Images` frames come from
clips that also appear with labelled frames in val/test.

| clip | train frames (all `Empty-Images`, no label file) | other split frames | frame ranges | min frame gap |
|---|---|---|---|---|
| karkkila_dji_0003 | 18 (24-44) | test 26 (0-42) | interleaved | 1 |
| karkkila_dji_0006 | 19 (84-118) | test 100 (0-110) | interleaved | 1 |
| ruokolahti_dji_0085 | 2 (176-177) | test 176 (0-175) | adjacent | 1 |
| heinola_dji_0028 | 111 (0-230) | val 156 (15-340) | interleaved | 1 |
| heinola_dji_0048 | 47 (0-277) | val 15 (217-231) | partly adjacent | 6 |

So **39 train frames share a clip with the test set and 158 with the val set**; 197 of 3,479 train images (5.7%), all
background-only negatives. The frame numbers show the `Empty-Images` frames are the no-smoke segments of the same recordings
(for example, test frames 0-42 of `karkkila_dji_0003` have exactly the gaps filled by train frames 24-39 and 43-44).
Identity is inferred from file names (the site token inside the name matches in every case); it is not confirmed from pixels.

### 1.3 Content duplicates
- Exact: **0** MD5 duplicate groups inside or across splits (4,954 files).
- Perceptual (64-bit dHash, Hamming <= 4): 0 identical hashes across splits; 29 near pairs across train/test, **all between
  different clips** (e.g. `evoDJI_0001` train frames versus `evoDJI_0007` test frames of the same Evo site). None between frames of
  the same clip. dHash on a 9x8 thumbnail is coarse (it flags similar scenes, and cannot rule out re-encoded or shifted
  near-duplicates), so this is a screen, not a proof of absence.

### 1.4 Impact on Level 2 results (sensitivity, not a fix)
The test set is frozen and was not changed. As a reporting view only, `boreal_test` was split into the 3 affected videos
(302 images) and the 2 videos with no train overlap (`evoDJI_0007`, `karkkila_dji_0004`; 399 images), and the saved predictions
were re-scored with the unchanged `eval.metrics.evaluate` (`eval/boreal_leak_sensitivity.py`):

| model | all (701) mAP50 / R@0.25 | affected (302) | unaffected (399) | affected - unaffected mAP50 |
|---|---|---|---|---|
| A seed 0 | 0.429 / 0.572 | 0.655 / 0.656 | 0.352 / 0.511 | +0.302 |
| A seed 1 | 0.384 / 0.496 | 0.555 / 0.520 | 0.341 / 0.479 | +0.214 |
| A seed 2 | 0.313 / 0.488 | 0.422 / 0.556 | 0.266 / 0.438 | +0.156 |
| B seed 0 | 0.951 / 0.945 | 0.971 / 0.947 | 0.937 / 0.944 | +0.034 |
| B seed 1 | 0.956 / 0.945 | 0.944 / 0.937 | 0.969 / 0.951 | -0.026 |
| B seed 2 | 0.953 / 0.950 | 0.947 / 0.940 | 0.962 / 0.956 | -0.015 |

Reading: Model B's difference changes sign across seeds and is small (-0.026 to +0.034), so there is no sign that the overlap
inflates Real -> Real; recall at 0.25 differs by at most 0.012. Model A never saw Boreal, so its large affected-minus-unaffected
gap is video difficulty, not leakage (the affected videos are easier for the synthetic model). Limits: two subsets of 2-3
videos, confounded by video; the training negatives are background-only; this bounds, but does not prove away, the effect. The
Model B headline (0.954) remains valid only as a number on this frozen split, and the audit result is reported next to it.
Checkpoint selection used a val split that shares 2 clips with train negatives (the val curve is the noisier one already noted).

### 1.5 What stays unresolved
Pixel-level confirmation that the `Empty-Images` frames are the same recordings as the val/test frames (only names, frame
numbering and adjacency are used); near-duplicate detection beyond dHash (no learned embeddings); the origin of the
`Empty-Images` folder in the source dataset. A corrected, version-2 split (clip-level, prefix-normalised) would fix this, but a
change of the frozen test or retraining Model B is a decision for the owner and was not done.
`boreal_empty_train_diag` (256 frames) is not "train-only clips": its `karkkila_dji_0003/0006` and `ruokolahti_dji_0085` frames
(39) share clips with test; it remains a Model-A-only probe.

## 2. Level 3 observation contract (Priority 2)
`level3_observation_contract.md`: proposal only. Defines the cell observation, box-to-cell rule, smoke-presence truth,
temporal dependence, the cell-level sensitivity/specificity to measure and the held-out data needed. It states explicitly that
Boreal cannot give cell-level truth, that box-level recall and image-level FP rates are not substitutes, and that no
log-odds update is implemented (`mapping/probability_map.py` is still a stub; `PriorityPlanner.score()` unchanged).

## 3. Report corrections (Priority 3)
1. **Optimizer updates vs minibatches.** The 10,900 in `level2_v2.md` (and `synth_smoke_v1_spec.md` line 25) is the number of
   **minibatches** (109 x 100). With `batch: 32`, `nbs: 64` (all six `args.yaml`), Ultralytics 8.4.173 accumulates gradients over
   `max(round(nbs/batch), 1) = 2` batches (ramped 1 -> 2 during the 327-iteration warmup), so a replay of the trainer's
   accumulate logic gives **5,532 optimizer updates per run**. Source: `ultralytics/engine/trainer.py` lines 321, 521, 598
   (read from the 8.4.173 wheel, not run). The count is derived, not logged; it is identical for all six runs, so comparability
   holds. The spec file has uncommitted owner edits and was left untouched; it carries the same wording and needs the same fix.
2. **Bootstrap claims.** (a) The claim that seed variability "dominates" test-set sampling was based on image-level
   intervals. Synthetic test scenes share a texture bank of 6 slots (and 16 puff sprites) per split, so they are not independent.
   Resampling the 6 texture slots (`eval/synth_cluster_bootstrap.py`, `synth_texture_cluster_bootstrap.json`) gives Real -> Synth
   mAP50 95% intervals B0 0.753 [0.677, 0.857], B1 0.656 [0.500, 0.874], B2 0.869 [0.810, 0.955], versus image-level 0.722-0.783,
   0.618-0.696, 0.845-0.892: 2-5 times wider, comparable to the 0.656-0.869 seed spread. For Synth -> Real the video-cluster
   intervals (5 clusters) are far wider than the seed SD. The "dominates" statement is withdrawn; with 6 or 5 clusters neither
   source of uncertainty can be said to dominate. (b) Model A on synthetic stays at the metric ceiling in both resamplings.
   (c) Synthetic image-level bootstrap intervals are optimistic for the same shared-bank reason.
3. **Reproducibility wording.** What was independently verified in this phase: (i) a from-scratch offline recompute
   from the saved predictions and ground truth reproduces all 1,602 numeric/structural values in the committed `cells/*.json`
   exactly (including the seeded bootstrap), and agrees with the pod-computed `pod_cells/` to <= 2.1e-6 on localisation fields and
   exactly elsewhere; (ii) the Boreal pins; (iii) the backup hashes (below). Not verified: bit-reproducibility of training or
   inference (different hosts and drivers), the exact inference source (section 4), the checkpoints' training logs beyond the
   recorded files. The sentence "every number is recomputable with no GPU" is narrowed to "the metrics and bootstraps are
   recomputable from the saved predictions/ground truth in the local backup (not in git)".
4. **Stale backup manifest hash.** `local_data/backups/level2_v2_seeds/SHA256SUMS` listed its own hash
   (`ec1c1fa0...`) and could never verify. The old file is preserved as `SHA256SUMS.stale_selfhash_20261029`
   (sha256 `3b577b43ddf38e768729010c2db12fb182dc2d9c6a974207fd52a0832d481ce6`); the new `SHA256SUMS` drops only the self-entry
   (sha256 `1e891b8626725c65abf57ad96636b51253b2e532368222dd9e27e08ddebb08e2`) and `sha256sum -c` passes for all 232 files.
5. **Inference-source provenance limitation.** The 24 pod-computed cells record `git.commit: null` and ran from
   `/workspace/repo_v2`, a non-git copy of the tree. The exact version of `predict()` that produced the saved predictions is
   therefore **not recorded** (the tree state at upload time cannot be recovered from the cells; the offline cells record
   `facba8f` with `eval_sim_synthgen_dirty: true`, which concerns only a docstring in `synthgen/generate_dataset.py`).
   Evidence that the metric code is equivalent: offline recompute matches the pod metrics (above). Evidence for the inference code
   itself: Model B seed 0 reproduces the one-off Ultralytics numbers within tolerance, nothing stronger. Future pod runs must record
   a tree hash or commit. Level 2 v1 cells have the same limitation.
6. **Prior documents.** `real_boreal_v1.md` and `models/MODELS.md` state the split is "verified video-disjoint"; correction
   notes pointing here were added. The earlier statements are left in place as history.

## 4. Evidence package for the next review
In git (`docs/experiments/level2_v2_audit/`): `boreal_scan.json.gz`, `boreal_split_audit.json`, `boreal_leak_sensitivity.json`,
`synth_texture_cluster_bootstrap.json`, `synth_test_texture_slot.json` (image -> texture slot, from the local `metadata.jsonl`).
Code and tests: `eval/boreal_split_audit.py`, `eval/boreal_leak_sensitivity.py`, `eval/synth_cluster_bootstrap.py`,
`tests/test_boreal_split_audit.py`. Local only: `local_data/integrity/boreal_scan.json.gz`, backup `SHA256SUMS*`.
Recompute commands:

    python3 -m eval.boreal_split_audit docs/experiments/level2_v2_audit/boreal_scan.json.gz
    python3 -m eval.boreal_leak_sensitivity --preds P --gts G --affected karkkila_dji_0003,karkkila_dji_0006,ruokolahti_dji_0085
    python3 -m eval.synth_cluster_bootstrap --preds P --gts G --slots docs/experiments/level2_v2_audit/synth_test_texture_slot.json
    (P, G = local_data/backups/level2_v2_seeds/level2_v2/{predictions,gts})

Questions for the reviewer: is the clip-level overlap with Model B's 197 background training frames material enough to require a
version-2 split and retraining, or sufficient to report with the sensitivity above; is cell-level truth for Boreal (impossible
today) acceptable to omit, restricting real-domain Level 3 claims to Model A negatives plus synthetic cell-level calibration.
