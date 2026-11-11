# External evaluation: dataset roles, eligibility, and statistics (PROPOSAL v2, 2026-11-11)

Status: **proposal for owner review; nothing frozen, no model run on FireMan or FASDD.** The frozen Level 1-3 baseline (`docs/baseline/`), `docs/external_eval/protocol.md` (draft), the six checkpoints, `eval/run_matrix.py`, `eval/metrics.py` and all finalized datasets are untouched. Evidence: `dataset_audit.md` (FireMan), `readiness.md` (FASDD, sections 2b-2g), `fireman_video_metadata.json` (per-video provenance and scene clusters).

## 1. Dataset roles (owner decision, 2026-11-11)
| dataset | role |
|---|---|
| Boreal | real-world training and evaluation (Model B) |
| `synth_smoke_v1` | synthetic training and evaluation (Model A) |
| FASDD_UAV | independent external **test only** |
| FireMan-UAV-RGBT (RGB) | independent external **test only** |
| FLAME2-DT | dropped, not to be downloaded |

External sets validate the Level 2 conclusions and widen the claim from "Boreal-like data" to other UAV-like data. They do not widen the training scope. FASDD and FireMan are always separate evaluation datasets and are never pooled.

## 2. Eligibility

### FireMan (RGB, 34 videos, 69,871 frames)
Two provenance tiers (`fireman_video_metadata.json`):
- **Primary (`verified_2023`): 5 videos, 19,459 frames** (`test/002`, `val/002`, `train/009`, `train/010`, `train/011`). Their paired thermal videos carry an on-screen date of 2023-08-26 (Matrice 300 RTK, about N65.46-65.48 E28.41-28.44). Boreal was recorded only in 2022, so no overlap with Boreal is possible. All five are one burn day in one area, so they are 5 videos but about one site; they are used as the primary FireMan set with that stated limit.
- **Secondary (`unverified`): 29 videos, 50,412 frames.** No dating evidence; FireMan lists two 2022 dates (2022-07-06, 2022-08-16) that fall on or one day after Boreal's Karkkila (2022-07-06) and Ruokolahti (2022-08-15) burns, so a shared campaign cannot be excluded. Excluded from the primary evaluation and reported separately. They move to primary only if the FireMan authors (or their paper) confirm a 2023 or non-overlapping 2022 date per video.
- Overlap (1,962 sampled frames, all-vs-all against 4,954 Boreal and 4,181 FASDD images): no MD5 match, no frame within 4 bits of Boreal, and no pixel-level duplicate (maximum 32 px correlation 0.968, duplicates score 0.99+; the closest pairs are different scenes). Details: `dataset_audit.md`.
- Scene clusters: 30 (hash <= 6 bits and correlation >= 0.9), one of them with 5 videos; the 5 primary videos are 5 different clusters but share a burn day.

### FASDD_UAV (official test split, 4,181 images)
- Use the **2,149 smoke-positive images** (`smoke_*`, `bothFireAndSmoke_*`) only. The 1,997 empty-label images are **not** verified negatives (about 10% of the screened ones show visible unlabeled smoke; `readiness.md` 2g), so no false-positive rate is reported for FASDD.
- **G5 fails for "UAV only"**: 74 of 100 screened images are fixed tower cameras. FASDD is therefore a mixed-platform external test (tower cameras plus UAV and ground videos), reported as such; the conclusion cannot be stated as "UAV generalisation" from FASDD.
- Scene clusters: dHash radius 4 on the 4,181 test images, restricted to the smoke-positive images; radius 2 and 6 as sensitivity.

## 3. Limitations to state in any write-up
1. FireMan boxes are CVAT keyframes plus linear interpolation (96-99% of frames); mid-segment boxes can drift. Sensitivity set: frames within 2 frames of a keyframe; videos without any keyframe change (a single static box) are reported separately.
2. Both datasets use whole-plume boxes (FireMan median area 0.43 of the image, 74% edge-touching; FASDD smoke-only edge-touching 66%, many tiny distant-plume boxes on tower-camera frames), unlike Boreal (about 30% edge-touching). mAP50 is primary; mAP50-95 secondary and flagged.
3. Frames without a smoke box are not verified negatives in either dataset (FireMan example: dense smoke and a fire box but no smoke box; FASDD: orange-lit smoke layers at night labelled fire only).
4. FireMan primary = one burn day, one camera type (640x512 H20T-registered RGB), so it tests a different camera and place, not the breadth of conditions.
5. Strong frame correlation: the effective number of independent observations per FireMan video is a median of about 5 (range 3-180); FASDD tower-camera frames repeat the same scene many times.
6. FireMan is boreal controlled burns like Boreal (same type of scene, different drone and year), which favours Model B; FASDD is a different mix (tower cameras, grass fires, forest fires).
7. Licences: FireMan CC BY 4.0 (attribution), FASDD CC BY-SA 4.0 (no redistribution of derived data).

## 4. Evaluation protocol (to be frozen before any inference)
1. **Test lists.** FireMan primary: up to 150 evenly spaced frames per video (`eval/fireman_prep.py`, rule documented in its docstring), pinned as a manifest with the SHA-256 of the sorted frame names (`python3 eval/fireman_prep.py manifest --xml-dir local_data/external_eval/fireman/annotations --per-video 150 --tiers verified_2023 --out eval/fireman_frames_primary_v1.json`, then `verify`). FireMan secondary: the same command with `--tiers unverified`. Labels: `convert` (smoke_region to class 0; fire and other rectangles dropped; at most 2 px of overshoot clipped and counted, anything larger raises). FASDD: the sealed test set with the smoke-only labels, restricted to the 2,149 smoke-positive images.
2. **Models and harness:** six frozen checkpoints (Model A seeds 0-2, Model B v2 seeds 0-2); `eval/run_matrix.py` and `eval/metrics.py` unchanged; same settings as the 2x2 (imgsz 640, conf floor 0.001 for AP, 0.25 operating point stated as the Level 2 one).
3. **Unit and aggregation.** Per-cluster scores through `eval/external_stats.py::cluster_scores` (one call of `metrics.evaluate` per cluster). Primary statistic: macro-average over clusters that contain smoke (each cluster weight 1). Frame-pooled and video-weighted averages are sensitivity analyses. FireMan primary has 5 clusters and one burn day: it is reported descriptively per video with the bootstrap shown for transparency, **no confirmatory claim from it alone**.
4. **Intervals and tests** (`eval/external_stats.py`, tested): cluster bootstrap (10,000 draws, fixed seed, percentile 95%), the same resamples for every family so Model B minus Model A is paired; within a replicate a family's score is the mean over its 3 seeds; seed SD reported next to the interval, not inside it. Exact sign-flip test on the cluster-level differences (Monte Carlo above 20 clusters) and the leave-one-cluster-out range.
5. **Pre-declared reading.** For each external set report the sign and interval of Model B minus Model A on mAP50. "The Level 2 pattern replicates" only if the sign matches Level 2 and the interval excludes 0; same sign with an interval containing 0 is "consistent, not established"; opposite sign is reported as such. FireMan primary alone cannot reach "replicates" because of its single-site design; the claim rests on FASDD (mixed platform) and FireMan secondary read together, with each caveat attached.
6. **Sensitivity set (fixed now):** FireMan primary only; FireMan primary plus secondary; without the 5-video scene cluster; keyframe-near frames only; per image size; FASDD UAV-looking images only (identified from the screening labels, exploratory); radius 2 and 6 clusters for FASDD.
7. **Link to monitoring (Level 3).** External perception results say nothing directly about monitoring performance; any statement about Level 3 stays based on the existing simulator runs.

## 5. Gate status (2026-11-11)
See `readiness.md` for the FASDD gates (G0, G0b, G1, G4, split consistency pass; G2 open with a negative preliminary finding; G3 no genuine overlap found, owner confirmation pending; G5 fails for UAV-only). FireMan: hash and pixel-level overlap with Boreal and FASDD found no duplicate; per-video dates verified for 5 of 34 videos (those form the primary set); manifests pinned (`eval/fireman_frames_primary_v1.json`, `..._secondary_v1.json`); screening sample (100 frames) generated and not yet reviewed by the owner.

## 6. Remaining blockers before inference
1. Owner visual review: FASDD `screening_v2/` (30 SPOT images) and `pairs_*` sheets; FireMan `screening_v1/` (30 SPOT frames) and the pixel-check pair sheets.
2. Decide whether FASDD may be used as a "mixed-platform" test given the G5 result, and whether FireMan primary (one burn day) is enough to be worth running.
3. Authors' confirmation of FireMan per-video dates (or another dating source) to enlarge the primary set.
4. Pin and verify the manifests, convert the labels, freeze `eval/datasets_external.yaml` (not created yet), download the FireMan frames of the chosen manifests by range requests (Zenodo throttles with HTTP 429; the RunPod pod IP was effectively blocked, so download from a machine that is not throttled).
5. The inference run itself needs a separate approval.
