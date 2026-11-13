# External evaluation: dataset roles, eligibility, and statistics (PROPOSAL v3, 2026-11-11)

Status: **proposal for owner review; nothing frozen, no model run on FireMan or FASDD.** The machine-readable form is `eval/datasets_external.yaml` (status `proposed_not_frozen`). The frozen Level 1-3 baseline (`docs/baseline/`), `docs/external_eval/protocol.md` (draft), the six checkpoints, `eval/run_matrix.py`, `eval/metrics.py` and all finalized datasets are untouched. Evidence: `dataset_audit.md` (FireMan), `readiness.md` (FASDD, sections 2b-2g), `fireman_video_metadata.json` (per-video provenance and scene clusters).

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
- **Independence of the five primary videos:** they were filmed 2023-08-26 between 12:00 and 12:29 within about 2 km (thermal on-screen time and GPS), so they are **one burn event**. By frames they form 5 clusters under the strict rule (no shared frames; hash <= 6 bits and correlation >= 0.9) and **2 clusters** under a looser rule (video pairs where at least 15% of frame pairs correlate >= 0.9 are merged: test/002, train/009, train/010, train/011 against val/002). The effective number of independent frames per video (autocorrelation of the box area) is 3.4, 5.3, 6.3, 3.0 and 4.9, about 23 in total for 19,459 frames (750 sampled). Conclusion: 1 burn event, at most 2-5 scene clusters, about 23 effective frames: **descriptive only, no interval, no test, no replication claim.**

### FASDD_UAV (official test split, 4,181 images)
- Use the **2,149 smoke-positive images** (`smoke_*`, `bothFireAndSmoke_*`) only. The 1,997 empty-label images are **not** verified negatives (about 10% of the screened ones show visible unlabeled smoke; `readiness.md` 2g), so no false-positive rate is reported for FASDD.
- **G5 fails for "UAV only"**: 74 of 100 screened images are fixed tower cameras. FASDD is therefore a mixed-platform external test (tower cameras plus UAV and ground videos), reported as such; the conclusion cannot be stated as "UAV generalisation" from FASDD.
- Scene clusters: dHash64 connected components on the smoke-positive images, **radius 6 primary** (680 clusters, largest 3.2% of images, the most conservative), radius 2 (1,232) and 4 (897) as sensitivity (`eval/fasdd_uav_test_smoke_positive_clusters_v1.json`).
- Platform (by image size, visually validated; `readiness.md` 2i): tower_fixed_camera 718, uav_nadir_1080p 306, ground_or_low_video_720p 1,125 (platform unknown). Platform-specific results are reported for the first two only, labelled "by image size"; the third only descriptively.

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
3. **Unit, aggregation and when statistics may be used.** Per-cluster scores through `eval/external_stats.py::cluster_scores` (one call of `metrics.evaluate` per cluster). Primary statistic: macro-average over clusters that contain smoke (each cluster weight 1). Frame-pooled and video-weighted averages are sensitivity analyses. FireMan primary (1 burn event, 2-5 clusters) is **descriptive only**: per-video values, no interval, no test. Statistics are used only where the cluster gate in `eval/datasets_external.yaml` is met: at least 20 clusters for an interval and the replication reading, at least 8 for an exact sign-flip test (interval labelled wide), fewer = descriptive. Per platform: tower (196 clusters) and 1080p UAV (212) pass; the 720p group is descriptive because its clusters overstate independence.
4. **Intervals and tests** (`eval/external_stats.py`, tested): cluster bootstrap (10,000 draws, fixed seed, percentile 95%), the same resamples for every family so Model B minus Model A is paired; within a replicate a family's score is the mean over its 3 seeds; seed SD reported next to the interval, not inside it. Exact sign-flip test on the cluster-level differences (Monte Carlo above 20 clusters) and the leave-one-cluster-out range.
5. **Pre-declared reading.** For each external set report the sign and interval of Model B minus Model A on mAP50. "The Level 2 pattern replicates" only if the sign matches Level 2 and the interval excludes 0; same sign with an interval containing 0 is "consistent, not established"; opposite sign is reported as such. FireMan primary cannot reach "replicates" (single burn day); a replication statement can rest only on FASDD (mixed platform, cluster gate met) and, if its provenance is later confirmed, FireMan secondary, always with the caveats attached. **Descriptive findings (per-video values, effect sizes, pooled point estimates) are written as descriptive; only a gate-passing interval is called statistical evidence.**
6. **Sensitivity set (fixed now):** FireMan primary alone and, separately, primary plus secondary (explicitly labelled provenance-unverified); keyframe-near frames only (within 2 frames of a keyframe); FASDD cluster radius 2 and 4; FASDD per platform group (by size) and without the 720p group.
7. **Link to monitoring (Level 3).** External perception results say nothing directly about monitoring performance; any statement about Level 3 stays based on the existing simulator runs.

## 5. Gate status (2026-11-11)
| | FireMan | FASDD |
|---|---|---|
| Licence | CC BY 4.0 | CC BY-SA 4.0 |
| Overlap with Boreal | no duplicate at hash or pixel level; provenance resolved by tier (5 verified-2023 videos primary) | no genuine overlap found; owner confirmation of the pair sheets pending |
| Platform / UAV-only | RGB drone video; one camera type in the primary set | **G5 fails for UAV-only** (718 tower, 306 UAV nadir, 1,125 video of unknown platform); accepted as mixed-platform |
| Label quality (G2) | boxes loose, 96-99% interpolated, 9 of 40 screened primary frames partly covered; no negatives | open: loose or phantom boxes in the positives; empty-label images hold unlabeled smoke (about 10%) so they are excluded |
| Manifest / pins | `eval/fireman_frames_{primary,secondary}_v1.json` reproduce exactly from the XML | `eval/fasdd_uav_test_smoke_positive_v1.txt` (2,149) + clusters file |
| Spot-check by owner | not yet | not yet |
| Ready for inference | no (owner review and approval) | no (owner review and approval) |

## 6. Remaining blockers before inference
1. Owner visual review and spot-check of the files listed in `visual_review_summary.md`.
2. Owner approval of `eval/datasets_external.yaml` (status `proposed_not_frozen`) and of the statistics rules in section 4.
3. Materialisation: FireMan primary frames (about 25 MB, 750 files of 640x512) with `eval/external_scripts/fetch_fireman_frames.py` (laptop, not a pod: Zenodo blocked the pod and throttles with HTTP 429), then `eval/external_materialize.py fireman` and `... fasdd` to build the harness views and pins.
4. The inference run itself needs a separate approval; it can run on the laptop CPU (the six checkpoints and the sealed FASDD test images are local or on the volume) or on a pod.

## 7. Unresolved FireMan provenance
- **What is verified:** 5 of 34 RGB videos (dates 2023-08-26) from the on-screen date of their paired thermal videos (`fireman_video_metadata.json` has the evidence per video).
- **What is not:** recording dates of the other 29 videos (50,412 frames). FireMan's README lists 2022-07-06, 2022-08-16, 2023-06-14, 2023-08-26 and the places Ruokolahti, Perho, Kestila, Taivalkoski without a mapping; Boreal's Karkkila burn was 2022-07-06 and Ruokolahti 2022-08-15. If the README order holds, 2022-07-06 is FireMan's Ruokolahti (not Boreal's Ruokolahti day), but this is unproven.
- **Who could become eligible if the authors confirm dates (or a non-overlapping 2022 date):**
  - 5 paired RGB-thermal videos without an on-screen date (13,604 frames, 12,178 with a smoke box): `test/video_003`, `train/video_012`, `train/video_013`, `train/video_015`, `val/video_005`; these are low-resolution (640x512 or 958x7xx) like the primary set, so the most useful to confirm.
  - 24 RGB-only 4K or 1080p videos (36,808 frames, 33,230 with a smoke box): `test/video_001, 004, 005`, `train/video_001` to `008`, `014`, `016` to `024`, `val/video_001, 003, 004`. They belong to 25 scene clusters in total with the others; one cluster holds `test/video_001`, `train/video_003`, `004`, `005`, `006`.
  - Any video confirmed as 2023 (2023-06-14 or 2023-08-26) or dated 2022 on a day and place that do not match Boreal can move to the primary tier by editing `fireman_video_metadata.json` and regenerating the manifests as new versions (v2), never by editing the pinned v1 files.
