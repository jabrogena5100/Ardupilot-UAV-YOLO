# External evaluation: dataset roles, FireMan eligibility, and proposed statistics (PROPOSAL, 2026-11-11)

Status: **proposal for owner review; nothing frozen, no model run on FireMan or FASDD.** The frozen Level 1-3 baseline (`docs/baseline/`), `docs/external_eval/protocol.md` (draft) and all finalized datasets are untouched. Evidence: `dataset_audit.md` (FireMan section) and `readiness.md` (FASDD).

## 1. Dataset roles (owner decision, 2026-11-11)
| dataset | role |
|---|---|
| Boreal | real-world training and evaluation (Model B) |
| `synth_smoke_v1` | synthetic training and evaluation (Model A) |
| FASDD_UAV | independent external **test only** (official test split, 4,181 images, already extracted and screened) |
| FireMan-UAV-RGBT (RGB only) | independent external **test only** |
| FLAME2-DT | dropped, not to be downloaded |

External sets validate the existing Level 2 conclusions (do synthetic-trained and Boreal-trained perception differences generalise to other UAV data, and do they matter for monitoring). They do not widen the training scope; no model is trained on them.

## 2. FireMan eligibility (what is known)
- **Pool:** 34 RGB videos, 69,871 frames (changelog says 69,942, not reconciled), 64,867 frames with a smoke box, 32 videos with smoke, 2 smoke-free videos. Thermal data and `Binary.zip` are not used.
- **Overlap with Boreal/FASDD (hash-based, 1,962 sampled frames):** no MD5 match; no frame within 4 bits of Boreal; 1 frame within 4 bits of FASDD (unverified). Frames within 8 bits exist in 13 videos (6 vs Boreal, 12 vs FASDD, some in both) and look like low-texture collisions but are **not pixel-verified**.
- **Tier E1 (no hash hit within 8 bits of either set):** 21 videos, 29,835 frames. **Tier E2 (at least one hit, pending pixel check):** 13 videos, 40,036 frames. E2 is not excluded; it is waiting for verification. If a hit is confirmed as a real duplicate, only that video (or scene cluster) is excluded.
- **Unit of independence:** 30 scene clusters (one cluster of 5 videos). Video is not a safe unit: the official split puts the same stationary scene in train and test.
- **Unresolved:** whether any FireMan burn is the same event as a Boreal event (Ruokolahti shares a name; dates unread; video-to-site map not stated). Needs the Boreal Methods dates (owner) and, ideally, FireMan's site/date per video. Until resolved, report all FireMan results with and without the videos that turn out to belong to a suspect site.
- **Official FireMan test split is not used as the test set:** 5 videos, one of which (`test/video_002`) is 68% of its frames, and its smoke-free video duplicates a training scene. Since nobody trains on FireMan, all eligible clusters are used.

## 3. Annotation and evaluation limitations (to state in any write-up)
1. Boxes are CVAT keyframes plus linear interpolation (96-99% of frames); mid-segment boxes may drift. Sensitivity set: frames within 2 frames of a keyframe.
2. Whole-plume boxes, median area 0.43 of the image, 74% edge-touching, often including sky: annotation style differs from Boreal (about 30% edge-touching); mAP50 is primary, mAP50-95 secondary and flagged. The same applies to FASDD (smoke-only edge-touch 66%).
3. Frames without a smoke box are not verified negatives (example: dense smoke, fire box, no smoke box). False-positive rates are reported only if screening shows the unlabelled-smoke upper bound is at most 5%; otherwise positives only.
4. Mixed cameras and sizes: 3840x2160 (21 videos, 33,575 frames), 640x512 (6, 20,090), 958x7xx (4, 12,973), 1920x1080 (3, 3,233). Report per size group.
5. Strong frame correlation: effective independent observations are in the hundreds, not 64,867.
6. Same-region, same-type burns as Boreal: even without duplicate frames, FireMan is closer to Boreal's world than FASDD is, which favours Model B; say so.
7. Licences: FireMan CC BY 4.0 (attribute), FASDD CC BY-SA 4.0 (no redistribution of derived data).

## 4. Proposed evaluation protocol (to be frozen before any inference)
1. **Test lists.** FireMan: up to 150 evenly spaced frames per eligible video (5,100 frames at most for all 34), list and SHA-256 of sorted names pinned in a new `eval/datasets_external.yaml` entry; labels converted from CVAT XML (`smoke_region` box to class 0; fire and other classes dropped) with a tested converter; frames fetched from the Zenodo zip by range requests, never committed. FASDD: the existing sealed 4,181-image test set, with the smoke-only labels.
2. **Models and harness:** the six frozen checkpoints (Model A seeds 0-2, Model B v2 seeds 0-2), `eval/run_matrix.py` and `eval/metrics.py` unchanged, same settings as the 2x2 (imgsz 640, conf floor 0.001 for AP, 0.25 operating point stated as the Level 2 one).
3. **Unit and aggregation.** Cluster = FireMan scene cluster (30) or FASDD dHash cluster at radius 4 (1,282 over the whole test split). Per cluster: mAP50, precision, recall at 0.25, mean confidence. Primary statistic = macro-average over clusters containing smoke; frame-pooled and video-weighted averages are sensitivity analyses.
4. **Intervals and tests.** Cluster bootstrap (resample clusters with replacement, 10,000 draws, fixed seed, percentile 95%) for each model family and, with the same resamples, for the paired difference Model B minus Model A. Within every replicate, a family's score is the mean over its 3 seeds; seed SD is reported separately and is not part of the interval (as in `protocol.md`). Add an exact sign-flip test on the cluster-level paired differences and a leave-one-cluster-out range. No claim from the 3 seeds alone.
5. **Pre-declared reading.** For each external set, report the sign and interval of Model B minus Model A on mAP50. "The Level 2 pattern replicates" only if the sign matches Level 2 and the interval excludes 0; same sign with an interval containing 0 is "consistent, not established"; opposite sign is reported as such. FireMan and FASDD are reported separately, never pooled (FASDD may reuse open-access sources).
6. **Sensitivity set (fixed now):** Tier E1 only; without the 5-video scene cluster; keyframe-near frames only; per size group; without any video from a suspect Boreal-overlap site once known.
7. **Link to monitoring (Level 3).** External perception results say nothing directly about monitoring performance; any statement about Level 3 stays based on the existing simulator runs.

## 5. Remaining blockers before inference
1. Visual review of the FASDD screening v2 sheets and the FireMan screening sheets (owner spot-check of the 30 marked frames each); decides whether negatives are usable.
2. Pixel verification of the Tier E2 candidates (about 22 Boreal and about 100 FASDD frame pairs): needs one short CPU pod because the Boreal and FASDD images live only on the volume (about $0.02, approval needed).
3. Ruokolahti same-event question: owner pastes Boreal's per-site dates (or the FireMan paper's site table) so videos can be mapped or excluded.
4. Build and test: CVAT-to-YOLO converter, pinned FireMan list, cluster-bootstrap script, external config; then freeze.
5. Download of the FireMan test frames (up to 5,100 frames, about 1.6 GB): Zenodo throttles at roughly 1 frame per second, so about an hour in the background.
6. FASDD: the pending G2/G3/G5 gates in `readiness.md`.
