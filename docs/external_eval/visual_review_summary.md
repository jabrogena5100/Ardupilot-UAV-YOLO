# Visual review summary (2026-11-11): FASDD and FireMan screening sheets

Reviewer: Claude (one model's visual judgement, not ground truth). The owner's independent look at the SPOT-marked frames is still required; do **not** open the `*_claude.csv` files before giving your own verdicts. Image files live under the gitignored `local_data/external_eval/`.

## What was and was not reviewed
| set | reviewed here | not reviewed |
|---|---|---|
| FASDD v2 sample (100 images + 12 near-grayscale extras) | all 10 sheets, every image | - |
| FASDD platform check (30 random positives per image-size group, thumbnails) | all 5 groups | - |
| FireMan primary sample (40 frames of the 5 date-verified videos, from the pinned manifest) | all 4 sheets | - |
| FireMan 100-frame sample (all 34 videos) | sheets 1, 2, 3, 4, 8 (64 frames) | sheets 5, 6, 7, 9 (36 frames, all provenance-unverified videos) |

## FASDD (smoke-positive images are the main set)
- **Platform:** 74 of 100 screened images are fixed mountain-top tower cameras (most with an `hpwren.ucsd.edu` caption, many at night), 10 are UAV nadir views, 16 are low-oblique or ground grass-fire video frames. Image size separates them in the sample without an exception: 3072x2048, 2048x1536, 1600x1200 = tower (161 viewed, 0 exceptions); 1920x1080 = UAV nadir (40 viewed, 0 exceptions); 1280x720 = grass-fire video, platform unknown (46 viewed). Among the 2,149 positives: 718 tower, 306 UAV nadir, 1,125 grass-fire video (52%).
- **Suspicious annotations (positive strata, 40 images):** 17 correct, 5 unsure (tiny boxes on faint horizon specks: sheet 1 #4, #6, #9, #11, #12), 15 loose (smoke box includes flames or large clear areas, sheets 2-3 #19, #22-#36), 3 phantom or loose-phantom (sheet 2 #20, #21 smoke box on a night fire glow; sheet 3 #26 smoke box over a dark area). Several FireAndSmoke frames are near-identical consecutive video frames (#34-#36).
- **Empty-label images (60 screened):** 6 show visible unlabeled smoke (sheet 5 #55-#59 orange-lit smoke layers over a fire glow; sheet 9 #97), 6 more are unsure (#60-#63, #70, #99): 10% (Wilson 95% interval 4.7-20.1%), 20% with the unsure ones. Not verified negatives.
- **Unresolved:** whether the 1280x720 videos are UAV or ground footage; whether tiny horizon boxes are smoke or cloud.

## FireMan primary (the five 2023-08-26 videos)
- All 40 frames show visible smoke with exactly one smoke box; no phantom box, no smoke-free frame (the pinned primary set has no negatives).
- Box style: 27 loose whole-plume boxes that include sky (acceptable), 4 whole-frame boxes (sheet 1 #1, #4, #5, #7), 9 partial-coverage frames where visible smoke or haze extends beyond the box (sheet 2 #17, #22; sheet 3 #26, #28, #31, #32, #34; sheet 4 #37, #40). Fire boxes on val/002 frames are ignored by the converter.
- Interpolation: the box follows the plume plausibly in these samples; drift cannot be judged from stills (96-99% of frames are interpolated).
## FireMan 100-frame sample (secondary videos, 64 frames seen)
- Good tight boxes on `train/video_016`, `019`, `020`, `021`. Questionable: `train/video_017` f114/f277 (large box over faint haze), `train/video_005` f8/f25 (corner box on a faint puff; this video has a smoke box in only 33 of 869 frames), `train/video_008` f815/f1321 (two overlapping boxes: the video has 2,779 boxes in 1,499 frames), `train/video_012` f2478 (dense smoke and a fire box but no smoke box, seen earlier).

## Files for your final review (exact paths)
1. FASDD: `local_data/external_eval/screening_v2/sheet_01.jpg` ... `sheet_10.jpg`; fill the `owner_verdict` column of `screening_manifest.csv` for the 30 rows with `spot_check = 1` (sheet/number: 1 #5, #8, #10, #12; 2 #14, #20, #23, #24; 3 #26, #29, #35; 4 #38, #45, #48; 5 #52, #53, #57, #58; 6 #61, #62, #70; 7 #73, #74, #82, #83; 8 #90, #92, #93, #94; 9 #98). Also `fasdd_platform_check_v1/sizegroup_1280x720.jpg` (is this UAV or ground?) and `sizegroup_1920x1080.jpg`.
2. FASDD overlap (G3): `local_data/external_eval/screening_v2/pairs_01.jpg` ... `pairs_03.jpg` (Boreal vs FASDD closest pairs).
3. FireMan primary: `local_data/external_eval/fireman/screening_primary_v1/sheet_01.jpg` ... `sheet_04.jpg` and `screening_manifest.csv` (12 SPOT frames: #8, #9, #10, #11, #14, #15, #17, #18, #19, #20, #31, #36). Question to answer: are the boxes acceptable as ground truth for an exploratory test?
4. FireMan overlap: `local_data/external_eval/fireman_pixcheck_v1/pairs_01.jpg` ... `pairs_03.jpg` (FireMan frames vs the most similar Boreal and FASDD images).
5. Optional, secondary set only: `local_data/external_eval/fireman/screening_v1/sheet_05.jpg`, `06`, `07`, `09` (not reviewed by me) and the 30 SPOT frames of `screening_manifest.csv` there.
