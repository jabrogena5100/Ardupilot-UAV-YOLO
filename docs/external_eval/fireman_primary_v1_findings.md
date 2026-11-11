# FireMan primary: external evaluation, first result (descriptive)

Run 2026-11-11, local CPU (torch 2.14.1+cpu, ultralytics 8.4.210), six frozen checkpoints (Model A = `synth_smoke_best_v1` seeds 0-2, Model B = `real_boreal_v2_best` seeds 0-2),
750 pinned frames of the 5 date-verified FireMan videos (`eval/fireman_frames_primary_v1.json`), batch 1, conf 0.25 / IoU 0.5 operating point.
Raw run: `local_data/external_eval/runs/ext_v1_fireman_primary` (gitignored; logs, `run_info_before/after.txt`). Baseline verifier passed before and after: nothing frozen was written.
Full tables: `fireman_primary_v1_report.md` / `.json`.

## Reading rules
One burn event, 5 strict clusters. **Descriptive only: no interval, no test, no replication or generalization claim.** FASDD is evaluated and reported separately: `fasdd_smoke_positive_v1_findings.md` (direction reversed, B - A -0.18).

## Result (mean over 3 seeds, SD over seeds in brackets)
| metric | Model A | Model B | B - A |
|---|---|---|---|
| mAP50 | 0.462 (0.173) | 0.683 (0.177) | +0.221 |
| mAP50-95 | 0.247 (0.063) | 0.430 (0.105) | +0.183 |
| precision @0.25 | 0.341 (0.156) | 0.743 (0.091) | +0.402 |
| recall @0.25 | 0.647 (0.185) | 0.683 (0.227) | +0.036 |

Per video, mAP50 B - A: +0.078, +0.633, +0.148, +0.097, +0.266 (the sign is positive in all five videos; recall B - A is negative in two). Frames within 2 of an annotation keyframe (74 frames, less interpolation error): B - A +0.174, same direction.

## Comparison with Level 2
Level 2 (Boreal v2 test) gave B - A about +0.39 mAP50. Here B - A is +0.22: same sign, smaller, and Model A is far better than on Boreal (0.46 vs 0.51 there, but the Model B value drops from 0.90 to 0.68). Consistent in direction with Level 2; it is **not** a replication under the pre-set rule (needs a gate-passing dataset and an interval excluding 0). The largest B - A gap is in precision (Model A over-predicts smoke), while recall is similar; Model A is more confident on its true positives.

## Limitations
Single burn event; effective independent frames about 23; interpolated, loose or partially covering CVAT boxes (27 of 40 screened frames loose, 4 whole-frame); no negative frames, so no false-positive rate; mAP is affected by box looseness. Seed SD is large (0.17-0.18), so the A/B gap is about 1.2 SD of the seed spread. Owner visual review of the screening sheets: **PENDING** (not claimed).
