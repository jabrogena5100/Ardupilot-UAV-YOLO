# Cross-dataset evaluation protocol (draft v0, to be frozen before any inference)

Purpose: test whether the Level 2 contrast between Model A (synthetic) and Model B (Boreal) holds on an independent drone smoke dataset (FASDD_UAV), using the frozen checkpoints only.
Not in scope: retraining, regenerating synthetic data, modifying checkpoints, rerunning Level 3, any threshold or architecture tuning, pooling with FLAME, the 24-page report.

## Fixed inputs
- Checkpoints: Model A = `synth_smoke_best_v1` seeds 0-2; Model B = `real_boreal_v2_best_seed0..2` (the checkpoints Level 3 used, clip-disjoint split). Secondary, reported separately: `real_boreal_best_v1` seeds 0-2. Hashes: `docs/baseline/baseline_v1.json`; the run aborts on mismatch.
- Code: `eval/run_matrix.py` / `eval/metrics.py` unchanged (same code path as the 2x2); only a new pinned config `eval/datasets_external.yaml` is added. Settings identical to the 2x2: imgsz 640, conf 0.001 for AP, nms_iou 0.7, report_conf 0.25.
- Class mapping: FASDD `smoke` -> class 0; `fire` boxes are dropped (the models have no fire class). Images that contain only fire are not smoke negatives unless the sampling audit (G2) shows smoke is reliably labeled.

## Gates (all must pass before inference; results recorded in `docs/external_eval/gates.md`)
- **G1 License/terms:** the dataset record states terms that permit research evaluation and local use; text copied into `gates.md`. Fail -> stop.
- **G2 Label compatibility:** a pre-declared random sample of 100 smoke-positive and 100 smoke-negative images (seed fixed, drawn from the UAV subset) is inspected; report the rate of visible-but-unlabeled smoke (Wilson 95%) and box-tightness notes. Proposed pass: unlabeled-smoke rate in "negative" images <= 5% (upper bound reported), otherwise negatives are excluded and only box metrics on positives are reported. Inspection is done by the model reading the images and is a screening, not ground truth; the owner should spot-check at least 30.
- **G3 Source overlap:** exact (SHA-256 / MD5) and perceptual (dHash <= 4 bits, as in `eval/boreal_v2_audit.py`) comparison of every candidate image against all Boreal frames (train, val, test, all 4,954). Any hit is removed *before* inference and counted. No candidate removal may use model outputs.
- **G4 Sufficiency:** after exclusions, >= 500 smoke-positive images from >= 20 recoverable source clusters (video id / sequence prefix / near-duplicate groups at dHash <= 4). If clusters cannot be recovered, intervals are labeled as image-level only (anti-conservative) and the claim is weakened accordingly.
- **G5 Type check:** only RGB images; thermal/IR, satellite (RS) and non-UAV (CV) images are excluded by the dataset's own subset and by a documented rule, never by looking at detector output.

## Test-set construction (frozen before inference)
Take FASDD_UAV; apply G3/G5 exclusions; record the file list and SHA-256 of the sorted names in `eval/datasets_external.yaml` (same pin definition as `eval/datasets.yaml`). No image is added or removed after the first inference run. If the dataset ships a test split use it and say so; otherwise the entire filtered UAV subset serves as test, which is legitimate because no model trains on it. Views: `ext_pos` (smoke boxes present), `ext_all` (only if G2 passes for negatives).

## Metrics and statistics
Per checkpoint: P, R, mAP50, mAP50-95, FP/FN at conf 0.25, mean confidence and localisation (same functions as the 2x2). Aggregate over 3 seeds per family as mean +- SD (ddof 1) -- bootstrap CIs (clustered if G4 allows) describe test sampling only, not seed variance. Primary contrast: Model A mean minus Model B mean on mAP50 and recall at conf 0.25, with the seed range of each. Report also each family's in-domain value from `docs/baseline` beside the external value (drop = in-domain minus external) with the caveat that in-domain sets differ.
No significance claim from 3 seeds; no claim that the result transfers to Level 3.

## Interpretation guards (written now so they cannot be chosen after seeing results)
- A beats B, B beats A or no difference are all reported. A large drop for both is a finding about FASDD heterogeneity, not about either training domain.
- Differences in annotation style (box tightness, label completeness) can mimic domain effects; mAP50 is primary, mAP50-95 is secondary and flagged.
- The threshold 0.25 is the Level 2 operating point, not the Level 3 per-arm thresholds; this is stated.
- If G2 or G3 reveals material contamination or label incompatibility, the report is "not evaluable" with the evidence; no workaround is substituted.

## Outputs (separate from the baseline)
`eval/datasets_external.yaml`, `docs/external_eval/{dataset_audit,gates,protocol,results}.md`, raw outputs under `local_data/external_eval/`. The frozen baseline files are never edited; `python3 eval/freeze_baseline.py --verify` must still pass after the external run.

## Next step needing the owner
Download FASDD_UAV (size unknown; license must be read on the Science Data Bank record first, G1). Nothing has been downloaded.
