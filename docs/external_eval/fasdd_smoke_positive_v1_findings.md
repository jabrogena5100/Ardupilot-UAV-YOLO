# FASDD_UAV smoke-positive: external evaluation, first result (descriptive; corrected 2026-11-11 after the final Codex audit)

Run 2026-11-11, local CPU (torch 2.14.1+cpu, ultralytics 8.4.210), the six frozen checkpoints (Model A = `synth_smoke_best_v1` seeds 0-2, Model B = `real_boreal_v2_best` seeds 0-2),
2,149 smoke-positive images of the official FASDD_UAV test split (`eval/fasdd_uav_test_smoke_positive_v1.txt`, file SHA-256 `66e72fa23f85ceb77a9ff448449dd55d7aeb88025d02c2f127a55d45fe911b5c`, as pinned in `eval/datasets_external.yaml`; verified with `sha256sum` on 2026-11-11; an earlier version of this file quoted `41864dcd...`, which is not the hash of the image list; that string is a `WANT` value in `eval/runpod/fasdd_pod_job.py` whose meaning was not re-derived here), same `eval/run_matrix.py` -> `eval/metrics.py` path, conf 0.25 / IoU 0.5.
Raw run: `local_data/external_eval/runs/ext_v1_fasdd_smoke_positive` (gitignored; `run_info_before/after.txt`, baseline verifier passed before and after). Full tables: `fasdd_smoke_positive_v1_report.md` / `.json`, which are recomputed offline from the saved predictions and ground truth (recomputation metadata, not original inference evidence; the inference evidence is the run directory's logs).
Protocol status: `eval/datasets_external.yaml` was never frozen or preregistered (see its `protocol_record`); every number here is exploratory and descriptive.

## Two estimands (never mixed)
- **Pooled AP** (headline): all images of a subset ranked together, mean over the 3 seeds of each family.
- **Cluster-macro AP**: AP computed inside each dHash64 scene cluster, then the unweighted mean over clusters. AP is nonlinear, so this differs from pooled AP in weighting and in the confidence ranking; it is a different quantity, not an uncertainty calculation for the pooled number.

## Result, pooled AP (mean over 3 seeds, SD over seeds in brackets)
| metric | Model A | Model B | B - A |
|---|---|---|---|
| mAP50 | 0.408 (0.010) | 0.224 (0.034) | **-0.184** |
| mAP50-95 | 0.216 (0.009) | 0.125 (0.018) | -0.091 |
| precision @0.25 | 0.459 | 0.626 | +0.167 |
| recall @0.25 | 0.412 | 0.202 | -0.210 |

By platform (image-size proxy; visually checked on screening samples only, see `readiness.md`), pooled mAP50 A / B / B-A:

| group | images | A | B | B - A | reporting status |
|---|---|---|---|---|---|
| 720p ground/low video (platform unknown) | 1,125 (52.3%) | 0.697 | 0.256 | -0.441 | **descriptive only** (config) |
| UAV nadir 1080p | 306 | 0.437 | 0.423 | -0.014 | conditional sensitivity allowed |
| fixed tower cameras | 718 | 0.011 | 0.003 | -0.008 | conditional sensitivity allowed; both near 0 |
| all excluding the 720p group | 1,024 | 0.176 | 0.189 | **+0.012** | conditional sensitivity allowed |

## Cluster-macro AP (conditional sensitivity analysis, not population inference)
dHash64 groups (radius 6: 680 groups overall) are similarity components, not demonstrated independent videos, cameras, scenes or burn events; visually different frames of one evolving scene can fall in different groups, and the inspected 720p and nadir sheets show repeated scene content. Agreement across radii 2, 4 and 6 does not establish independence. The cluster-count gate (>= 20) therefore only says there are enough groups to compute the statistic; it is **not** evidence that the groups are independent, and "gate passed" is not a finding.

Cluster-macro mAP50 (A / B / B - A), r6: all 0.523 / 0.282 / -0.241 (point value only: the subset contains the descriptive-only 720p group, so no interval or test is computed); 720p 0.788 / 0.237 / -0.552 (descriptive only); UAV nadir 0.623 / 0.587 / -0.036; tower 0.044 / 0.015 / -0.029; all excluding 720p 0.344 / 0.311 / -0.033.
Conditional percentile bootstrap intervals for B - A (cluster-macro, groups resampled, seeds averaged inside each replicate; test-set sampling under the grouping assumption only, never seed or training uncertainty): nadir [-0.059, -0.014]; tower [-0.047, -0.014]; all excluding 720p [-0.047, -0.018] (r6; r4 and r2 are in the report). Sign-flip p-values are Monte Carlo estimates (n_mc = 200,000) and are in the report with their floor 1/(n_mc+1) = 5.0e-06; they are conditional on the same assumption and are **not** claimed as population-level significance. The earlier text that quoted the overall interval [-0.270, -0.213], "p < 1e-5" and "gate met (680 clusters)" for the pooled set is **withdrawn**: that interval belongs to the cluster-macro estimand (-0.241), not the pooled -0.184, and it included the descriptive-only group.
Note that the two estimands disagree in sign on the subset excluding 720p (pooled +0.012, cluster-macro -0.033): on that subset the ranking depends on the estimand, which is a further reason not to read it as a clear ordering. No interval for pooled AP was computed (it would need pooled AP recomputed after each paired whole-cluster resample, with defensible clusters).

## Reading
Pooled over the whole mixed-platform test set, Model A scores higher than Model B (-0.184 for B - A), the opposite ordering from the Boreal v2 test (B - A about +0.39) and from FireMan primary (+0.22, one burn event). This difference is carried mostly by the 720p group (52.3% of the images; platform unknown; descriptive only), where A is much higher. On UAV nadir imagery the pooled mean mAP50 differs by only -0.014; **equivalence was not tested** and overlapping seed SDs are not a test. On the cluster-macro estimand the same subset gives -0.036 with a conditional interval below zero, so the data do not support a statement that the two families are tied there either. Tower cameras are uninformative (both near 0). G5 (UAV-only) fails, so the pooled number mixes platforms and must not be called a UAV result.
Defensible conclusion: model rankings vary across the evaluated datasets (Boreal v2, FireMan primary, FASDD smoke-positive). Why (platform mix, annotation style, the `synth_smoke_v1` edge-touch bias, source-scene differences) is a hypothesis, not a finding. Edge-touch diagnostic (recall on GT boxes touching the border): A 0.61-0.63 vs B 0.22-0.36; not touching: A about 0.05, B about 0.03; both models recover few interior boxes.

## Limitations
Smoke-positive images only: no negatives, so no false-positive rate (about 10% of FASDD empty-label images show unlabeled smoke per the G2 screening). G2 (annotation review) and G3 (overlap check) are incomplete: the screener was Claude and the owner's independent review is pending, so external validation remains **provisional**. 720p group provenance is unknown-platform video. Three seeds per family; seed SD is 0.01-0.03 here. Platform labels are by image size, with 161 / 40 / 46 images viewed per group. Loose, partial and questionable boxes affect AP.
Tooling note: the first report attempt crashed the 2 GB laptop; `external_stats.sign_flip_test` Monte Carlo is now chunked (same definition). Spend this step: none (local CPU).
