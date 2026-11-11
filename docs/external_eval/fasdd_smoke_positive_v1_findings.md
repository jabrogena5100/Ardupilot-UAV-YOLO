# FASDD_UAV smoke-positive: external evaluation, first result

Run 2026-11-11, local CPU (torch 2.14.1+cpu, ultralytics 8.4.210), the six frozen checkpoints (Model A = `synth_smoke_best_v1` seeds 0-2, Model B = `real_boreal_v2_best` seeds 0-2),
2,149 smoke-positive images of the official FASDD_UAV test split (`eval/fasdd_uav_test_smoke_positive_v1.txt`, SHA-256 `41864dcd...f1734`), same `eval/run_matrix.py` -> `eval/metrics.py` path, conf 0.25 / IoU 0.5.
Raw run: `local_data/external_eval/runs/ext_v1_fasdd_smoke_positive` (gitignored; `run_info_before/after.txt`, baseline verifier passed before and after). Full tables: `fasdd_smoke_positive_v1_report.md` / `.json`.
Images came from the sealed extraction on the RunPod volume (`eval/runpod/fasdd_pod_job.py`, MD5 of every image checked against the screened copy) and were copied to the laptop; the pinned list and the 2,149-image count reproduce (`tests/test_external_pins.py` passes).

## Result (mean over 3 seeds, SD over seeds in brackets)
| metric | Model A | Model B | B - A |
|---|---|---|---|
| mAP50 | 0.408 (0.010) | 0.224 (0.034) | **-0.184** |
| mAP50-95 | 0.216 (0.009) | 0.125 (0.018) | -0.091 |
| precision @0.25 | 0.459 | 0.626 | +0.167 |
| recall @0.25 | 0.412 | 0.202 | -0.210 |

By platform (image-size proxy, visually validated in `readiness.md`): 720p ground/low video (1,125 images) mAP50 A 0.697 vs B 0.256 (-0.441); UAV nadir 1080p (306) A 0.437 vs B 0.423 (-0.014, seeds overlap); fixed tower cameras (718) both about 0 (0.011 vs 0.003).
Cluster bootstrap (dHash64 scene clusters, r6 = 680 clusters, paired B - A on per-cluster mAP50, seeds averaged inside each replicate): -0.241, 95% interval [-0.270, -0.213], sign-flip p < 1e-5 (Monte Carlo, 200,000 draws); r4 and r2 give -0.274 and -0.286. The interval covers test-set sampling only, never seed or training uncertainty.

## Reading
**Direction is reversed relative to Level 2 (Boreal test, B - A about +0.39) and to FireMan primary (+0.22): on FASDD, Model A (synthetic) beats Model B (real Boreal).** The gate for an interval is met (680 clusters), but the pre-set replication rule was a gate-passing dataset *replicating the Level 2 direction*; it does not, so this is evidence **against** a simple "real-trained generalizes better" reading, not a replication. Taken with FireMan, the external evidence is mixed and dataset-dependent.

The result is carried almost entirely by the 720p group (-0.44). On true UAV nadir imagery the two families are indistinguishable (-0.014 with seed SD 0.03-0.04), and the tower group is uninformative (both about 0). G5 (UAV-only) fails, so the pooled number mixes platforms and should not be called a UAV result.
Edge-touch diagnostic (recall, GT boxes touching the border): A 0.61-0.63 vs B 0.22-0.36; not touching: A about 0.05, B about 0.03. Both models recover few interior boxes; A's advantage is mostly on border-touching boxes, which is where the `synth_smoke_v1` edge-touch bias (about 0.76 vs Boreal 0.30) could help it. This is a hypothesis to test, not a finding.

## Limitations
Smoke-positive images only: no negatives, so no false-positive rate and no empty-image FP (about 10% of FASDD empty-label images show unlabeled smoke, per the G2 screening). G2 (label quality screening) and the G3 visual check are still owner-pending; the screener was Claude. 720p group provenance is unknown-platform video. The pooled cluster interval is dominated by the 720p group; pooled-pixel (-0.184) and cluster-macro (-0.241) differ because clusters weight scenes equally. Seed SD is small here (0.01-0.03) but three seeds per family remain few.
Tooling note: the first report attempt crashed the 2 GB laptop; `external_stats.sign_flip_test` Monte Carlo allocated a 200,000 x 680 array (about 1 GB). It is now chunked (same definition; 18 stats tests pass); peak RSS is about 200 MB. Spend this step: none (local CPU).
