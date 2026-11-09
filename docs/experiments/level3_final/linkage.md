# Level 2 to Level 3 linkage (generated)

Level 2 columns: each checkpoint on the **synthetic positive test set** (`synth_test_pos`, 701 frames), operating point conf 0.25, IoU 0.5, from `docs/experiments/boreal_v2/results/report.json` (Model B = the Boreal v2 clip-disjoint retrain used in Level 3, not the older v1 checkpoints). Level 3 columns: the same checkpoint flown in 8 scenarios at its own **per-arm threshold** `tau` (A: 0.30-0.45, B: 0.01-0.02), so Level 2 (0.25) and Level 3 operating points differ. Time columns are lower bounds (right-censored at 600 s).

| arm | Level 3 tau | L2 mAP50 | L2 recall | L2 precision | L2 FP/img | L3 first true det (s, lb) | L3 ign-cell visit (s, lb) | ign-cell visit censored | L3 false det obs/run | L3 distinct false cells/run | L3 coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 0.45 | 0.995 | 1.000 | 1.000 | 0.000 | 169 | 294 | 1/8 | 0.2 | 0.1 | 0.184 |
| A1 | 0.3 | 0.995 | 1.000 | 1.000 | 0.000 | 183 | 276 | 0/8 | 0.5 | 0.5 | 0.180 |
| A2 | 0.35 | 0.995 | 1.000 | 0.999 | 0.001 | 180 | 234 | 0/8 | 0.1 | 0.1 | 0.185 |
| B2s0 | 0.01 | 0.587 | 0.495 | 0.853 | 0.086 | 283 | 496 | 4/8 | 48.8 | 5.5 | 0.149 |
| B2s1 | 0.01 | 0.522 | 0.405 | 0.816 | 0.091 | 350 | 496 | 5/8 | 81.4 | 7.8 | 0.140 |
| B2s2 | 0.02 | 0.726 | 0.589 | 0.939 | 0.039 | 257 | 471 | 5/8 | 36.9 | 6.0 | 0.154 |

Six checkpoints (3 per family) cannot support a regression of Level 3 on Level 2 quality; this table places the two sets of numbers side by side and makes no dose-response claim. Level 2 uses conf 0.25; Level 3 used per-arm thresholds, so the operating points differ.
