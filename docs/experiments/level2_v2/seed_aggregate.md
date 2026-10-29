# Level 2: seed aggregation

Per-seed values come from independently trained checkpoints (seed 0 = level2_v1, seeds 1-2 = level2_v2) evaluated through the same harness. **SD is the sample standard deviation across seeds (n = number of seeds; with n=3 it is itself very uncertain). Bootstrap intervals below are a different quantity and are not seed uncertainty.**

## mAP50

| cell | seed 0 | seed 1 | seed 2 | mean | SD | min-max |
|---|---|---|---|---|---|---|
| Synth -> Synth | 0.995 | 0.995 | 0.995 | 0.995 | 0.000 | 0.995-0.995 |
| Synth -> Real | 0.429 | 0.384 | 0.313 | 0.375 | 0.059 | 0.313-0.429 |
| Real -> Synth | 0.753 | 0.656 | 0.869 | 0.759 | 0.107 | 0.656-0.869 |
| Real -> Real | 0.951 | 0.956 | 0.953 | 0.954 | 0.003 | 0.951-0.956 |

## mAP50-95

| cell | seed 0 | seed 1 | seed 2 | mean | SD | min-max |
|---|---|---|---|---|---|---|
| Synth -> Synth | 0.985 | 0.987 | 0.986 | 0.986 | 0.001 | 0.985-0.987 |
| Synth -> Real | 0.216 | 0.158 | 0.116 | 0.164 | 0.050 | 0.116-0.216 |
| Real -> Synth | 0.389 | 0.316 | 0.467 | 0.390 | 0.076 | 0.316-0.467 |
| Real -> Real | 0.673 | 0.672 | 0.679 | 0.675 | 0.004 | 0.672-0.679 |

## recall@0.25

| cell | seed 0 | seed 1 | seed 2 | mean | SD | min-max |
|---|---|---|---|---|---|---|
| Synth -> Synth | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000-1.000 |
| Synth -> Real | 0.572 | 0.496 | 0.488 | 0.519 | 0.046 | 0.488-0.572 |
| Real -> Synth | 0.568 | 0.565 | 0.749 | 0.627 | 0.105 | 0.565-0.749 |
| Real -> Real | 0.945 | 0.945 | 0.950 | 0.947 | 0.002 | 0.945-0.950 |

## FP/img@0.25

| cell | seed 0 | seed 1 | seed 2 | mean | SD | min-max |
|---|---|---|---|---|---|---|
| Synth -> Synth | 0.000 | 0.000 | 0.001 | 0.000 | 0.001 | 0.000-0.001 |
| Synth -> Real | 0.424 | 0.371 | 0.514 | 0.436 | 0.072 | 0.371-0.514 |
| Real -> Synth | 0.063 | 0.160 | 0.057 | 0.093 | 0.058 | 0.057-0.160 |
| Real -> Real | 0.121 | 0.040 | 0.050 | 0.070 | 0.044 | 0.040-0.121 |

## Cross-domain change (cross-domain minus in-domain, same family and seed)

| model family | metric | per-seed change | mean | SD |
|---|---|---|---|---|
| A (synthetic-trained): synth_test_pos -> boreal_test | mAP50 | -0.566, -0.611, -0.682 | -0.620 | 0.059 |
| A (synthetic-trained): synth_test_pos -> boreal_test | mAP50-95 | -0.769, -0.829, -0.869 | -0.822 | 0.050 |
| A (synthetic-trained): synth_test_pos -> boreal_test | recall@0.25 | -0.428, -0.504, -0.512 | -0.481 | 0.046 |
| A (synthetic-trained): synth_test_pos -> boreal_test | FP/img@0.25 | +0.424, +0.371, +0.512 | +0.436 | 0.071 |
| B (real-trained): boreal_test -> synth_test_pos | mAP50 | -0.198, -0.301, -0.084 | -0.194 | 0.108 |
| B (real-trained): boreal_test -> synth_test_pos | mAP50-95 | -0.284, -0.356, -0.212 | -0.284 | 0.072 |
| B (real-trained): boreal_test -> synth_test_pos | recall@0.25 | -0.378, -0.380, -0.201 | -0.320 | 0.103 |
| B (real-trained): boreal_test -> synth_test_pos | FP/img@0.25 | -0.058, +0.120, +0.007 | +0.023 | 0.090 |

## Bootstrap 95% CIs per seed (test-set sampling only, fixed checkpoint)

Image-level resampling treats frames as independent. Boreal frames from one video are correlated, so the video-cluster row (5 test videos) is the more honest one for Boreal; the percentile method is crude with so few clusters.

| cell | seed | resampling | mAP50 | mAP50-95 | recall@0.25 |
|---|---|---|---|---|---|
| Synth -> Synth | 0 | images | [0.995, 0.995] | [0.982, 0.988] | [1.000, 1.000] |
| Synth -> Synth | 1 | images | [0.995, 0.995] | [0.984, 0.989] | [1.000, 1.000] |
| Synth -> Synth | 2 | images | [0.995, 0.995] | [0.982, 0.988] | [1.000, 1.000] |
| Synth -> Real | 0 | images | [0.388, 0.474] | [0.196, 0.242] | [0.537, 0.613] |
| Synth -> Real | 0 | videos | [0.206, 0.969] | [0.101, 0.468] | [0.327, 0.934] |
| Synth -> Real | 1 | images | [0.344, 0.429] | [0.140, 0.181] | [0.461, 0.535] |
| Synth -> Real | 1 | videos | [0.181, 0.885] | [0.076, 0.282] | [0.249, 0.782] |
| Synth -> Real | 2 | images | [0.280, 0.355] | [0.102, 0.134] | [0.452, 0.526] |
| Synth -> Real | 2 | videos | [0.094, 0.769] | [0.038, 0.209] | [0.230, 0.801] |
| Real -> Synth | 0 | images | [0.722, 0.783] | [0.366, 0.414] | [0.529, 0.603] |
| Real -> Synth | 1 | images | [0.618, 0.696] | [0.293, 0.340] | [0.528, 0.602] |
| Real -> Synth | 2 | images | [0.845, 0.892] | [0.444, 0.491] | [0.715, 0.782] |
| Real -> Real | 0 | images | [0.925, 0.973] | [0.652, 0.697] | [0.917, 0.970] |
| Real -> Real | 0 | videos | [0.862, 0.994] | [0.569, 0.747] | [0.885, 0.992] |
| Real -> Real | 1 | images | [0.930, 0.975] | [0.651, 0.692] | [0.916, 0.969] |
| Real -> Real | 1 | videos | [0.845, 0.994] | [0.579, 0.732] | [0.845, 0.998] |
| Real -> Real | 2 | images | [0.927, 0.973] | [0.658, 0.701] | [0.920, 0.973] |
| Real -> Real | 2 | videos | [0.847, 0.995] | [0.581, 0.733] | [0.859, 0.997] |
