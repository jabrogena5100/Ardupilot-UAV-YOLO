# Level 2: seed aggregation

Per-seed values come from independently trained checkpoints (seed 0 = level2_v1, seeds 1-2 = level2_v2) evaluated through the same harness. **SD is the sample standard deviation across seeds (n = number of seeds; with n=3 it is itself very uncertain). Bootstrap intervals below are a different quantity and are not seed uncertainty.**

## mAP50

| cell | seed 0 | seed 1 | seed 2 | mean | SD | min-max |
|---|---|---|---|---|---|---|
| Synth -> Synth | 0.995 | 0.995 | 0.995 | 0.995 | 0.000 | 0.995-0.995 |
| Synth -> Real | 0.602 | 0.522 | 0.413 | 0.512 | 0.095 | 0.413-0.602 |
| Real -> Synth | 0.587 | 0.522 | 0.726 | 0.612 | 0.104 | 0.522-0.726 |
| Real -> Real | 0.923 | 0.903 | 0.870 | 0.899 | 0.027 | 0.870-0.923 |

## mAP50-95

| cell | seed 0 | seed 1 | seed 2 | mean | SD | min-max |
|---|---|---|---|---|---|---|
| Synth -> Synth | 0.985 | 0.987 | 0.986 | 0.986 | 0.001 | 0.985-0.987 |
| Synth -> Real | 0.283 | 0.244 | 0.186 | 0.238 | 0.049 | 0.186-0.283 |
| Real -> Synth | 0.274 | 0.245 | 0.348 | 0.289 | 0.053 | 0.245-0.348 |
| Real -> Real | 0.547 | 0.539 | 0.508 | 0.531 | 0.021 | 0.508-0.547 |

## recall@0.25

| cell | seed 0 | seed 1 | seed 2 | mean | SD | min-max |
|---|---|---|---|---|---|---|
| Synth -> Synth | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000-1.000 |
| Synth -> Real | 0.570 | 0.514 | 0.506 | 0.530 | 0.035 | 0.506-0.570 |
| Real -> Synth | 0.495 | 0.405 | 0.589 | 0.496 | 0.092 | 0.405-0.589 |
| Real -> Real | 0.920 | 0.906 | 0.871 | 0.899 | 0.025 | 0.871-0.920 |

## FP/img@0.25

| cell | seed 0 | seed 1 | seed 2 | mean | SD | min-max |
|---|---|---|---|---|---|---|
| Synth -> Synth | 0.000 | 0.000 | 0.001 | 0.000 | 0.001 | 0.000-0.001 |
| Synth -> Real | 0.224 | 0.297 | 0.434 | 0.319 | 0.106 | 0.224-0.434 |
| Real -> Synth | 0.086 | 0.091 | 0.039 | 0.072 | 0.029 | 0.039-0.091 |
| Real -> Real | 0.067 | 0.113 | 0.206 | 0.129 | 0.071 | 0.067-0.206 |

## Cross-domain change (cross-domain minus in-domain, same family and seed)

| model family | metric | per-seed change | mean | SD |
|---|---|---|---|---|
| A (synthetic-trained): synth_test_pos -> boreal_v2_test | mAP50 | -0.393, -0.473, -0.582 | -0.483 | 0.095 |
| A (synthetic-trained): synth_test_pos -> boreal_v2_test | mAP50-95 | -0.702, -0.743, -0.800 | -0.748 | 0.049 |
| A (synthetic-trained): synth_test_pos -> boreal_v2_test | recall@0.25 | -0.430, -0.486, -0.494 | -0.470 | 0.035 |
| A (synthetic-trained): synth_test_pos -> boreal_v2_test | FP/img@0.25 | +0.224, +0.297, +0.433 | +0.318 | 0.106 |
| B (real-trained): boreal_v2_test -> synth_test_pos | mAP50 | -0.336, -0.381, -0.144 | -0.287 | 0.126 |
| B (real-trained): boreal_v2_test -> synth_test_pos | mAP50-95 | -0.273, -0.294, -0.160 | -0.242 | 0.072 |
| B (real-trained): boreal_v2_test -> synth_test_pos | recall@0.25 | -0.425, -0.501, -0.282 | -0.403 | 0.111 |
| B (real-trained): boreal_v2_test -> synth_test_pos | FP/img@0.25 | +0.018, -0.022, -0.167 | -0.057 | 0.098 |

## Bootstrap 95% CIs per seed (test-set sampling only, fixed checkpoint)

Image-level resampling treats frames as independent. Boreal frames from one video are correlated, so the video-cluster row (5 test videos) is the more honest one for Boreal; the percentile method is crude with so few clusters.

| cell | seed | resampling | mAP50 | mAP50-95 | recall@0.25 |
|---|---|---|---|---|---|
| Synth -> Synth | 0 | images | [0.995, 0.995] | [0.982, 0.988] | [1.000, 1.000] |
| Synth -> Synth | 1 | images | [0.995, 0.995] | [0.984, 0.989] | [1.000, 1.000] |
| Synth -> Synth | 2 | images | [0.995, 0.995] | [0.982, 0.988] | [1.000, 1.000] |
| Synth -> Real | 0 | images | [0.561, 0.645] | [0.257, 0.311] | [0.532, 0.607] |
| Synth -> Real | 0 | videos | [0.175, 0.810] | [0.051, 0.410] | [0.179, 0.806] |
| Synth -> Real | 1 | images | [0.480, 0.565] | [0.218, 0.270] | [0.476, 0.553] |
| Synth -> Real | 1 | videos | [0.087, 0.756] | [0.022, 0.387] | [0.103, 0.755] |
| Synth -> Real | 2 | images | [0.370, 0.460] | [0.163, 0.211] | [0.468, 0.545] |
| Synth -> Real | 2 | videos | [0.036, 0.716] | [0.008, 0.362] | [0.078, 0.759] |
| Real -> Synth | 0 | images | [0.550, 0.625] | [0.253, 0.297] | [0.458, 0.534] |
| Real -> Synth | 1 | images | [0.481, 0.557] | [0.223, 0.268] | [0.371, 0.439] |
| Real -> Synth | 2 | images | [0.693, 0.758] | [0.329, 0.371] | [0.552, 0.623] |
| Real -> Real | 0 | images | [0.895, 0.950] | [0.524, 0.573] | [0.888, 0.946] |
| Real -> Real | 0 | videos | [0.838, 0.966] | [0.429, 0.651] | [0.827, 0.960] |
| Real -> Real | 1 | images | [0.870, 0.931] | [0.514, 0.566] | [0.872, 0.933] |
| Real -> Real | 1 | videos | [0.807, 0.967] | [0.429, 0.641] | [0.825, 0.960] |
| Real -> Real | 2 | images | [0.840, 0.898] | [0.484, 0.532] | [0.838, 0.900] |
| Real -> Real | 2 | videos | [0.699, 0.940] | [0.372, 0.599] | [0.726, 0.934] |
