# Boreal v2 Level 2 results (offline report)

SD = sample standard deviation over independently trained seeds (n = 3: itself very uncertain). Bootstrap intervals (clip-clustered for Boreal, image-level for synthetic) describe test-set sampling for one fixed checkpoint only and are not seed uncertainty. **Boreal v2 estimates are not directly comparable with Boreal v1 estimates: train and test populations differ.**

## mAP50

| cell | seed 0 | seed 1 | seed 2 | mean | SD |
|---|---|---|---|---|---|
| Synth -> Synth (A on synth_test_pos) | 0.995 | 0.995 | 0.995 | 0.995 | 0.000 |
| Synth -> Real (A on boreal_v2_test) | 0.602 | 0.522 | 0.413 | 0.512 | 0.095 |
| Real -> Synth (B-v2 on synth_test_pos) | 0.587 | 0.522 | 0.726 | 0.612 | 0.104 |
| Real -> Real (B-v2 on boreal_v2_test) | 0.923 | 0.903 | 0.870 | 0.899 | 0.027 |

## mAP50-95

| cell | seed 0 | seed 1 | seed 2 | mean | SD |
|---|---|---|---|---|---|
| Synth -> Synth (A on synth_test_pos) | 0.985 | 0.987 | 0.986 | 0.986 | 0.001 |
| Synth -> Real (A on boreal_v2_test) | 0.283 | 0.244 | 0.186 | 0.238 | 0.049 |
| Real -> Synth (B-v2 on synth_test_pos) | 0.274 | 0.245 | 0.348 | 0.289 | 0.053 |
| Real -> Real (B-v2 on boreal_v2_test) | 0.547 | 0.539 | 0.508 | 0.531 | 0.021 |

## recall @ conf 0.25, IoU 0.5

| cell | seed 0 | seed 1 | seed 2 | mean | SD |
|---|---|---|---|---|---|
| Synth -> Synth (A on synth_test_pos) | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Synth -> Real (A on boreal_v2_test) | 0.570 | 0.514 | 0.506 | 0.530 | 0.035 |
| Real -> Synth (B-v2 on synth_test_pos) | 0.495 | 0.405 | 0.589 | 0.496 | 0.092 |
| Real -> Real (B-v2 on boreal_v2_test) | 0.920 | 0.906 | 0.871 | 0.899 | 0.025 |

## precision @ 0.25

| cell | seed 0 | seed 1 | seed 2 | mean | SD |
|---|---|---|---|---|---|
| Synth -> Synth (A on synth_test_pos) | 1.000 | 1.000 | 0.999 | 1.000 | 0.001 |
| Synth -> Real (A on boreal_v2_test) | 0.702 | 0.616 | 0.519 | 0.612 | 0.091 |
| Real -> Synth (B-v2 on synth_test_pos) | 0.853 | 0.816 | 0.939 | 0.869 | 0.063 |
| Real -> Real (B-v2 on boreal_v2_test) | 0.927 | 0.882 | 0.797 | 0.869 | 0.066 |

## false positives per image @ 0.25

| cell | seed 0 | seed 1 | seed 2 | mean | SD |
|---|---|---|---|---|---|
| Synth -> Synth (A on synth_test_pos) | 0.000 | 0.000 | 0.001 | 0.000 | 0.001 |
| Synth -> Real (A on boreal_v2_test) | 0.224 | 0.297 | 0.434 | 0.319 | 0.106 |
| Real -> Synth (B-v2 on synth_test_pos) | 0.086 | 0.091 | 0.039 | 0.072 | 0.029 |
| Real -> Real (B-v2 on boreal_v2_test) | 0.067 | 0.113 | 0.206 | 0.129 | 0.071 |

## Clustered uncertainty (Boreal: source-clip bootstrap, 8 clusters; synthetic: image bootstrap)

| cell | seed | unit | mAP50 [95% CI] | mAP50-95 [95% CI] | recall@0.25 [95% CI] | FP/img@0.25 [95% CI] |
|---|---|---|---|---|---|---|
| Synth -> Real (A on boreal_v2_test) | 0 | cluster (8 clusters) | 0.602 [0.175, 0.810] | 0.283 [0.051, 0.410] | 0.570 [0.179, 0.806] | 0.224 [0.149, 0.342] |
| Synth -> Real (A on boreal_v2_test) | 1 | cluster (8 clusters) | 0.522 [0.087, 0.756] | 0.244 [0.022, 0.387] | 0.514 [0.103, 0.755] | 0.297 [0.214, 0.423] |
| Synth -> Real (A on boreal_v2_test) | 2 | cluster (8 clusters) | 0.413 [0.036, 0.716] | 0.186 [0.008, 0.362] | 0.506 [0.078, 0.759] | 0.434 [0.200, 0.769] |
| Real -> Real (B-v2 on boreal_v2_test) | 0 | cluster (8 clusters) | 0.923 [0.838, 0.966] | 0.547 [0.429, 0.651] | 0.920 [0.827, 0.960] | 0.067 [0.013, 0.112] |
| Real -> Real (B-v2 on boreal_v2_test) | 1 | cluster (8 clusters) | 0.903 [0.807, 0.967] | 0.539 [0.429, 0.641] | 0.906 [0.825, 0.960] | 0.113 [0.041, 0.188] |
| Real -> Real (B-v2 on boreal_v2_test) | 2 | cluster (8 clusters) | 0.870 [0.699, 0.940] | 0.508 [0.372, 0.599] | 0.871 [0.726, 0.934] | 0.206 [0.131, 0.317] |

Synthetic cells (image-level; texture-slot clustering is a separate offline step):

| cell | seed | unit | mAP50 [95% CI] | mAP50-95 [95% CI] | recall@0.25 [95% CI] | FP/img@0.25 [95% CI] |
|---|---|---|---|---|---|---|
| Synth -> Synth (A on synth_test_pos) | 0 | image | 0.995 [0.995, 0.995] | 0.985 [0.982, 0.988] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| Synth -> Synth (A on synth_test_pos) | 1 | image | 0.995 [0.995, 0.995] | 0.987 [0.984, 0.989] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| Synth -> Synth (A on synth_test_pos) | 2 | image | 0.995 [0.995, 0.995] | 0.986 [0.982, 0.988] | 1.000 [1.000, 1.000] | 0.001 [0.000, 0.004] |
| Real -> Synth (B-v2 on synth_test_pos) | 0 | image | 0.587 [0.550, 0.625] | 0.274 [0.253, 0.297] | 0.495 [0.458, 0.534] | 0.086 [0.066, 0.107] |
| Real -> Synth (B-v2 on synth_test_pos) | 1 | image | 0.522 [0.481, 0.557] | 0.245 [0.223, 0.268] | 0.405 [0.371, 0.439] | 0.091 [0.070, 0.114] |
| Real -> Synth (B-v2 on synth_test_pos) | 2 | image | 0.726 [0.693, 0.758] | 0.348 [0.329, 0.371] | 0.589 [0.552, 0.623] | 0.039 [0.024, 0.053] |

## Negative-scene diagnostics

Real negatives: 65 Empty-Images frames from `heinola_dji_0048` and `karkkila_dji_0003` (neither family trained on these clips). `empty_img_fp_rate` = fraction of empty images with at least one detection >= 0.25.

| cell | seed | empty-image FP rate | FP/img |
|---|---|---|---|
| Model A on held-out real negatives (boreal_v2_test_neg, 65 frames, 2 clips) | 0 | 0.138 | 0.169 |
| Model A on held-out real negatives (boreal_v2_test_neg, 65 frames, 2 clips) | 1 | 0.169 | 0.185 |
| Model A on held-out real negatives (boreal_v2_test_neg, 65 frames, 2 clips) | 2 | 0.262 | 0.308 |
| Model B-v2 on held-out real negatives | 0 | 0.015 | 0.015 |
| Model B-v2 on held-out real negatives | 1 | 0.015 | 0.015 |
| Model B-v2 on held-out real negatives | 2 | 0.015 | 0.015 |
| Model A on synthetic negatives (synth_test_all, 1051 incl. 350 empty) | 0 | 0.000 | 0.000 |
| Model A on synthetic negatives (synth_test_all, 1051 incl. 350 empty) | 1 | 0.000 | 0.000 |
| Model A on synthetic negatives (synth_test_all, 1051 incl. 350 empty) | 2 | 0.000 | 0.001 |
| Model B-v2 on synthetic test_all | 0 | 0.069 | 0.082 |
| Model B-v2 on synthetic test_all | 1 | 0.089 | 0.095 |
| Model B-v2 on synthetic test_all | 2 | 0.014 | 0.030 |

## Edge-touch recall and localisation (Boreal v2 test, @0.25)

| cell | seed | recall edge-touching | recall not edge-touching | mean IoU (TP) | mean conf (TP) |
|---|---|---|---|---|---|
| Synth -> Real (A on boreal_v2_test) | 0 | 0.892 | 0.482 | 0.740 | 0.841 |
| Synth -> Real (A on boreal_v2_test) | 1 | 0.703 | 0.463 | 0.739 | 0.861 |
| Synth -> Real (A on boreal_v2_test) | 2 | 0.689 | 0.456 | 0.736 | 0.900 |
| Real -> Real (B-v2 on boreal_v2_test) | 0 | 0.966 | 0.908 | 0.796 | 0.809 |
| Real -> Real (B-v2 on boreal_v2_test) | 1 | 0.939 | 0.897 | 0.798 | 0.790 |
| Real -> Real (B-v2 on boreal_v2_test) | 2 | 0.946 | 0.851 | 0.789 | 0.779 |

## Sensitivity: drop the 6 test frames with a <=4-bit near-duplicate in train/val (recomputed offline)

| model | mAP50 full -> kept | mAP50-95 full -> kept |
|---|---|---|
| A|seed0 | 0.602 -> 0.603 | 0.283 -> 0.284 |
| A|seed1 | 0.522 -> 0.522 | 0.244 -> 0.244 |
| A|seed2 | 0.413 -> 0.414 | 0.186 -> 0.186 |
| B|seed0 | 0.923 -> 0.924 | 0.547 -> 0.547 |
| B|seed1 | 0.903 -> 0.903 | 0.539 -> 0.540 |
| B|seed2 | 0.870 -> 0.870 | 0.508 -> 0.508 |
