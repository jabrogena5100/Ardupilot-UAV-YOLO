# Level 2 evaluation matrix

conf floor 0.001, NMS IoU 0.7, imgsz 640; @op = IoU 0.5 at conf >= 0.25. Same code path for every row (`eval/run_matrix.py` -> `eval/metrics.py`).

| train -> test | images | mAP50 | mAP50-95 | P@op | R@op | FNR@op | FP/img@op | empty-img FP | mean conf (TP) | mean IoU (TP) |
|---|---|---|---|---|---|---|---|---|---|---|
| synthetic -> boreal_test | 701 | 0.429 | 0.216 | 0.579 | 0.572 | 0.428 | 0.424 | 1.000 | 0.818 | 0.761 |
| synthetic -> synth_test_pos | 701 | 0.995 | 0.985 | 1.000 | 1.000 | 0.000 | 0.000 | n/a | 0.968 | 0.975 |
| synthetic -> synth_test_all | 1051 | 0.995 | 0.985 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.968 | 0.975 |
| synthetic -> boreal_empty_train_diag | 256 | 0.000 | 0.000 | 0.000 | n/a | n/a | 0.422 | 0.379 | n/a | n/a |
| synthetic -> boreal_test | 701 | 0.384 | 0.158 | 0.577 | 0.496 | 0.504 | 0.371 | 1.000 | 0.782 | 0.728 |
| synthetic -> synth_test_pos | 701 | 0.995 | 0.987 | 1.000 | 1.000 | 0.000 | 0.000 | n/a | 0.964 | 0.976 |
| synthetic -> synth_test_all | 1051 | 0.995 | 0.987 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.964 | 0.976 |
| synthetic -> boreal_empty_train_diag | 256 | 0.000 | 0.000 | 0.000 | n/a | n/a | 0.160 | 0.141 | n/a | n/a |
| synthetic -> boreal_test | 701 | 0.313 | 0.116 | 0.492 | 0.488 | 0.512 | 0.514 | 1.000 | 0.867 | 0.709 |
| synthetic -> synth_test_pos | 701 | 0.995 | 0.986 | 0.999 | 1.000 | 0.000 | 0.001 | n/a | 0.966 | 0.976 |
| synthetic -> synth_test_all | 1051 | 0.995 | 0.986 | 0.999 | 1.000 | 0.000 | 0.001 | 0.000 | 0.966 | 0.976 |
| synthetic -> boreal_empty_train_diag | 256 | 0.000 | 0.000 | 0.000 | n/a | n/a | 0.516 | 0.496 | n/a | n/a |
| real -> boreal_test | 701 | 0.951 | 0.673 | 0.888 | 0.945 | 0.055 | 0.121 | 1.000 | 0.803 | 0.843 |
| real -> synth_test_pos | 701 | 0.753 | 0.389 | 0.900 | 0.568 | 0.432 | 0.063 | n/a | 0.695 | 0.767 |
| real -> synth_test_all | 1051 | 0.737 | 0.378 | 0.892 | 0.568 | 0.432 | 0.046 | 0.011 | 0.695 | 0.767 |
| real -> boreal_empty_train_diag | 256 | 0.000 | 0.000 | 0.000 | n/a | n/a | 0.004 | 0.004 | n/a | n/a |
| real -> boreal_test | 701 | 0.956 | 0.672 | 0.960 | 0.945 | 0.055 | 0.040 | 1.000 | 0.794 | 0.840 |
| real -> synth_test_pos | 701 | 0.656 | 0.316 | 0.780 | 0.565 | 0.435 | 0.160 | n/a | 0.736 | 0.746 |
| real -> synth_test_all | 1051 | 0.637 | 0.306 | 0.756 | 0.565 | 0.435 | 0.122 | 0.043 | 0.736 | 0.746 |
| real -> boreal_empty_train_diag | 256 | 0.000 | 0.000 | 0.000 | n/a | n/a | 0.004 | 0.004 | n/a | n/a |
| real -> boreal_test | 701 | 0.953 | 0.679 | 0.951 | 0.950 | 0.050 | 0.050 | 1.000 | 0.833 | 0.846 |
| real -> synth_test_pos | 701 | 0.869 | 0.467 | 0.929 | 0.749 | 0.251 | 0.057 | n/a | 0.738 | 0.773 |
| real -> synth_test_all | 1051 | 0.853 | 0.456 | 0.904 | 0.749 | 0.251 | 0.053 | 0.040 | 0.738 | 0.773 |
| real -> boreal_empty_train_diag | 256 | 0.000 | 0.000 | 0.000 | n/a | n/a | 0.039 | 0.039 | n/a | n/a |

Image-level bootstrap 95% CIs (test-set sampling only; not training-seed variance; Boreal frames from the same video are correlated, so its intervals are optimistic):

| cell | mAP50 | mAP50-95 | R@op | P@op | FP/img@op |
|---|---|---|---|---|---|
| model_a__boreal_test (images) | [0.388, 0.474] | [0.196, 0.242] | [0.537, 0.613] | [0.541, 0.615] | [0.382, 0.465] |
| model_a__boreal_test (videos) | [0.206, 0.969] | [0.101, 0.468] | [0.327, 0.934] | [0.341, 0.951] | [0.048, 0.645] |
| model_a__synth_test_pos (images) | [0.995, 0.995] | [0.982, 0.988] | [1.000, 1.000] | [1.000, 1.000] | [0.000, 0.000] |
| model_a__synth_test_all (images) | [0.995, 0.995] | [0.982, 0.988] | [1.000, 1.000] | [1.000, 1.000] | [0.000, 0.000] |
| model_a__boreal_empty_train_diag (images) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.355, 0.500] |
| model_a_seed1__boreal_test (images) | [0.344, 0.429] | [0.140, 0.181] | [0.461, 0.535] | [0.540, 0.616] | [0.332, 0.409] |
| model_a_seed1__boreal_test (videos) | [0.181, 0.885] | [0.076, 0.282] | [0.249, 0.782] | [0.374, 0.902] | [0.085, 0.550] |
| model_a_seed1__synth_test_pos (images) | [0.995, 0.995] | [0.984, 0.989] | [1.000, 1.000] | [1.000, 1.000] | [0.000, 0.000] |
| model_a_seed1__synth_test_all (images) | [0.995, 0.995] | [0.984, 0.989] | [1.000, 1.000] | [1.000, 1.000] | [0.000, 0.000] |
| model_a_seed1__boreal_empty_train_diag (images) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.109, 0.211] |
| model_a_seed2__boreal_test (images) | [0.280, 0.355] | [0.102, 0.134] | [0.452, 0.526] | [0.456, 0.529] | [0.472, 0.552] |
| model_a_seed2__boreal_test (videos) | [0.094, 0.769] | [0.038, 0.209] | [0.230, 0.801] | [0.227, 0.809] | [0.188, 0.796] |
| model_a_seed2__synth_test_pos (images) | [0.995, 0.995] | [0.982, 0.988] | [1.000, 1.000] | [0.996, 1.000] | [0.000, 0.004] |
| model_a_seed2__synth_test_all (images) | [0.995, 0.995] | [0.982, 0.988] | [1.000, 1.000] | [0.996, 1.000] | [0.000, 0.003] |
| model_a_seed2__boreal_empty_train_diag (images) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.453, 0.578] |
| model_b__boreal_test (images) | [0.925, 0.973] | [0.652, 0.697] | [0.917, 0.970] | [0.866, 0.910] | [0.096, 0.147] |
| model_b__boreal_test (videos) | [0.862, 0.994] | [0.569, 0.747] | [0.885, 0.992] | [0.776, 0.990] | [0.010, 0.262] |
| model_b__synth_test_pos (images) | [0.722, 0.783] | [0.366, 0.414] | [0.529, 0.603] | [0.873, 0.928] | [0.046, 0.081] |
| model_b__synth_test_all (images) | [0.706, 0.768] | [0.355, 0.403] | [0.532, 0.604] | [0.866, 0.920] | [0.033, 0.058] |
| model_b__boreal_empty_train_diag (images) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.000, 0.012] |
| model_b_seed1__boreal_test (images) | [0.930, 0.975] | [0.651, 0.692] | [0.916, 0.969] | [0.946, 0.974] | [0.026, 0.054] |
| model_b_seed1__boreal_test (videos) | [0.845, 0.994] | [0.579, 0.732] | [0.845, 0.998] | [0.845, 0.995] | [0.005, 0.155] |
| model_b_seed1__synth_test_pos (images) | [0.618, 0.696] | [0.293, 0.340] | [0.528, 0.602] | [0.741, 0.818] | [0.131, 0.190] |
| model_b_seed1__synth_test_all (images) | [0.600, 0.675] | [0.283, 0.332] | [0.529, 0.602] | [0.715, 0.796] | [0.099, 0.145] |
| model_b_seed1__boreal_empty_train_diag (images) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.000, 0.012] |
| model_b_seed2__boreal_test (images) | [0.927, 0.973] | [0.658, 0.701] | [0.920, 0.973] | [0.936, 0.965] | [0.036, 0.066] |
| model_b_seed2__boreal_test (videos) | [0.847, 0.995] | [0.581, 0.733] | [0.859, 0.997] | [0.861, 0.995] | [0.005, 0.138] |
| model_b_seed2__synth_test_pos (images) | [0.845, 0.892] | [0.444, 0.491] | [0.715, 0.782] | [0.907, 0.949] | [0.040, 0.076] |
| model_b_seed2__synth_test_all (images) | [0.828, 0.878] | [0.434, 0.480] | [0.717, 0.783] | [0.880, 0.929] | [0.039, 0.069] |
| model_b_seed2__boreal_empty_train_diag (images) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.016, 0.062] |

Edge-touch diagnostic (recall at the same operating point, GT boxes touching the image border vs not):

| cell | recall edge-touching | recall not edge-touching |
|---|---|---|
| model_a__boreal_test | 0.873 | 0.497 |
| model_a__synth_test_pos | 1.000 | 1.000 |
| model_a__synth_test_all | 1.000 | 1.000 |
| model_a__boreal_empty_train_diag | n/a | n/a |
| model_a_seed1__boreal_test | 0.746 | 0.434 |
| model_a_seed1__synth_test_pos | 1.000 | 1.000 |
| model_a_seed1__synth_test_all | 1.000 | 1.000 |
| model_a_seed1__boreal_empty_train_diag | n/a | n/a |
| model_a_seed2__boreal_test | 0.746 | 0.424 |
| model_a_seed2__synth_test_pos | 1.000 | 1.000 |
| model_a_seed2__synth_test_all | 1.000 | 1.000 |
| model_a_seed2__boreal_empty_train_diag | n/a | n/a |
| model_b__boreal_test | 0.993 | 0.933 |
| model_b__synth_test_pos | 0.594 | 0.478 |
| model_b__synth_test_all | 0.594 | 0.478 |
| model_b__boreal_empty_train_diag | n/a | n/a |
| model_b_seed1__boreal_test | 0.993 | 0.933 |
| model_b_seed1__synth_test_pos | 0.631 | 0.340 |
| model_b_seed1__synth_test_all | 0.631 | 0.340 |
| model_b_seed1__boreal_empty_train_diag | n/a | n/a |
| model_b_seed2__boreal_test | 1.000 | 0.937 |
| model_b_seed2__synth_test_pos | 0.788 | 0.616 |
| model_b_seed2__synth_test_all | 0.788 | 0.616 |
| model_b_seed2__boreal_empty_train_diag | n/a | n/a |

Reproduction check against the one-off Ultralytics numbers (reported only):

| cell | metric | reference | harness | diff | tolerance |
|---|---|---|---|---|---|
| model_b on boreal_test | mAP50 | 0.9442 | 0.9511 | +0.0069 | within 0.01 |
| model_b on boreal_test | mAP50-95 | 0.6688 | 0.6730 | +0.0042 | within 0.02 |

Limitations: `synth_smoke_v1` edge-touch rate ~0.76 vs Boreal ~0.30 (documented, not corrected); the Boreal test set has no usable negative images; Model A is a baseline, not evidence the synthetic domain is optimal.
