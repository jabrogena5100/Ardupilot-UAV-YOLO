# Level 2 evaluation matrix

conf floor 0.001, NMS IoU 0.7, imgsz 640; @op = IoU 0.5 at conf >= 0.25. Same code path for every row (`eval/run_matrix.py` -> `eval/metrics.py`).

| train -> test | images | mAP50 | mAP50-95 | P@op | R@op | FNR@op | FP/img@op | empty-img FP | mean conf (TP) | mean IoU (TP) |
|---|---|---|---|---|---|---|---|---|---|---|
| synthetic -> boreal_v2_test | 744 | 0.602 | 0.283 | 0.702 | 0.570 | 0.430 | 0.224 | 0.138 | 0.841 | 0.740 |
| synthetic -> boreal_v2_test_pos | 679 | 0.608 | 0.284 | 0.713 | 0.568 | 0.432 | 0.233 | n/a | 0.837 | 0.739 |
| synthetic -> boreal_v2_test_neg | 65 | 0.000 | 0.000 | 0.000 | n/a | n/a | 0.169 | 0.138 | n/a | n/a |
| synthetic -> synth_test_pos | 701 | 0.995 | 0.985 | 1.000 | 1.000 | 0.000 | 0.000 | n/a | 0.968 | 0.975 |
| synthetic -> synth_test_all | 1051 | 0.995 | 0.985 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.968 | 0.975 |
| synthetic -> boreal_v2_test | 744 | 0.522 | 0.244 | 0.616 | 0.514 | 0.486 | 0.297 | 0.169 | 0.861 | 0.739 |
| synthetic -> boreal_v2_test_pos | 679 | 0.524 | 0.243 | 0.620 | 0.513 | 0.487 | 0.320 | n/a | 0.862 | 0.738 |
| synthetic -> boreal_v2_test_neg | 65 | 0.000 | 0.000 | 0.000 | n/a | n/a | 0.185 | 0.169 | n/a | n/a |
| synthetic -> synth_test_pos | 701 | 0.995 | 0.987 | 1.000 | 1.000 | 0.000 | 0.000 | n/a | 0.964 | 0.976 |
| synthetic -> synth_test_all | 1051 | 0.995 | 0.987 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.964 | 0.976 |
| synthetic -> boreal_v2_test | 744 | 0.413 | 0.186 | 0.519 | 0.506 | 0.494 | 0.434 | 0.277 | 0.900 | 0.736 |
| synthetic -> boreal_v2_test_pos | 679 | 0.417 | 0.187 | 0.523 | 0.504 | 0.496 | 0.467 | n/a | 0.901 | 0.736 |
| synthetic -> boreal_v2_test_neg | 65 | 0.000 | 0.000 | 0.000 | n/a | n/a | 0.308 | 0.262 | n/a | n/a |
| synthetic -> synth_test_pos | 701 | 0.995 | 0.986 | 0.999 | 1.000 | 0.000 | 0.001 | n/a | 0.966 | 0.976 |
| synthetic -> synth_test_all | 1051 | 0.995 | 0.986 | 0.999 | 1.000 | 0.000 | 0.001 | 0.000 | 0.966 | 0.976 |
| real -> boreal_v2_test | 744 | 0.923 | 0.547 | 0.927 | 0.920 | 0.080 | 0.067 | 0.015 | 0.809 | 0.796 |
| real -> boreal_v2_test_pos | 679 | 0.925 | 0.549 | 0.932 | 0.920 | 0.080 | 0.068 | n/a | 0.809 | 0.796 |
| real -> boreal_v2_test_neg | 65 | 0.000 | 0.000 | 0.000 | n/a | n/a | 0.015 | 0.015 | n/a | n/a |
| real -> synth_test_pos | 701 | 0.587 | 0.274 | 0.853 | 0.495 | 0.505 | 0.086 | n/a | 0.756 | 0.752 |
| real -> synth_test_all | 1051 | 0.570 | 0.267 | 0.801 | 0.495 | 0.505 | 0.082 | 0.069 | 0.756 | 0.752 |
| real -> boreal_v2_test | 744 | 0.903 | 0.539 | 0.882 | 0.906 | 0.094 | 0.113 | 0.015 | 0.790 | 0.798 |
| real -> boreal_v2_test_pos | 679 | 0.907 | 0.547 | 0.883 | 0.904 | 0.096 | 0.122 | n/a | 0.791 | 0.800 |
| real -> boreal_v2_test_neg | 65 | 0.000 | 0.000 | 0.000 | n/a | n/a | 0.015 | 0.015 | n/a | n/a |
| real -> synth_test_pos | 701 | 0.522 | 0.245 | 0.816 | 0.405 | 0.595 | 0.091 | n/a | 0.724 | 0.762 |
| real -> synth_test_all | 1051 | 0.487 | 0.231 | 0.740 | 0.405 | 0.595 | 0.095 | 0.089 | 0.724 | 0.762 |
| real -> boreal_v2_test | 744 | 0.870 | 0.508 | 0.797 | 0.871 | 0.129 | 0.206 | 0.015 | 0.779 | 0.789 |
| real -> boreal_v2_test_pos | 679 | 0.875 | 0.510 | 0.799 | 0.874 | 0.126 | 0.224 | n/a | 0.778 | 0.787 |
| real -> boreal_v2_test_neg | 65 | 0.000 | 0.000 | 0.000 | n/a | n/a | 0.015 | 0.015 | n/a | n/a |
| real -> synth_test_pos | 701 | 0.726 | 0.348 | 0.939 | 0.589 | 0.411 | 0.039 | n/a | 0.696 | 0.751 |
| real -> synth_test_all | 1051 | 0.712 | 0.340 | 0.928 | 0.589 | 0.411 | 0.030 | 0.014 | 0.696 | 0.751 |

Image-level bootstrap 95% CIs (test-set sampling only; not training-seed variance; Boreal frames from the same video are correlated, so its intervals are optimistic):

| cell | mAP50 | mAP50-95 | R@op | P@op | FP/img@op |
|---|---|---|---|---|---|
| model_a__boreal_v2_test (images) | [0.561, 0.645] | [0.257, 0.311] | [0.532, 0.607] | [0.665, 0.743] | [0.191, 0.255] |
| model_a__boreal_v2_test (videos) | [0.175, 0.810] | [0.051, 0.410] | [0.179, 0.806] | [0.322, 0.833] | [0.149, 0.342] |
| model_a__boreal_v2_test_pos (images) | [0.566, 0.653] | [0.260, 0.314] | [0.528, 0.610] | [0.676, 0.747] | [0.203, 0.265] |
| model_a__boreal_v2_test_pos (videos) | [0.184, 0.816] | [0.054, 0.409] | [0.179, 0.804] | [0.331, 0.842] | [0.151, 0.363] |
| model_a__boreal_v2_test_neg (images) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.062, 0.292] |
| model_a__boreal_v2_test_neg (videos) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.000, 0.611] |
| model_a__synth_test_pos (images) | [0.995, 0.995] | [0.982, 0.988] | [1.000, 1.000] | [1.000, 1.000] | [0.000, 0.000] |
| model_a__synth_test_all (images) | [0.995, 0.995] | [0.982, 0.988] | [1.000, 1.000] | [1.000, 1.000] | [0.000, 0.000] |
| model_a_seed1__boreal_v2_test (images) | [0.480, 0.565] | [0.218, 0.270] | [0.476, 0.553] | [0.578, 0.659] | [0.258, 0.335] |
| model_a_seed1__boreal_v2_test (videos) | [0.087, 0.756] | [0.022, 0.387] | [0.103, 0.755] | [0.224, 0.725] | [0.214, 0.423] |
| model_a_seed1__boreal_v2_test_pos (images) | [0.482, 0.567] | [0.220, 0.270] | [0.475, 0.551] | [0.583, 0.659] | [0.278, 0.361] |
| model_a_seed1__boreal_v2_test_pos (videos) | [0.091, 0.757] | [0.024, 0.385] | [0.103, 0.752] | [0.236, 0.729] | [0.234, 0.480] |
| model_a_seed1__boreal_v2_test_neg (images) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.092, 0.292] |
| model_a_seed1__boreal_v2_test_neg (videos) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.106, 0.389] |
| model_a_seed1__synth_test_pos (images) | [0.995, 0.995] | [0.984, 0.989] | [1.000, 1.000] | [1.000, 1.000] | [0.000, 0.000] |
| model_a_seed1__synth_test_all (images) | [0.995, 0.995] | [0.984, 0.989] | [1.000, 1.000] | [1.000, 1.000] | [0.000, 0.000] |
| model_a_seed2__boreal_v2_test (images) | [0.370, 0.460] | [0.163, 0.211] | [0.468, 0.545] | [0.481, 0.559] | [0.395, 0.473] |
| model_a_seed2__boreal_v2_test (videos) | [0.036, 0.716] | [0.008, 0.362] | [0.078, 0.759] | [0.077, 0.777] | [0.200, 0.769] |
| model_a_seed2__boreal_v2_test_pos (images) | [0.376, 0.462] | [0.164, 0.211] | [0.467, 0.541] | [0.487, 0.558] | [0.431, 0.507] |
| model_a_seed2__boreal_v2_test_pos (videos) | [0.038, 0.719] | [0.008, 0.364] | [0.078, 0.756] | [0.084, 0.779] | [0.221, 0.839] |
| model_a_seed2__boreal_v2_test_neg (images) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.169, 0.462] |
| model_a_seed2__boreal_v2_test_neg (videos) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.043, 1.000] |
| model_a_seed2__synth_test_pos (images) | [0.995, 0.995] | [0.982, 0.988] | [1.000, 1.000] | [0.996, 1.000] | [0.000, 0.004] |
| model_a_seed2__synth_test_all (images) | [0.995, 0.995] | [0.982, 0.988] | [1.000, 1.000] | [0.996, 1.000] | [0.000, 0.003] |
| model_bv2__boreal_v2_test (images) | [0.895, 0.950] | [0.524, 0.573] | [0.888, 0.946] | [0.908, 0.945] | [0.051, 0.085] |
| model_bv2__boreal_v2_test (videos) | [0.838, 0.966] | [0.429, 0.651] | [0.827, 0.960] | [0.880, 0.983] | [0.013, 0.112] |
| model_bv2__boreal_v2_test_pos (images) | [0.898, 0.952] | [0.523, 0.577] | [0.890, 0.947] | [0.913, 0.951] | [0.049, 0.087] |
| model_bv2__boreal_v2_test_pos (videos) | [0.848, 0.966] | [0.438, 0.651] | [0.827, 0.960] | [0.888, 0.985] | [0.015, 0.113] |
| model_bv2__boreal_v2_test_neg (images) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.000, 0.046] |
| model_bv2__boreal_v2_test_neg (videos) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.000, 0.056] |
| model_bv2__synth_test_pos (images) | [0.550, 0.625] | [0.253, 0.297] | [0.458, 0.534] | [0.820, 0.887] | [0.066, 0.107] |
| model_bv2__synth_test_all (images) | [0.533, 0.608] | [0.246, 0.292] | [0.459, 0.532] | [0.764, 0.841] | [0.064, 0.099] |
| model_bv2_seed1__boreal_v2_test (images) | [0.870, 0.931] | [0.514, 0.566] | [0.872, 0.933] | [0.860, 0.905] | [0.090, 0.137] |
| model_bv2_seed1__boreal_v2_test (videos) | [0.807, 0.967] | [0.429, 0.641] | [0.825, 0.960] | [0.795, 0.954] | [0.041, 0.188] |
| model_bv2_seed1__boreal_v2_test_pos (images) | [0.875, 0.932] | [0.521, 0.573] | [0.873, 0.931] | [0.859, 0.906] | [0.096, 0.147] |
| model_bv2_seed1__boreal_v2_test_pos (videos) | [0.818, 0.969] | [0.446, 0.643] | [0.825, 0.960] | [0.800, 0.954] | [0.046, 0.218] |
| model_bv2_seed1__boreal_v2_test_neg (images) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.000, 0.046] |
| model_bv2_seed1__boreal_v2_test_neg (videos) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.000, 0.056] |
| model_bv2_seed1__synth_test_pos (images) | [0.481, 0.557] | [0.223, 0.268] | [0.371, 0.439] | [0.774, 0.860] | [0.070, 0.114] |
| model_bv2_seed1__synth_test_all (images) | [0.449, 0.529] | [0.210, 0.257] | [0.369, 0.445] | [0.697, 0.785] | [0.076, 0.113] |
| model_bv2_seed2__boreal_v2_test (images) | [0.840, 0.898] | [0.484, 0.532] | [0.838, 0.900] | [0.768, 0.825] | [0.173, 0.241] |
| model_bv2_seed2__boreal_v2_test (videos) | [0.699, 0.940] | [0.372, 0.599] | [0.726, 0.934] | [0.629, 0.875] | [0.131, 0.317] |
| model_bv2_seed2__boreal_v2_test_pos (images) | [0.846, 0.903] | [0.485, 0.535] | [0.842, 0.903] | [0.770, 0.827] | [0.189, 0.259] |
| model_bv2_seed2__boreal_v2_test_pos (videos) | [0.716, 0.941] | [0.378, 0.600] | [0.731, 0.935] | [0.643, 0.871] | [0.139, 0.400] |
| model_bv2_seed2__boreal_v2_test_neg (images) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.000, 0.046] |
| model_bv2_seed2__boreal_v2_test_neg (videos) | [0.000, 0.000] | [0.000, 0.000] | n/a | [0.000, 0.000] | [0.000, 0.056] |
| model_bv2_seed2__synth_test_pos (images) | [0.693, 0.758] | [0.329, 0.371] | [0.552, 0.623] | [0.916, 0.960] | [0.024, 0.053] |
| model_bv2_seed2__synth_test_all (images) | [0.680, 0.745] | [0.321, 0.364] | [0.555, 0.624] | [0.904, 0.952] | [0.020, 0.041] |

Edge-touch diagnostic (recall at the same operating point, GT boxes touching the image border vs not):

| cell | recall edge-touching | recall not edge-touching |
|---|---|---|
| model_a__boreal_v2_test | 0.892 | 0.482 |
| model_a__boreal_v2_test_pos | 0.892 | 0.480 |
| model_a__boreal_v2_test_neg | n/a | n/a |
| model_a__synth_test_pos | 1.000 | 1.000 |
| model_a__synth_test_all | 1.000 | 1.000 |
| model_a_seed1__boreal_v2_test | 0.703 | 0.463 |
| model_a_seed1__boreal_v2_test_pos | 0.696 | 0.463 |
| model_a_seed1__boreal_v2_test_neg | n/a | n/a |
| model_a_seed1__synth_test_pos | 1.000 | 1.000 |
| model_a_seed1__synth_test_all | 1.000 | 1.000 |
| model_a_seed2__boreal_v2_test | 0.689 | 0.456 |
| model_a_seed2__boreal_v2_test_pos | 0.682 | 0.456 |
| model_a_seed2__boreal_v2_test_neg | n/a | n/a |
| model_a_seed2__synth_test_pos | 1.000 | 1.000 |
| model_a_seed2__synth_test_all | 1.000 | 1.000 |
| model_bv2__boreal_v2_test | 0.966 | 0.908 |
| model_bv2__boreal_v2_test_pos | 0.966 | 0.908 |
| model_bv2__boreal_v2_test_neg | n/a | n/a |
| model_bv2__synth_test_pos | 0.579 | 0.208 |
| model_bv2__synth_test_all | 0.579 | 0.208 |
| model_bv2_seed1__boreal_v2_test | 0.939 | 0.897 |
| model_bv2_seed1__boreal_v2_test_pos | 0.939 | 0.895 |
| model_bv2_seed1__boreal_v2_test_neg | n/a | n/a |
| model_bv2_seed1__synth_test_pos | 0.463 | 0.208 |
| model_bv2_seed1__synth_test_all | 0.463 | 0.208 |
| model_bv2_seed2__boreal_v2_test | 0.946 | 0.851 |
| model_bv2_seed2__boreal_v2_test_pos | 0.953 | 0.852 |
| model_bv2_seed2__boreal_v2_test_neg | n/a | n/a |
| model_bv2_seed2__synth_test_pos | 0.620 | 0.484 |
| model_bv2_seed2__synth_test_all | 0.620 | 0.484 |

Limitations: `synth_smoke_v1` edge-touch rate ~0.76 vs Boreal ~0.30 (documented, not corrected); the Boreal test set has no usable negative images; Model A is a baseline, not evidence the synthetic domain is optimal.
