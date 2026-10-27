# Level 2 evaluation matrix

conf floor 0.001, NMS IoU 0.7, imgsz 640; @op = IoU 0.5 at conf >= 0.25. Same code path for every row (`eval/run_matrix.py` -> `eval/metrics.py`).

| train -> test | images | mAP50 | mAP50-95 | P@op | R@op | FNR@op | FP/img@op | empty-img FP | mean conf (TP) | mean IoU (TP) |
|---|---|---|---|---|---|---|---|---|---|---|
| synthetic -> synth_test_pos | 701 | 0.995 | 0.985 | 1.000 | 1.000 | 0.000 | 0.000 | n/a | 0.968 | 0.975 |
| synthetic -> boreal_test | 701 | 0.429 | 0.216 | 0.579 | 0.572 | 0.428 | 0.424 | 1.000 | 0.818 | 0.761 |
| real -> synth_test_pos | 701 | 0.753 | 0.389 | 0.900 | 0.568 | 0.432 | 0.063 | n/a | 0.695 | 0.767 |
| real -> boreal_test | 701 | 0.951 | 0.673 | 0.888 | 0.945 | 0.055 | 0.121 | 1.000 | 0.803 | 0.843 |
| synthetic -> synth_test_all | 1051 | 0.995 | 0.985 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.968 | 0.975 |
| real -> synth_test_all | 1051 | 0.737 | 0.378 | 0.892 | 0.568 | 0.432 | 0.046 | 0.011 | 0.695 | 0.767 |

Edge-touch diagnostic (recall at the same operating point, GT boxes touching the image border vs not):

| cell | recall edge-touching | recall not edge-touching |
|---|---|---|
| model_a__synth_test_pos | 1.000 | 1.000 |
| model_a__boreal_test | 0.873 | 0.497 |
| model_b__synth_test_pos | 0.594 | 0.478 |
| model_b__boreal_test | 0.993 | 0.933 |
| model_a__synth_test_all | 1.000 | 1.000 |
| model_b__synth_test_all | 0.594 | 0.478 |

Reproduction check against the one-off Ultralytics numbers (reported only):

| cell | metric | reference | harness | diff | tolerance |
|---|---|---|---|---|---|
| model_b on boreal_test | mAP50 | 0.9442 | 0.9511 | +0.0069 | within 0.01 |
| model_b on boreal_test | mAP50-95 | 0.6688 | 0.6730 | +0.0042 | within 0.02 |

Limitations: `synth_smoke_v1` edge-touch rate ~0.76 vs Boreal ~0.30 (documented, not corrected); the Boreal test set has no usable negative images; Model A is a baseline, not evidence the synthetic domain is optimal.
