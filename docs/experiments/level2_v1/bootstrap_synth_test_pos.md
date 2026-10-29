# Level 2 evaluation matrix

conf floor 0.001, NMS IoU 0.7, imgsz 640; @op = IoU 0.5 at conf >= 0.25. Same code path for every row (`eval/run_matrix.py` -> `eval/metrics.py`).

| train -> test | images | mAP50 | mAP50-95 | P@op | R@op | FNR@op | FP/img@op | empty-img FP | mean conf (TP) | mean IoU (TP) |
|---|---|---|---|---|---|---|---|---|---|---|
| synthetic -> synth_test_pos | 701 | 0.995 | 0.985 | 1.000 | 1.000 | 0.000 | 0.000 | n/a | 0.968 | 0.975 |
| real -> synth_test_pos | 701 | 0.753 | 0.389 | 0.900 | 0.568 | 0.432 | 0.063 | n/a | 0.695 | 0.767 |

Image-level bootstrap 95% CIs (test-set sampling only; not training-seed variance; Boreal frames from the same video are correlated, so its intervals are optimistic):

| cell | mAP50 | mAP50-95 | R@op | P@op | FP/img@op |
|---|---|---|---|---|---|
| model_a__synth_test_pos | [0.995, 0.995] | [0.983, 0.988] | [1.000, 1.000] | [1.000, 1.000] | [0.000, 0.000] |
| model_b__synth_test_pos | [0.722, 0.779] | [0.367, 0.415] | [0.529, 0.601] | [0.876, 0.926] | [0.046, 0.081] |

Edge-touch diagnostic (recall at the same operating point, GT boxes touching the image border vs not):

| cell | recall edge-touching | recall not edge-touching |
|---|---|---|
| model_a__synth_test_pos | 1.000 | 1.000 |
| model_b__synth_test_pos | 0.594 | 0.478 |

Limitations: `synth_smoke_v1` edge-touch rate ~0.76 vs Boreal ~0.30 (documented, not corrected); the Boreal test set has no usable negative images; Model A is a baseline, not evidence the synthetic domain is optimal.
