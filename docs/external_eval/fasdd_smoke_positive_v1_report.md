# External evaluation report: fasdd_smoke_positive

2149 images; six frozen checkpoints; operating point conf 0.25, IoU 0.5; Model A = synthetic-trained, Model B = Boreal-v2-trained.

**Overall (pooled images)**

| metric | Model A mean (SD) | Model B mean (SD) | B - A |
|---|---|---|---|
| mAP50 | 0.408 (0.010) | 0.224 (0.034) | -0.184 |
| mAP50-95 | 0.216 (0.009) | 0.125 (0.018) | -0.091 |
| precision | 0.459 (0.025) | 0.626 (0.063) | 0.167 |
| recall | 0.412 (0.008) | 0.202 (0.046) | -0.210 |
| f1_max | 0.482 (0.020) | 0.311 (0.048) | -0.171 |
| mean_conf_tp | 0.898 (0.019) | 0.700 (0.008) | -0.198 |
| loc_mean_iou_tp | 0.780 (0.012) | 0.780 (0.008) | 0.000 |
| detection_rate | 0.550 (0.010) | 0.263 (0.059) | -0.288 |

## By platform_by_size

**ground_or_low_video_720p (1125 images)**

| metric | Model A mean (SD) | Model B mean (SD) | B - A |
|---|---|---|---|
| mAP50 | 0.697 (0.015) | 0.256 (0.071) | -0.441 |
| mAP50-95 | 0.339 (0.021) | 0.091 (0.034) | -0.248 |
| precision | 0.776 (0.039) | 0.637 (0.089) | -0.139 |
| recall | 0.637 (0.016) | 0.230 (0.093) | -0.407 |
| f1_max | 0.706 (0.016) | 0.353 (0.088) | -0.353 |
| mean_conf_tp | 0.894 (0.021) | 0.586 (0.039) | -0.307 |
| loc_mean_iou_tp | 0.748 (0.014) | 0.691 (0.033) | -0.058 |
| detection_rate | 0.819 (0.019) | 0.288 (0.116) | -0.531 |

**tower_fixed_camera (718 images)**

| metric | Model A mean (SD) | Model B mean (SD) | B - A |
|---|---|---|---|
| mAP50 | 0.011 (0.004) | 0.003 (0.003) | -0.008 |
| mAP50-95 | 0.005 (0.001) | 0.001 (0.002) | -0.004 |
| precision | 0.033 (0.006) | 0.037 (0.025) | 0.005 |
| recall | 0.046 (0.001) | 0.005 (0.003) | -0.041 |
| f1_max | 0.045 (0.005) | 0.016 (0.004) | -0.030 |
| mean_conf_tp | 0.599 (0.041) | 0.518 (0.034) | -0.081 |
| loc_mean_iou_tp | 0.746 (0.040) | 0.670 (0.071) | -0.076 |
| detection_rate | 0.048 (0.001) | 0.006 (0.003) | -0.042 |

**uav_nadir_1080p (306 images)**

| metric | Model A mean (SD) | Model B mean (SD) | B - A |
|---|---|---|---|
| mAP50 | 0.437 (0.043) | 0.423 (0.034) | -0.014 |
| mAP50-95 | 0.362 (0.042) | 0.330 (0.023) | -0.033 |
| precision | 0.721 (0.016) | 0.855 (0.026) | 0.134 |
| recall | 0.333 (0.001) | 0.356 (0.003) | 0.023 |
| f1_max | 0.463 (0.005) | 0.514 (0.008) | 0.050 |
| mean_conf_tp | 0.961 (0.013) | 0.842 (0.007) | -0.119 |
| loc_mean_iou_tp | 0.912 (0.000) | 0.890 (0.009) | -0.022 |
| detection_rate | 0.742 (0.003) | 0.772 (0.010) | 0.031 |

## Cluster-level uncertainty (dHash64 scene clusters; paired bootstrap of Model B minus Model A on per-cluster mAP50)

- cluster_r6/all: 680 clusters, gate interval_and_test; B - A = -0.241, 95% interval [-0.270, -0.213], P(diff>0) 0.000; sign-flip p 0.0000; leave-one-cluster-out [-0.242, -0.240]
- cluster_r6/tower_fixed_camera: 196 clusters, gate interval_and_test; B - A = -0.029, 95% interval [-0.047, -0.014], P(diff>0) 0.000; sign-flip p 0.0000; leave-one-cluster-out [-0.031, -0.024]
- cluster_r6/ground_or_low_video_720p: 274 clusters, gate interval_and_test; B - A = -0.552, 95% interval [-0.598, -0.505], P(diff>0) 0.000; sign-flip p 0.0000; leave-one-cluster-out [-0.554, -0.550]
- cluster_r6/uav_nadir_1080p: 212 clusters, gate interval_and_test; B - A = -0.036, 95% interval [-0.059, -0.014], P(diff>0) 0.001; sign-flip p 0.0024; leave-one-cluster-out [-0.040, -0.034]
- cluster_r4/all: 897 clusters, gate interval_and_test; B - A = -0.274, 95% interval [-0.299, -0.249], P(diff>0) 0.000; sign-flip p 0.0000; leave-one-cluster-out [-0.275, -0.273]
- cluster_r2/all: 1232 clusters, gate interval_and_test; B - A = -0.286, 95% interval [-0.308, -0.264], P(diff>0) 0.000; sign-flip p 0.0000; leave-one-cluster-out [-0.287, -0.285]

