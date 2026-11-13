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

## Two estimands (do not mix them)

- **Pooled AP** (headline above): all images of a subset ranked together, mean over seeds. No interval is computed for it.
- **Cluster-macro AP**: AP inside each dHash64 scene cluster, then the unweighted mean over clusters. Weighting and confidence ranking differ from pooled AP (AP is nonlinear), so its value is a different number, not a refinement of the pooled one.
- The bootstrap interval and sign-flip p below belong to the cluster-macro estimand only, are **conditional sensitivity analyses** (they assume the dHash groups behave as independent units, which is not demonstrated: similarity components are not videos, cameras or burn events), and are **not population-level inference**. Subsets the config restricts to descriptive reporting get point values only.
- Bootstrap fraction = share of resamples with B - A > 0; it is neither a posterior probability nor the sign-flip p-value.

| subset (cluster radius) | n clusters | pooled mAP50 A / B / B-A | cluster-macro mAP50 A / B / B-A | conditional interval (cluster-macro, percentile) | bootstrap B-A>0 | sign-flip |
|---|---|---|---|---|---|---|
| all (r6) | 680 | 0.408 / 0.224 / -0.184 | 0.523 / 0.282 / -0.241 | descriptive only (config restriction) | - | - |
| tower_fixed_camera (r6) | 196 | 0.011 / 0.003 / -0.008 | 0.044 / 0.015 / -0.029 | [-0.047, -0.014] | 0/10000 | p-hat = 1.5e-05 (Monte Carlo, n_mc = 200000) |
| ground_or_low_video_720p (r6) | 274 | 0.697 / 0.256 / -0.441 | 0.788 / 0.237 / -0.552 | descriptive only (config restriction) | - | - |
| uav_nadir_1080p (r6) | 212 | 0.437 / 0.423 / -0.014 | 0.623 / 0.587 / -0.036 | [-0.059, -0.014] | 11/10000 | p-hat = 0.0024 (Monte Carlo, n_mc = 200000) |
| all_excluding_ground_or_low_video_720p (r6) | 407 | 0.176 / 0.189 / +0.012 | 0.344 / 0.311 / -0.033 | [-0.047, -0.018] | 0/10000 | p-hat = 1.5e-05 (Monte Carlo, n_mc = 200000) |
| all (r4) | 897 | 0.408 / 0.224 / -0.184 | 0.545 / 0.271 / -0.274 | descriptive only (config restriction) | - | - |
| tower_fixed_camera (r4) | 236 | 0.011 / 0.003 / -0.008 | 0.050 / 0.016 / -0.034 | [-0.051, -0.020] | 0/10000 | p-hat = 5.0e-06 (MC floor, n_mc = 200000; not 0) |
| ground_or_low_video_720p (r4) | 416 | 0.697 / 0.256 / -0.441 | 0.780 / 0.226 / -0.554 | descriptive only (config restriction) | - | - |
| uav_nadir_1080p (r4) | 245 | 0.437 / 0.423 / -0.014 | 0.623 / 0.594 / -0.029 | [-0.051, -0.008] | 35/10000 | p-hat = 0.0068 (Monte Carlo, n_mc = 200000) |
| all_excluding_ground_or_low_video_720p (r4) | 481 | 0.176 / 0.189 / +0.012 | 0.342 / 0.310 / -0.032 | [-0.045, -0.019] | 0/10000 | p-hat = 1e-05 (Monte Carlo, n_mc = 200000) |
| all (r2) | 1232 | 0.408 / 0.224 / -0.184 | 0.562 / 0.276 / -0.286 | descriptive only (config restriction) | - | - |
| tower_fixed_camera (r2) | 318 | 0.011 / 0.003 / -0.008 | 0.053 / 0.017 / -0.036 | [-0.050, -0.023] | 0/10000 | p-hat = 5.0e-06 (MC floor, n_mc = 200000; not 0) |
| ground_or_low_video_720p (r2) | 640 | 0.697 / 0.256 / -0.441 | 0.787 / 0.265 / -0.522 | descriptive only (config restriction) | - | - |
| uav_nadir_1080p (r2) | 274 | 0.437 / 0.423 / -0.014 | 0.627 / 0.602 / -0.025 | [-0.046, -0.003] | 118/10000 | p-hat = 0.026 (Monte Carlo, n_mc = 200000) |
| all_excluding_ground_or_low_video_720p (r2) | 592 | 0.176 / 0.189 / +0.012 | 0.318 / 0.288 / -0.031 | [-0.043, -0.018] | 0/10000 | p-hat = 5.0e-06 (MC floor, n_mc = 200000; not 0) |

