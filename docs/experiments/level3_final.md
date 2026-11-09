# Level 3 final paired evaluation (2026-11-09)

**Purpose.** Does the training domain of the detector (Model A synthetic, Model B real/Boreal) change autonomous monitoring behaviour on ArduPilot SITL, against a no-detector patrol baseline? Protocol, frozen before any held-out run: `level3_methodology.md` section 15.3.

**Design.** 8 held-out scenarios (2001, 2002, 2004, 2005, 2006, 2012, 2014, 2015) x 7 arms (baseline; A0, A1, A2; B2s0, B2s1, B2s2) = 56 runs, T = 600 s, single UAV, oracle off. Run on the CPU-only VM (no paid compute), 2 parallel slots via `eval/level3_queue.py`. Unit of analysis is the scenario (family value = mean over seeds); CIs are percentile bootstrap over scenarios, p is the exact two-sided sign-flip test (minimum possible p with n = 8 is 0.0078).

**Completion audit (after a VM reboot, 2026-11-09 21:43 UTC).** No arducopter/mavproxy/fog/agent/queue processes and no tmux server were running; the queue log ends `queue finished: 31 run(s), 0 not clean` at 13:05 UTC, before the gap. All 56 `results/*/run_meta.json` have `fog_exit_code` 0 and empty `leftover_after_cleanup`; wall times are 700-725 s, so no run shows suspend distortion and none was quarantined. Provenance caveat: runs were made with commit `0c2f6c6` plus uncommitted harness edits (`git_dirty: true` in every `run_meta.json`); those edits are committed afterwards and not altered.
The first attempt at the baseline-2001 repeat exited rc=1 after 1 s (queue log, 09:34 UTC); the repeat was re-run (09:47-09:59) and is clean. The failed attempt's cause was not investigated.

## Results (mean over 8 scenarios; family = mean of seeds)

| metric | baseline | A | B | A - B (95% CI, p) |
|---|---|---|---|---|
| first true detection (s, censored at 600) | no detector | 177.1 | 296.6 | -119.5 (-173.9, -72.4), p=0.0078 |
| first visit of ignition cell (s) | 221.4 | 268.0 | 487.7 | -219.7 (-279.8, -157.3), p=0.0078 |
| grid coverage at 600 s | 0.209 | 0.183 | 0.148 | +0.035 (0.025, 0.047), p=0.0078 |
| false detections / run | 0 | 0.29 | 55.7 | -55.4 (-87.2, -33.6), p=0.0078 |
| false-priority diversion (HUD s) | 0 | 1.6 | 80.6 | -79.0 (-119.3, -41.6), p=0.016 |
| true diversion (HUD s) | 0 | 173.7 | 237.1 | -63.4 (-148.5, 15.5), p=0.23 |

Against the baseline: ignition-cell first visit A +46.6 s (0.2, 97.3; p=0.117, not significant), B +266.3 s (214.9, 318.8; p=0.0078); coverage A -0.026, B -0.061 (both p=0.0078); false detections B +55.7 (p=0.0078), A +0.29 (p=0.25).
Seed spread (per-arm SD over 8 scenarios) is in `level3_final/stats.json` (`per_arm`); seeds within a family agree in direction (e.g. first true detection A0/A1/A2 169/183/180 s; B2s0/s1/s2 283/350/257 s). Bootstrap CIs describe scenario sampling only.
Scenario 2004 Model B seed 1 (B2s1) made 305 false detections and **no true detection** (censored at 600 s in the family mean, which makes B's first-detection mean a lower bound for that seed).

## Repeatability (B2): one repeat each, scenario 2001
| run | first true det (s) | ignition-cell first visit (s) | coverage | true diversion (s) |
|---|---|---|---|---|
| baseline orig / repeat | n/a | 87.0 / 100.1 | 0.22 / 0.18 | 0 / 0 |
| A0 orig / repeat | 101.2 / 114.9 | 108.3 / 122.0 | 0.21 / 0.19 | 59 / 43 |
Same scene, fire and detector; control timing differs: first-visit and first-detection times moved by 13-14 s, consistent with the roughly 20 s jitter noted earlier. n = 1 repeat per config, so this is an indication of size, not an estimate of variance.

## Interpretation (limited to what was measured)
- Detections change priority and trajectory in both perception arms, and both arms covered less of the grid than the plain patrol. The planner leaves the patrol to investigate targets, so a detector does not help find the ignition cell sooner than the baseline here: the planner is a placeholder (`planning/priority.py`).
- Model A (in-domain) detected sooner, produced almost no false alarms and reached the ignition cell sooner than Model B (cross-domain). This largely restates the Level 2 Real->Synth gap (Model B was evaluated on synthetic imagery); it is not evidence about real flights.
- True diversion did not differ significantly between A and B; B's extra cost is false-priority diversion and lost coverage.

## Limitations
Smoke is a flat billboard (no volume or 3D occlusion); flat ground, one pinhole camera, one UAV; 8 scenarios and 3 seeds per family; ignition cells reuse 4 cells x 2 fire seeds; Model B thresholds sit at the detector floor for 2 of 3 seeds (B3); detection truth radius of one cell (B4); diversion truth is time-agnostic so false diversion is a lower bound; wall-clock SITL timing means runs are not bit-reproducible; first-detection means are censored at 600 s; many tests on one small sample, no multiplicity correction.

## Artifacts
Raw runs (2.8 GB, gitignored): `local_data/level3/final/{results,results_repeat}`, queue logs `queue.log`, `queue_logs/`. Committed small outputs: `docs/experiments/level3_final/{per_run.csv,stats.json,repeats.json,fig_level3_scenarios.png,matrix.json}`. Reproduce: `python3 eval/level3_analyze.py local_data/level3/final/results/* --out OUTDIR`.
Spend: none (no RunPod resources used).
