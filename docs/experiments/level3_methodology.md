# Level 3 methodology (calibration and evaluation protocol)

Written 2026-11-02 on branch `integration/yolo-perception-prep`, **before** the detector calibration frames were rendered. Rules below are fixed in the code
(docstrings of the named files); where a rule was changed after seeing data it says so. Nothing here is a Level 3 result.

## 1. Question and what Level 3 may claim
Level 2 measured box-level perception per training domain. Level 3 asks whether the difference propagates into **autonomous monitoring**: detection, false alarms and
monitoring behaviour of a UAV whose planner is the same in every arm. A statement about improved coverage, detection time or misdirection is allowed only from the
sealed held-out evaluation (section 3), with all seeds, and with the limitations of section 9. The calibration pilot (section 8) can only show whether the
configuration is non-trivial and measurable.

## 2. Arms and what differs between them
`baseline` (perception off), Model A seeds 0-2, Model B v2 seeds 0-2 (`eval/level3_pairs.py::ARMS`, registry hashes verified at load). Between arms of one scenario
exactly five perception keys may differ (`enabled`, `weights`, `weights_sha256`, `model_tag`, `conf_threshold`), and `conf_threshold` must equal the arm's calibrated value;
`level3_pairs.assert_paired` refuses anything else (fire, seeds, camera, rate, altitude, speed, yaw behaviour, planner settings are identical). Every arm, baseline included,
runs the same planner (section 6) and the same yaw-to-track flight behaviour, so the only difference is whether detections exist.

## 3. Scenario sets and leakage control (`eval/level3_scenarios.py`, pinned in `eval/level3_scenarios.json`)
A scenario seed fixes the rendered environment, the FireSim seed, the ignition cell and the agents' seeds. 4 **calibration** scenarios (seeds 1001, 1002, 1003, 1005) and
8 sealed **held-out** scenarios (2001, 2002, 2004, 2005, 2006, 2012, 2014, 2015), chosen once, before any tuning:
- disjoint seeds; disjoint **ground-texture slots** (the 6-entry test-split texture bank is partitioned: calibration slots {0,1,2} (in use: 0, 1), held-out {3,4,5}), so no ground texture is rendered in both sets;
- disjoint ignition cells (calibration i+j even, held-out i+j odd; distinct within a set; all >= 3 cells from the takeoff cell);
- the held-out list is hash-pinned; calibration tools call `require_calibration` and refuse held-out seeds; `level3_pairs plan --split heldout` needs `--final-evaluation` and frozen thresholds/fire files;
- **not separated** (limitation): the 16 test-split smoke sprites, the sky model and the detector weights are shared; the weights were trained on neither renderer split's test images.
`tests/test_level3_calibration.py` re-verifies all of this from the pinned file.

## 4. Camera geometry (convention stated once)
`pitch_deg` = degrees **below the horizontal** of the optical axis (0 = horizon, 90 = nadir). Final proposed geometry: **60 deg**, **78 deg horizontal FOV**, 1280 x 720,
**120 m** above the launch point, flat ground, no roll, no lens distortion, **look direction = vehicle heading**; the vehicle yaws to its direction of travel at >= 2 m/s
(`perception.camera.yaw_to_track`, on in every arm; off by default elsewhere). With these numbers the image spans ground ranges of about 12 m (bottom row) to 168 m (top row) ahead.
Box -> cell: the ground point under the **bottom-centre** of the box (the plume base) through `sim/camera.py`, then `GridSpec.cell_of` (i = east, j = north). The earlier nadir geometry
was outside the training pose distribution (training pitch 28-89 deg, mostly oblique); 60 deg is inside it. The geometry is **not tuned**: it was fixed by decision before calibration.

## 5. Detection threshold: sweep and rule (`eval/level3_calibrate_detector.py`)
Per arm, tau is chosen on calibration scenarios only. Frames come from short passes (10 fire-approach passes and 10 free passes per scenario, 8 frames each, 4 s apart, 10 m/s, heading = travel
direction) over a FireSim stepped at the frozen settings; every arm runs on the same rendered frames at a 0.10 floor and tau in {0.10, 0.15, ..., 0.70} is applied offline.
Truth is simulation-side: a detection is TRUE when a burning cell lies within one cell (Chebyshev) of its cell at that instant (the rule `Fog.ingest_detection` uses); a frame is VISIBLE when
the renderer's smoke alpha (>= 0.08) covers >= 1 % of the frame (the dataset's own label-validity constants), EMPTY at 0, AMBIGUOUS in between (excluded from recall and from the empty-frame rate).
Reported per tau: precision, recall (visible frames with >= 1 true detection), false detections per frame, empty-frame false-alarm rate, detection latency (median seconds from the first visible
frame of a pass to its first true detection; undetected passes counted as censored).
**Predetermined rule (written before any frame was rendered):** per arm, tau = argmax F0.5 (precision-weighted: a false alarm misdirects the planner) over the grid, ties to the higher tau; if
the maximum is 0 the arm is flagged UNUSABLE and tau = 0.70. One tau per arm (not per seed-pooled family), recorded in `eval/level3_thresholds.json` with the whole sweep.

## 6. Planner (what flies the UAV; identical in every arm)
`swarm_agent.step_grid` order: hard recentering > claims of neighbours > **investigate evidence** > coverage frontier (`GridMission`, nearest unvisited cell, edge-to-edge TRAVERSE sweeps).
`planning/priority.PriorityPlanner` (deterministic; no randomness) turns detections into evidence per cell:
1. a detection with `conf < conf_threshold` is dropped by the planner (still broadcast and scored by fog); the planner threshold is the same `perception.conf_threshold` as the detector's, so there is no second hidden threshold;
2. evidence has a TTL (`detection_ttl_s`, 20 s), repeat sightings add 0.05 each; score = min(1, max conf + bonus);
3. `choose` picks the highest score, ties by Chebyshev distance to the UAV, and the target is **sticky** until its evidence expires or it is cleared;
4. on arrival (within 1 m of the cell centre) the cell is cleared. Original placeholder behaviour: the next frame re-flags the same cell, so the UAV can dwell on one fire (observed in the v0 demo: 5 % of cells visited in 420 s).
   **Level 3 setting:** `investigate_cooldown_s = 60`: for 60 s after an investigation new detections of that cell are ignored by the planner (fog still scores them). Default 0 = unchanged behaviour. Fixed a priori, same in all arms;
5. the baseline arm has no detections, so it never leaves the coverage frontier: the planner is the same code and the only difference is the information it receives. The principled replacement (log-odds map, `mapping/probability_map.py`) waits for cell-level sensitivity/specificity; not part of this protocol.
The planner sees only detections (own and neighbours') and, with `oracle_hazards: false` (required and asserted in every config), nothing about the fire.

## 7. Fire scenario (`eval/level3_fire_calibration.py`, frozen in `eval/level3_fire.json`)
The v0 fire saturated the grid. FireSim is stepped offline for the calibration scenarios over a candidate grid; the admissibility rule and its amendment history are in the file's docstring (rule version 1 found no admissible
candidate; version 2 was amended **after seeing the version-1 calibration sweep only**, held-out never run). Chosen: **p_ignite 0.0025, burn_steps 300 (fixed), von Neumann, no wind, T = 600 s**; calibration scenarios burn 17-24 % of cells by T and always have a burning cell at T
and >= 2 at 60 s. Frozen before any evaluation. Risk to report, not hide: spread is a steep percolation-like transition (0.3 -> 0.55 burned between p = 0.003 and 0.005), so held-out fires may vary more than calibration ones.

## 8. Run protocol (`eval/level3_pairs.py`)
Fog starts first and is the **run clock**: with `fire_model.start_gate` it ignites nothing, steps nothing and scores nothing until the required number of agents report altitude >= 100 m for 5 s; then it resets t = 0, ignites the scenario's cell
and starts the T = 600 s window (detections before the gate are counted and ignored). SITL and agents start after fog; every child runs in its own process group with `PYTHONHASHSEED=0`; cleanup kills only processes the run started and records leftovers;
a run refuses to start if SITL/agent/fog processes or the camera port are already present. Each run directory holds `run_meta.json` (git commit and dirty flag, config/fire/thresholds/scenario hashes, times), `manifest.json` (weights SHA-256, versions, camera, oracle flag), `start_gate.json`, fog's CSVs (including the new
`fire_cells_<tag>.csv`: ignition time and first-visit time per burned cell) and the agent logs. `eval/level3_summarize.py` makes one descriptive row per run.

## 9. Known limitations (carry into any write-up)
Smoke is stacked flat billboards with simulation-chosen plume geometry (80 m, one plume per burning cell); flat ground; one pinhole; synthetic imagery means Model A is in-domain and Model B cross-domain (Level 2 Real->Synth), so an arm difference largely restates the Level 2 gap
unless mission behaviour adds to it; 4 calibration / 8 held-out scenarios and 3 seeds per family give wide intervals; SITL wall-clock timing makes runs not bit-reproducible (the scene, fire and detector are deterministic; control timing is not); `cache=ram` deviation of Model B v2 (Level 2 docs); detection truth radius of one cell.

## 10. Calibration pilot (2026-11-02, calibration scenarios 1001 and 1002 only; held-out never run)
Single UAV, T = 600 s, 60 deg / 120 m geometry, frozen fire (section 7), thresholds from `eval/level3_thresholds.json` (A0 tau 0.45, B2s0 tau 0.10). Raw runs: `local_data/level3/pilot/` (gitignored); rows from `eval/level3_summarize.py`.

| arm | scenario | det_true | det_false | visited_frac | fire cells | fire cells visited | frames | wall_s |
|---|---|---|---|---|---|---|---|---|
| baseline | 1002 | 0 | 0 | 0.15 | 17 | 0 | - | 707 |
| A0 (Model A seed 0) | 1001 | 0 | 0 | 0.15 | 20 | 1 | - | 715 |
| A0 | 1002 | 0 | 0 | 0.15 | 17 | 0 | 103 | 715 |
| B2s0 (Model B v2 seed 0) | 1001 | 0 | 1 | 0.15 | 20 | 1 | 104 | 717 |
| B2s0 | 1002 | 0 | 9 | 0.15 | 17 | 0 | 94 | 714 |

Baseline 1001 has no valid run: attempt 1 exited at the start gate (kept in `pilot_attempt1_sync_discarded/`, the gate was fixed afterwards) and a re-run was interrupted by a host crash (no `experiment_end`). It is excluded, not imputed.

**Finding: the configuration is non-informative as piloted.** No arm produced a single true detection in 600 s, including Model A, which is in-domain and had precision = recall = 1.0 on the calibration frames. Coverage is identical across all arms (0.15) and the
UAV visited 0-1 of 17-20 fire cells. The only arm difference is B2s0 false alarms (1 and 9), which are misdirection risk, not detection. Cause (measured from the logs): the 10x10 grid is 1 mile across (cells about 161 m), the ignition
cells are 5-9 cells from takeoff, the grid-frontier sweep covers 15 % of cells in 600 s (about 46 % of HUD lines are RECENTER), and the 60 deg camera sees about 12-168 m ahead, so the coverage baseline does not come near the fire. The detector is not the limiting factor.
No planner effect (investigate mode) was exercised: `pct_investigate` is 0 for A0 and 12.6 for B2s0 only because of false alarms.

What this does and does not show: it shows the full pipeline (start gate, scoring at ingest, provenance, cleanup) runs end to end with no leftover processes; it does not show anything about perception-driven monitoring. **Do not run the held-out set with this scenario.**
Options (a methodology change must be fixed before any held-out run and re-piloted on calibration scenarios): shrink the monitored area or ignite within the first sweep rows; extend T; cruise faster; or a prior-free search pattern that reaches the fire in T for the baseline with probability well below 1 (so detection time is informative).

## 11. Cost of the full paired evaluation (measured, not extrapolated beyond the arithmetic)
One run = about 715 s wall on this 1 vCPU / 2 GB VM (T = 600 s plus about 100 s start gate), about 740 MB peak (benchmark), CPU inference about 0.3 s YOLO + 0.5 s render per frame, no GPU, no paid resource. Arms: baseline + 3 Model A + 3 Model B = 7;
8 held-out scenarios: 56 runs, about 11.1 h sequential (one run at a time on this VM). Not parallelisable here (1 core). A second pilot of the revised scenario (7 arms x 2 calibration scenarios = 14 runs) is about 2.8 h.
