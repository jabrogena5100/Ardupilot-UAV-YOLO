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
- disjoint ignition cells (calibration i+j even, held-out i+j odd; distinct within a set; held-out >= 3 cells from takeoff; calibration cells are version 2, section 12);
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

What this does and does not show: it shows the full pipeline (start gate, scoring at ingest, provenance, cleanup) runs end to end with no leftover processes; it does not show anything about perception-driven monitoring. **Do not run the held-out set with this scenario.** (Superseded by sections 12-13: two corrections, then a re-pilot.)
Options (a methodology change must be fixed before any held-out run and re-piloted on calibration scenarios): shrink the monitored area or ignite within the first sweep rows; extend T; cruise faster; or a prior-free search pattern that reaches the fire in T for the baseline with probability well below 1 (so detection time is informative).

## 11. Cost of the full paired evaluation (measured, not extrapolated beyond the arithmetic)
One run = about 715 s wall on this 1 vCPU / 2 GB VM (T = 600 s plus about 100 s start gate), about 740 MB peak (benchmark), CPU inference about 0.3 s YOLO + 0.5 s render per frame, no GPU, no paid resource. Arms: baseline + 3 Model A + 3 Model B = 7;
8 held-out scenarios: 56 runs, about 11.1 h sequential (one run at a time on this VM). Not parallelisable here (1 core). A second pilot of the revised scenario (7 arms x 2 calibration scenarios = 14 runs) is about 2.8 h.

## 12. Corrections after the non-informative pilot (2026-11-02)
**12.1 Geofence bug (found from the pilot logs).** `Agent.geofence_push` returned `-3 * push`, where `push` already points inward, so a vehicle outside the fence was driven further out. In the first pilot the baseline reached corner cell (0,0) at about 240 s, left the box
and sat in RECENTER with the pushback driving it outward for the remaining 360 s (2,388 GEOFENCE events, 261 of 564 HUD lines RECENTER; coverage stuck at 15 %). Fixed (sign), regression test `tests/test_geofence.py` (fails on the old code). This affected every arm identically, so section 10's
"identical coverage" was partly this bug, not only geometry. Post-fix baseline: 22 % coverage in 600 s, 472 GEOFENCE events (it still touches the fence on edge-to-edge sweeps) and 8 RECENTER lines.

**12.2 Calibration ignition v2 (camera-geometry rule, fixed before the pilot).** `eval/level3_visibility.py` projects every cell through `sim/camera.py` along a flown trajectory (from the telemetry log) and counts perception frames (every 4 s) in which the plume base is inside the image with a
>= 40 px margin and the 80 m plume is >= 40 px tall. The baseline ignores fire, so one recorded baseline flight (scenario 1001 config, post-fix; `local_data/level3/pilot2/vis_baseline_postfix.json`) gives the visibility of any cell for any ignition. Rule: even parity (disjoint from held-out), Chebyshev distance >= 2 from takeoff (not at takeoff),
first visible at 60-300 s and >= 4 visible frames. Exactly four cells qualify: (2,2) first visible at 96 s (5 frames), (3,1) 116 s (6), (1,1) 188 s (10), (0,0) 212 s (13), assigned to seeds 1001, 1002, 1003, 1005 in that order (`level3_scenarios.CAL_IGNITION_V2`). Camera, speed, altitude, area and T = 600 s unchanged. Caveats: one trajectory sample; the same cell was first visited at 101.5 s and 122.8 s in two nominally identical baseline flights (SITL timing jitter of about 20 s, n = 1 pair).
**Held-out ignitions are unchanged** (pinned, hash-checked) and are still the old random cells; they must be re-derived by the same rule from a baseline flight, as an explicit step, before any final evaluation (open blocker B1).

**12.3 Fire settings kept frozen (p_ignite 0.0025, burn_steps 300).** With ignition v2 the old admissibility rule finds no admissible candidate, only because scenario 1005 (corner ignition) has 1 burning cell at 60 s (rule: >= 2). All four scenarios burn 16-28 % of cells and keep a burning cell at T; the pilot scenarios 1001 and 1002 meet every criterion. The rule and settings were not loosened or re-tuned; the exception is asserted in `tests/test_level3_calibration.py`.

**12.4 Model B threshold at the grid edge.** All three Model B seeds had selected 0.10, the lowest grid value. `eval/level3_boundary_check.py` re-ran the same checkpoints on the same cached calibration frames at a 0.01 floor and applied the same F0.5 rule to an extended grid: B2s0 -> 0.01 (F0.5 0.784, was 0.722 at 0.10), B2s1 -> 0.01 (0.741, was 0.654), B2s2 -> 0.02 (0.869, was 0.806). F0.5 rises monotonically as tau falls, so 0.10 was a grid artefact. Two seeds are again at the edge (0.01 is the detector's practical floor), so their optimum is untestable below. At those thresholds Model B false-alarms on 11-19 % of empty calibration frames (0.06-0.10 false detections per frame). Amendment recorded in `eval/level3_thresholds.json`
(`amendment_1`, `sweep_extended`); made after seeing the boundary selection, calibration scenarios only.

## 13. Corrected calibration pilot (2026-11-02; scenarios 1001 and 1002 only; held-out never run)
Single UAV, T = 600 s, baseline vs Model A seed 0 (tau 0.45) vs Model B v2 seed 0 (tau 0.01); raw runs `local_data/level3/pilot2/` (gitignored), per-run record `eval/level3_pilot_report.py` (pilot_report.json), all six clean exits, no leftover processes, wall 706-710 s each. n = 1 run per cell: descriptive, no test.

| arm | scen (ignition) | first smoke visible (geometry) s | visible frames | first true det s | true / false det | first false det s | ignition cell first visited s | coverage | time in INVESTIGATE s | first investigate targets |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1001 (2,2) | 120 | 60 | none (no detector) | 0 / 0 | - | 122.8 | 0.21 | 0 | - |
| A0 | 1001 | 124 | 59 | 123.2 | 83 / 0 | - | 126.9 | 0.20 | 223 | (2,2) (3,1) (2,1) (1,1) |
| B2s0 | 1001 | 285 | 26 | 248.0 | 31 / 18 | 14.9 | never | 0.14 | 202 | (4,4) (4,3) (5,3) (4,2) |
| baseline | 1002 (3,1) | 116 | 54 | none | 0 / 0 | - | 161.6 | 0.22 | 0 | - |
| A0 | 1002 | 131 | 80 | 122.2 | 110 / 0 | - | 136.5 | 0.17 | 289 | (3,1) (2,2) (2,1) (2,0) |
| B2s0 | 1002 | 204 | 81 | 199.4 | 129 / 14 | 9.4 | 211.5 | 0.17 | 388 | (5,4) (5,5) (3,1) (2,2) |

Reading (what the data support, and what they do not):
- **Geometry gives the baseline a real chance:** the flown camera had the plume base in view in 54-60 baseline frames, first at 116-120 s, and the baseline UAV flew over the ignition cell at 123 s / 162 s. So the earlier null was not a detector failure and is no longer a geometry failure. The baseline has no detections by design; its monitoring signal is coverage-based first look, which is a different quantity from detection time.
- **Differences arise from YOLO plus the route changes it triggers, not from the patrol geometry:** all arms share scenario, planner and the same early patrol until the first detection. Model A's first true detection (123 s, 122 s) coincides with first geometric visibility (the geometric criterion is conservative and the clock alignment is about +-10-20 s, which is why some gaps are slightly negative). Model B's first alarms were false (at 15 s and 9 s, away from the fire: targets (4,4), (4,3), (5,3) / (5,4), (5,5)); it spent 202-388 s in INVESTIGATE, flew the fire area later (its first geometric visibility 285 s / 204 s versus 120 s / 116 s for the baseline) and its first true detection came 125 s after Model A's in scenario 1001 and 77 s after in 1002. Coverage differs by arm (0.14-0.22).
- **Perception-driven route changes exist and are large:** 36-68 % of HUD time in INVESTIGATE in the perception arms, none in the baseline.
- **Not supported:** any claim of improvement or ranking. n = 1 per cell, two scenarios, run-to-run timing jitter of about 20 s (section 12.2) is as large as some of the differences (ignition-cell first-visit differences of 4 s, 25 s and 50 s are within or near it); "true detection" uses a one-cell tolerance, so a mislocalised detection next to the fire counts; Model A is in-domain and Model B cross-domain (the gap largely restates Level 2 Real->Synth); Model B's thresholds sit at the detector floor.

**Verdict:** the corrected configuration is non-trivial and measurable: it produces detections, false alarms and detection-induced route changes that differ by arm. It is fit to proceed to a larger calibration-only replication, not to a performance claim.

## 14. Blockers and decisions still open
- **B1 held-out ignition cells** are unconverted (random cells, section 12.2): re-derive by the section 12.2 rule from a baseline flight before the final evaluation; this changes the pinned held-out file and its hash (a deliberate, recorded step). Calibration seeds 1003 and 1005 have not been flown.
- **B2 repeatability:** about 20 s timing jitter between identical flights; a repeat-run estimate (same arm and scenario, several repeats) is needed to size effects.
- **B3 Model B thresholds** at the detector floor for 2 of 3 seeds; planner receives near-all low-confidence boxes; consider reporting a second operating point (e.g. 0.10) as sensitivity, not as a tuned alternative.
- **B4 truth radius** (one cell) can credit mislocalised detections.
- **B5 sample size:** 4 calibration / 8 held-out scenarios, 3 seeds per family.
Runtime of the full paired evaluation is unchanged (section 11: 56 runs x about 710 s = about 11 h on this VM); a calibration-only replication of the remaining scenarios (1003, 1005) for the three pilot arms is 6 runs, about 71 min.

## 15. Pilot 2 completion and final protocol freeze (2026-11-09)
**15.1 Pilot 2 completed.** Scenarios 1003 and 1005 were flown for baseline, A0 and B2s0 (6 runs, all clean exits, no leftover processes); with 1001 and 1002 that is 12 runs, 4 scenarios x 3 arms. Analysis: `eval/level3_analyze.py` (raw runs and `analysis/` under gitignored `local_data/level3/pilot2/`). Descriptive means over 4 scenarios (n = 1 run per cell):

| family | first true det (s) | first visit of ignition cell (s) | coverage at 600 s | false det / run | false-priority diversion (HUD s) | true diversion (HUD s) |
|---|---|---|---|---|---|---|
| baseline | no detector | 172.8 | 0.218 | 0 | 0 | 0 |
| A0 | 174.0 | 233.9 | 0.193 | 0.25 | 0 | 224 |
| B2s0 | 220.8 | 435.8 | 0.170 | 18.3 | 23.8 | 258 |

Confirmed from artifacts: detections change priority and trajectory (INVESTIGATE episodes, 7-12 distinct targets per perception run, none in the baseline) and are scored against fog ground truth at ingest. Observation: both perception arms reached the ignition cell *later* than the baseline, because the investigate planner leaves the patrol for detected cells (and for Model B, for false alarms). With n = 4 the smallest possible sign-flip p-value is 0.125; nothing here is significant and the pilot is not a result.
**Metric corrections made during analysis:** (a) the first draft's "first look" metric (first visit minus ignition, clipped at 0) was 0 in all runs because fog records a cell's first visit even when it precedes ignition; replaced by the first visit of the *ignition cell* (burning from t = 0). The "fraction of fire cells visited" is likewise inflated by pre-ignition visits (baseline 0.5-0.83) and is descriptive only. (b) Contrasts of first true detection against the baseline are undefined (no detector) and are not computed. (c) Diversion truth is time-agnostic (target within Chebyshev 1 of any cell that burned during the run), so false diversion is a lower bound.
**15.2 Blocker B1 resolved (before any held-out run).** Held-out ignition cells re-derived by the section 12.2 rule from the same recorded baseline flight (odd parity; window widened from 60-300 s to 60-400 s because only 3 odd cells qualify at 300 s): (3,2), (2,1), (1,0), (0,1). The 8 held-out scenarios reuse each cell twice with different fire seeds and different texture slots (`HELD_IGNITION_V2`); the "distinct cells within the set" control is relaxed accordingly and the superseded v1 hash (558f7282...) is recorded in `eval/level3_scenarios.json` (`amendment_1`). The held-out set had never been flown.
**15.3 Frozen for the final evaluation:** fire (`eval/level3_fire.json`), thresholds (`eval/level3_thresholds.json`: A0 0.45, A1 0.30, A2 0.35, B2s0 0.01, B2s1 0.01, B2s2 0.02), camera (60 deg, 78 deg HFOV, 120 m), planner (cooldown 60 s), single UAV, T = 600 s, held-out scenarios, 7 arms, checkpoints by registry hash. Primary metrics: first true detection (A vs B), first visit of the ignition cell, coverage, false detections, false-priority diversion seconds. Unit of analysis is the scenario (family value = mean over seeds); sign-flip p and bootstrap CI over 8 scenarios; seed SD reported separately. B2 (run-to-run jitter) is addressed by two repeat runs (baseline and A0, scenario 2001). B3-B5 remain limitations.

**15.4 Final evaluation executed (2026-11-09).** Protocol unchanged; 56 runs plus 2 repeats completed and analysed: `level3_final.md`.
