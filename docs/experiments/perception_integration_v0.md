# Perception integration v0: YOLO on a virtual nadir camera in the SITL loop (wiring and resources; NOT a Level 3 result)

Date: 2026-11-02. Branch `integration/yolo-perception-prep`. No training, no paid compute. Frozen Boreal v2 and Level 2 files untouched.

**What this is:** an end-to-end working path from a UAV pose to a scored detection, plus the resource measurements needed to decide how many UAVs this VM can carry.
**What this is not:** evidence about coverage, time to detection, misdirection or any difference between Model A and Model B. Nothing below supports such a claim; the single-UAV run is one run, one scenario, one seed.

## 1. What was built
```
ArduCopter SITL --MAVLink--> swarm_agent.run_perception (rate_hz, wall clock)
   Pose(lat, lon, alt, heading)  --HTTP POST /frame (own pose only)-->  fog process: sim/camera_service.py
        FrameSynth.render(pose, SmokeSource list)   <-- sources made from fog's BURNING cells (truth stays in fog)
   <-- JPEG --  perception/frames.py::HttpFrameSource
   YOLO predict (Ultralytics, CPU, imgsz 640, conf>=0.25)  -> pixel boxes
   Camera.from_pose(...).unproject_ground(box centre) -> GridSpec.cell_of -> Detection(i, j, ...)   [GridSpec from grid.cfg: i east, j north]
   -> DETECTION message -> Fog.ingest_detection (scored at ingest) -> detections_<tag>.csv; PriorityPlanner.on_detection
```
- `sim/frame_synth.py`: the virtual camera. Subclasses synthgen's `Renderer` (unchanged, frozen) and replaces its `_smoke` hook so smoke is a set of world-placed plumes projected through `sim/camera.py`. Environment (texture, sun, haze, grade) comes from the **test-split** banks, seeded by a scenario seed, so two models in one scenario see identical pixels for identical poses.
- `sim/camera_service.py`: fog-side HTTP service; returns JPEG only; no endpoint exposes sources, cells or fire state. Off unless `perception.camera_service.enabled`.
- `perception/frames.py`, `perception/detector.py`: agent side. `YoloPerception` verifies the weights' SHA-256 (registry names are built in; other files need `weights_sha256` or `allow_unverified`), takes the camera spec and box-to-cell anchor from config, and raises under `perception.strict`; otherwise `from_config` still falls back to `NullPerception` with a warning. Default checkpoint: `local_data/backups/boreal_v2/real_boreal_v2_best_seed0.pt` (never committed).
- `fog/run_log.py` manifest now records weights SHA-256, camera/service/plume settings, oracle flag, python/ultralytics/torch/numpy/cv2 versions and the git commit.
- `eval/level3_pairs.py`: generates one config per (scenario seed, arm) and refuses a set in which arms differ in anything but the perception arm; `run` executes one config (SITL -> agents -> fog, fog started only after the agents report READY) and cleans up only processes it started.
- Tests: `tests/test_perception_integration.py` (transforms, box->cell, truth isolation, hash checks, baseline unchanged, renderer determinism, manifest) and `tests/test_level3_pairs.py`.
- Settings are provisional, not scientific: **nadir** camera (pitch 90, hfov 78 deg horizontal, 1280x720), **conf 0.25**, rate 0.25 Hz, altitude 120 m (inside the synthetic training range 60-320 m). Plume geometry (80 m tall, 20 m spread, 260 puffs, one plume per burning cell) is a simulation parameter with no empirical basis.

## 2. CPU benchmark (this VM: 1 vCPU AMD Ryzen AI 5 340, 1968 MB RAM + 2 GB swap, about 640 MB in use at idle; Python 3.10.18, torch 2.8.0+cpu, ultralytics 8.4.173, 1 torch thread)
`eval/bench_perception_cpu.py`, raw lines in `perception_bench_2026-11-02.jsonl`; 16 frames per model (8 over the burning cells at 120 m, 8 empty), scenario seed 1.

| Arm | SHA-256 (first 8) | YOLO ms (median) | Render ms (median) | Peak RSS MB (render+model) | Boxes on 8 fire frames | Boxes on 8 empty frames |
|---|---|---|---|---|---|---|
| B2s0 | `090c0f4c` | 306.1 | 506.6 | 735 | 4/8 | 0/8 |
| B2s1 | `5a9616ce` | 297.6 | 496.6 | 740 | 4/8 | 0/8 |
| B2s2 | `c4d443a8` | 300.9 | 520.8 | 739 | 4/8 | 0/8 |
| As0 | `1da10c48` | 311.7 | 512.6 | 738 | 8/8 | 0/8 |
| As1 | `31c4528e` | 304.2 | 505.1 | 738 | 8/8 | 0/8 |
| As2 | `3d5a3168` | 310.7 | 520.5 | 737 | 8/8 | 0/8 |
Reading: YOLO is about 0.3 s and rendering about 0.5 s per frame on the one core; one process holding the model and the renderer peaks near 740 MB. The model-only part is 431 MB. The box counts are a wiring smoke test on synthetic frames: Model A is in-domain and Model B cross-domain for these renders (Level 2: Real->Synth), so the difference mirrors the known domain gap and says nothing about missions.

## 3. Single-UAV end-to-end demonstration (one run)
`exp_swarm_perception_single.yaml`, Model B v2 seed 0, 1 UAV, 120 m, fire from cell (5,5), oracle off, 420 s after the UAV reported READY; results in `perception_single_demo_2026-11-02/`.
- 102 frames, 51 detections, 46 scored true and 5 false at ingest (precision 0.90), first true detection 24 s after ignition (`detection_lag_s` 24.0); infer median 360 ms in situ (max 1.9 s under CPU contention). The UAV's INVESTIGATE mode was driven by these detections.
- **Not comparable to anything**: one seed, no baseline arm run alongside, a scenario chosen so the fire is next to the takeoff point, the placeholder planner, 5 % of cells visited.
- An earlier attempt (fog started 75 s before the UAV was airborne) lost its fire before the UAV arrived and scored 0 true / 14 false; this is a scenario-start problem, not a detector one (finding 3).

## 4. Findings that matter for Level 3
1. **Nadir is outside the training pose distribution** (synth_smoke_v1 pitch 28-89 deg, mostly oblique/horizon views). Detection quality from nadir is therefore not what Level 2 measured. A different pitch is a config change (`perception.camera.pitch_deg`; the box anchor switches to the plume base automatically below 80 deg) but must be decided once for all arms, not tuned per model.
2. **A bright lake patch at the image edge draws boxes** (0.40-0.57) from Model B seed 0 on an otherwise empty frame (`perception_single_demo_2026-11-02/empty_nadir_frame_lake_false_positive.jpg`). Such false positives are exactly the misdirection signal Level 3 wants to measure; the 5 false detections in the demo are consistent with this but were not individually traced.
3. **Fire clock vs UAV boot:** fog starts the fire at its own t=0 while a UAV needs about 75 s to be ready. `level3_pairs.py run` starts fog after READY; a multi-UAV protocol needs an explicit common start (e.g. fog gating ignition on all agents READY).
4. **The fire is too fast for a 10x10 grid** at these parameters (p_ignite 0.02, burn 240 s still burns 76-100 % of cells within 420 s). Scenario design (spread rate, ignition location away from takeoff) is open.
5. **Placeholder planner:** the UAV stayed in INVESTIGATE (5 % visited); any baseline-vs-AI difference will partly measure `PriorityPlanner` placeholder behaviour (CLAUDE.md: replacement waits for the sensitivity/specificity from the eval matrix).
6. **Baseline arm = no perception.** It has no detections, so detection metrics do not exist for it; its comparison with AI arms is coverage-based. Model A vs Model B arms share every other setting.
7. `sim/camera.py` imports `fog.state`, which imports `wildfire_sim`; the agent process therefore loads FireSim's module transitively (never instantiates it). The tests forbid any direct import and truth-bearing attribute use in the agent-side perception modules; moving `cell_index` out of `fog/state.py` would remove the chain (not done: it touches the frozen projection code path).
8. Assumed (not documented in the repo): the repo's `fov_deg: 78` is horizontal; it matches the generator's horizontal 72-86 range.

## 5. Resources for multi-UAV (this VM is insufficient)
Measured: one UAV (fog + renderer + agent with model + SITL) used about 1.35 GB of 1.97 GB at 90 s (idle baseline 0.64 GB). Per additional UAV with an in-process model: about +0.45 GB (model, measured 431 MB) and about 0.2 CPU-seconds per second at 0.25 Hz (0.8 s per frame), plus SITL per vehicle (unmeasured). Estimate [INFERRED] for 4 UAVs: about 4 GB RAM and at least 3-4 vCPU, i.e. not feasible on 1 vCPU / 2 GB.
Options, none applied: (a) give the VM at least 4 vCPU / 8 GB (recommended; the methodology stays unchanged); (b) a shared inference process (one model copy) to save about 0.45 GB per UAV (not implemented); (c) a lower `rate_hz` applied identically to every arm, which changes observation density; (d) running on a RunPod CPU pod (needs an ArduPilot SITL build there; not evaluated); (e) `--speedup` < 1 is allowed by CLAUDE.md but agent perception rate is wall-clock, so it would need a sim-time rate (not implemented).

## 6. Recommended next experiment
Before any paired Level 3 matrix: (1) enlarge the VM or pick option (b); (2) decide the camera pose once (nadir at 120 m is documented here, but see finding 1) and the detection threshold on a calibration scenario set, never on the evaluation scenarios; (3) design the fire scenario (finding 3-4) and add the common-start gate; (4) run `level3_pairs.py plan` for the 7 arms x N scenario seeds with 1 UAV first (about 12 min per run on this VM including boot), then 2-4 UAVs; (5) analyse with the pre-declared metrics from `level3_observation_contract.md`. Until those runs exist, no statement about improved coverage, detection time or mission performance is supported.
