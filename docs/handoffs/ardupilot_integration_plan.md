# Plan: integrating the Boreal v2 Model B checkpoints into the ArduPilot simulation

Written 2026-11-02. **Planning and documentation only. Nothing was implemented, no model was run, no Level 3 calibration was done.** Facts are tagged
**[VERIFIED]** (read from a file in this repo during this review), **[INFERRED]** (reasoned, not measured) or **[UNKNOWN]**. Companion docs:
`docs/experiments/boreal_v2_results.md` (what the models do), `docs/experiments/level3_observation_contract.md` (proposed Level 3 observation unit; not implemented), `docs/handoffs/checkpoint_backup.md`.

## 1. What exists today
- **[VERIFIED] The seam is built.** `perception/base.py` defines `Detection(i, j, cls, conf, bbox, frame_id, meta)`, `Pose(lat, lon, alt, heading_deg, cell, t, sysid)`, `NullPerception` (default) and
  `from_config(cfg, grid)`, which imports `perception.detector.YoloPerception` and **falls back to `NullPerception` with a printed warning on any failure** (missing weights, missing torch, bad config).
- **[VERIFIED] `perception/detector.py`, `sim/frame_synth.py` and `mapping/probability_map.py` are empty files (0 lines).** `YoloPerception` does not exist, so `perception.enabled: true` currently runs with no detections.
- **[VERIFIED] The agent hook exists.** `swarm_agent.py::run_perception` rate-limits to `perception.rate_hz` (default 1 Hz, independent of the 10 Hz control loop), builds a `Pose` from the vehicle state, calls
  `perception.observe(pose)`, broadcasts every detection as a `DETECTION` message (`comms/schema.make_detection`: sys, i, j, cls, conf, model, optional bbox/lat/lon/alt/frame) and feeds `PriorityPlanner.on_detection`.
- **[VERIFIED] The shared projection exists.** `sim/camera.py::Camera` (flat-earth pinhole; `unproject_ground`, `footprint`, `cells_in_footprint`, `bbox_ground_anchor`, `bbox_to_cell`) is the same code `synthgen/render.py` uses; `CameraIntrinsics.hfov_deg` is the horizontal FOV;
  `Camera.from_pose(...)` takes a `perception.base.Pose`. The renderer's domain randomisation uses `hfov_deg` in [72, 86] (`synthgen/configs/domain_rand.yaml`).
- **[VERIFIED] The flight layer is pymavlink, not DroneKit.** `custom_vehicle.py::CustomVehicle` wraps pymavlink (heartbeat, message rates, `goto_global`, `goto_local_ned`, `send_ned_velocity`, `condition_yaw`, ...). No file in the repo imports DroneKit.
  `launch_sitl_swarm.py` starts ArduCopter SITL instances in tmux (agents connect on UDP 14555+i). **ArduPilot SITL has no camera.**
- **[VERIFIED] Scoring is at ingest.** Fog scores each `DETECTION` against truth when it arrives (`Fog.ingest_detection`), writing `detections_<tag>.csv`; the oracle (`fire_model.oracle_hazards`) must stay off for Level 3. `planning/priority.py` scores a cell as `max(conf)` plus a repeat bonus: a deliberate placeholder.
- **[VERIFIED] The run manifest (`fog/run_log.py`) records weights path, size, mtime and `model_tag`, but no SHA-256 and no library versions.**

## 2. Which checkpoint to integrate first, and why
**Recommendation: `real_boreal_v2_best_seed0.pt`** (SHA-256 `090c0f4cbe2718d2b084a41b8a3736fa27b673bc66b20c64672b13c83c299a05`) **for wiring and smoke tests only.**
- It follows the registry convention (seed 0 is the reference member of each family: `models/MODELS.md`), is hash-pinned and fully documented.
- Validation does not discriminate between the seeds (val mAP50-95 0.620 / 0.625 / 0.624 [VERIFIED]), so there is no principled validation-based preference. Seed 0 also has the best Boreal v2 *test* numbers (mAP50 0.923, FP/img 0.067), but **that must not be the reason**: choosing on test is selection bias, n=3, and the differences are within the seed SD (0.027).
- **The Level 3 comparison must not use a single checkpoint per family.** Seed spread is large for the synthetic-trained family (Synth->Real mAP50 0.41-0.60), so the closed-loop comparison needs Model A x 3 seeds vs Model B-v2 x 3 seeds (the SITL scenarios, seeds and detection thresholds identical across models). Therefore design the integration around `weights` + `weights_sha256` + `model_tag` config values, not a hard-coded file.
- Prefer the v2 checkpoints over the v1 Model B for any new Level 3 work: their train/test partition is clip-disjoint (v1's is not). Do **not** use `models/real_boreal_bestv1.pt` or `models/synthetic_best.pt` (legacy files, not registry entries; the placeholder `weights: models/synthetic_best.pt` in `from_config`'s docstring is stale).
- Known model caveats to carry into Level 3: trained with the documented `cache=ram` deviation; 8-clip test; false positives on real empty scenes 1/65 but 1-9% on synthetic empty scenes; FP/img 0.07-0.21 at conf 0.25 (a false-positive misdirection risk for the planner).

## 3. Inference dependencies and preprocessing
- **Pin what the weights were trained and evaluated with** [VERIFIED]: `ultralytics==8.4.173`, `torch==2.8.0` (CUDA 12.8 build on the pod; the Ubuntu VM is CPU-only, so install the CPU wheel), `numpy`, `Pillow`, `PyYAML`, and OpenCV (pulled in by ultralytics). Python 3.12 was used on pods; the VM's Python version is [UNKNOWN].
  Ultralytics is AGPL-licensed software: [INFERRED] fine for a course project, worth a line in the report.
- **Harness inference settings** (`eval/datasets_boreal_v2.yaml`, [VERIFIED]): `imgsz 640`, `conf` floor 0.001 (needed for AP), NMS IoU 0.7, `max_det 300`, no test-time augmentation, single class `0: smoke`. The planner-side operating threshold is **not decided** [UNKNOWN]: Level 2 reports at 0.25, the config docstring says 0.45; the contract says a single fixed `tau` per model chosen on a calibration split, never on test.
- **Preprocessing:** pass a uint8 H x W x 3 frame **in BGR order** (Ultralytics' convention for numpy arrays; the harness read files with OpenCV) and let Ultralytics letterbox to 640 and normalise; do not resize or normalise by hand. Frames are 16:9 (synthetic generator 1280 x 720; Boreal sources 4096 x 2160, 3840 x 2160 or 1920 x 1080). Returned boxes are pixels in the *input frame*; scale and clip before projection.
- **Latency [UNKNOWN]:** v1 measured about 0.8 ms/image on a 4090; CPU latency on the VM was never measured (a YOLOv8s forward at 640 on a VM CPU is plausibly 0.1-0.5 s [INFERRED]). With `rate_hz` 1 and N agents sharing one VM CPU, and SITL slowed (`--speedup` < 1), this is probably workable but must be measured before choosing agent count and rate.

## 4. Interfaces and data flow (target design)
```
ArduCopter SITL --(MAVLink, UDP 14555+i)--> CustomVehicle (pymavlink state: lat, lon, rel alt, heading)
   -> swarm_agent.run_perception (rate_hz, default 1 Hz)  builds Pose(lat, lon, alt, heading, cell, t, sysid)
   -> YoloPerception.observe(pose)                                   [TO WRITE: perception/detector.py]
        1. frame = frame_synth.render(pose, camera, smoke_truth)     [TO WRITE: sim/frame_synth.py; there is no real camera]
        2. results = model.predict(frame, imgsz=640, conf=tau, iou=0.7, max_det=300, augment=False)   # boxes xyxy px, conf, cls 0
        3. each box -> Camera.from_pose(pose, intrinsics, lat0, lon0) -> ground footprint / bbox_to_cell -> grid cell (i, j)
        4. return [Detection(i, j, "smoke", conf, bbox, frame_id)]
   -> schema.make_detection -> UDP multicast bus (gossip)
        -> Fog.ingest_detection  (scored against truth at ingest -> detections_<tag>.csv, experiment_end precision / detection_lag_s)
        -> PriorityPlanner.on_detection (each agent, incl. neighbours) -> target cell -> mission/grid_mission -> CustomVehicle.send_ned_velocity / goto_local_ned
dashboard (rx_only) subscribes to DETECTION / BELIEF / FOG_STATE
```
Constructor contract for `YoloPerception(cfg, grid)`: read `weights`, **verify `weights_sha256` before loading** (refuse on mismatch, raise so `from_config` warns loudly), `model_tag` (e.g. `real_boreal_v2_seed0`), `conf_threshold`, `rate_hz`, camera intrinsics, and expose `enabled = True`, `model_tag`, `observe()`, `close()`. Log one explicit line at start (`perception: YOLO loaded <tag> sha256=<first 12>`): the fail-soft design otherwise makes a wrong path look like "the detector found nothing".

## 5. Gaps and unresolved assumptions (most consequential first)
1. **There is no camera in SITL, so frames must be synthesised (`sim/frame_synth.py`, empty).** The renderer needs the smoke's ground position to draw it, but agents must never own the fire (CLAUDE.md: "Fog owns the fire"). A fog-owned truth channel dedicated to the renderer (not `HAZARD_UPDATE`, not visible to the planner) is needed and is unspecified.
2. **Imagery domain confound [INFERRED from Level 2 numbers].** If every Level 3 frame is a synthetic render, the synthetic-trained model is in-domain (Synth->Synth mAP50 0.995) and the Boreal-trained model is cross-domain (Real->Synth 0.61 +- 0.10). A Level 3 "Model A vs Model B" difference would then largely restate the Level 2 domain gap, not isolate perception->planning propagation. Options: (a) accept and state it; (b) also run on rendered frames styled closer to Boreal (limited by renderer realism); (c) real-footage replay (`ReplayPerception`), which cannot give cell-level truth (contract section 3). This needs an explicit decision before building the loop.
3. **Camera conventions [UNKNOWN, documented as unconfirmed]:** `Pose` has no pitch/roll/gimbal tilt; the repo's `camera: {tilt_deg: 30}` and `fov_deg: 78` do not say from-nadir vs from-horizon or horizontal vs diagonal. `sim/camera.py` requires `pitch_deg` (below the horizon) and `hfov_deg` and says to convert explicitly. No `perception:` section exists in `exp_swarm_*.yaml` yet.
4. **Operating threshold `tau` and the box->cell rule are undecided** (contract sections 1-2). Level 3 should log `(frame, cell, y_hat, score)` and the smoke-presence truth `z`, not just fog's `detection_truth_radius` hits.
5. **Probability map / calibration is blocked by design** (contract sections 5-7): cell-level sensitivity/specificity do not exist for real data, only for a *new* synthetic cell-observation set and SITL passes. Keep `PriorityPlanner.score()` as is; do not write `mapping/probability_map.py` yet.
6. **CPU throughput with several agents on the VirtualBox VM is unmeasured** (section 3).
7. **Provenance gap:** the run manifest lacks the weights SHA-256 and library versions; add both before any reported Level 3 run (small change; not done here).
8. **Checkpoint availability:** the checkpoints are on the RunPod volume only until `docs/handoffs/checkpoint_backup.md` is executed; the VM path is expected under `local_data/backups/boreal_v2/` and `local_data/` is not visible to Claude Code Cloud.
9. **Terminology:** the request mentioned DroneKit; the code uses pymavlink via `CustomVehicle`. No DroneKit dependency is needed or planned unless you want it.

## 6. Recommended next step (one, in order)
On the Ubuntu VM, **after the checkpoint backup is verified**: implement `perception/detector.py::YoloPerception` against fake and replayed frames only (no SITL, no renderer), loading `real_boreal_v2_best_seed0.pt` by path + SHA-256, with unit tests (hash refusal, fail-soft path, box->cell via `Camera.bbox_to_cell`, no tensors leaking out of `observe`), and **measure CPU latency of all six checkpoints (Model A and B-v2, seeds 0-2) on a few synthetic frames**. That removes the two biggest unknowns (latency, interface correctness) without touching flight code. Then decide gap 2 (imagery domain) and gap 1 (truth channel) before writing `sim/frame_synth.py`.

## 7. Acceptance checks for the later implementation
- `perception.enabled: true` with a wrong path or wrong hash fails loudly in the run log, not silently to `NullPerception`.
- Identical seeds/scenarios/thresholds across models; manifest records weights SHA-256, ultralytics/torch versions, camera conventions and `tau`.
- Oracle off; `detections_<tag>.csv` scored at ingest; no `FireSim` in `swarm_agent.py`.
- Level 2 box-level numbers are never used as Level 3 likelihoods (contract section 5).

## 8. Review and implementation plan (added 2026-11-02, Ubuntu session; planning only, no integration code written)
Review against the code, with `docs/handoffs/` claims re-checked:
- **Confirmed [VERIFIED]:** `perception/detector.py`, `sim/frame_synth.py`, `mapping/probability_map.py` are 0 bytes; no DroneKit import anywhere; `swarm_agent.run_perception` / `Pose` / `from_config` behave as section 1 says
  (a constructor that raises returns `NullPerception` with a warning: tested by monkeypatching); `manifest.json` records no SHA-256.
- **New finding 1 (interface):** `swarm_agent` calls `from_config(pcfg, grid=self.grid)`, where `self.grid` is a `mission.GridMission`, **not** `sim.camera.GridSpec`. `YoloPerception` must build
  `GridSpec.from_miles(grid.cfg.miles, grid.cfg.cells)` and take `lat0/lon0` from `grid.cfg`. The index convention agrees: `i` runs east and `j` north in `fog/state.cell_index`, `GridSpec` and `GridMission`.
- **New finding 2 (resources):** this Ubuntu VM has **1 vCPU, about 2 GB RAM**, Python 3.10.18, and neither `ultralytics` nor `torch` installed. CPU YOLOv8s inference alongside several ArduCopter SITL processes may not fit;
  measure first (step 1). Fallback: run inference on a bigger machine or one process serving all agents. Pods used Python 3.12; torch 2.8.0 / ultralytics 8.4.173 both support 3.10 [INFERRED: not yet installed here].
- **New finding 3 (docs):** the `from_config` docstring still names `models/synthetic_best.pt` (legacy); update when `YoloPerception` lands. `.gitignore` on the science branch lacked `local_data/` (only `*.pt`); added.
- No contradiction found in the checkpoint-selection logic (section 2): seed 0 for wiring only, 3+3 seeds for any reported comparison.

Implementation order (each step is its own small commit with tests; none touches frozen Level 2/Boreal v2 results or flight code until step 4):
1. **Environment + latency:** CPU wheels (`pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu`, `ultralytics==8.4.173`) in a venv; time `predict(imgsz=640)` for the six registry checkpoints on a few synthetic frames; record RAM. Gate: decide agent count / rate_hz.
2. **`perception/detector.py::YoloPerception`** with injected frame source (protocol `FrameSource.get(pose) -> BGR uint8 HxWx3`): weights path + `weights_sha256` check (raise on mismatch), `model_tag`, `conf_threshold`, `camera: {hfov_deg, pitch_deg_below_horizon, width, height}`, `GridSpec` from `grid.cfg`; boxes -> `Camera.bbox_to_cell` -> `Detection`; log one start line with tag and hash prefix. Unit tests with a stub model and a fake frame source (hash refusal, bbox->cell, out-of-grid box dropped, no tensors returned).
3. **Provenance:** add `weights_sha256`, ultralytics/torch versions, camera conventions and `tau` to `fog/run_log.py` manifest.
4. **Frame source + truth channel** (decisions first: imagery domain, fog-owned render-truth channel, section 5 gaps 1-2): `sim/frame_synth.py`, fed only by fog, never `FireSim` in the agent; oracle stays off.
5. **Smoke run** in SITL with `real_boreal_v2_best_seed0.pt`: check it produces `detections_<tag>.csv`, no NullPerception warning. This is wiring evidence only.
6. **Level 3 experiment** (Model A x3 seeds vs Model B-v2 x3 seeds, identical scenarios, seeds and `tau`) is the only step that can support a statement about autonomous monitoring.

**Scope guard:** Level 2 box-level metrics are perception evidence only. No claim of improved coverage, time to detection or misdirection is made or implied until step 6 produces it; steps 1-5 yield no such evidence.

## 9. Implementation status (2026-11-02, Ubuntu session)
Steps 1-3 of section 8 are done (benchmark, `YoloPerception`, manifest provenance) and the frame source/truth channel (step 4) is implemented as a fog-side HTTP service with a nadir camera; a single-UAV run (step 5) works. Step 6 (the Level 3 matrix) is not run. Interface decisions: the truth channel is `sim/camera_service.py` (fog-owned; JPEG only), the agent sees `perception/frames.py::FrameSource`, `GridSpec` comes from `grid.cfg`, the nadir box centre (not the box bottom) is the ground anchor. Details, measurements, findings and the recommended next experiment: `docs/experiments/perception_integration_v0.md`.
