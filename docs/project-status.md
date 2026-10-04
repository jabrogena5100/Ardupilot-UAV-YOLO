# Project Status — ECE 496 AI-Enabled Autonomous Wildfire Monitoring

**Last updated:** 2026-10-04

## Current objective

Complete the ECE 496 research project investigating whether training a wildfire/smoke detector on synthetic versus real imagery affects:

1. Neural-network perception performance.
2. Cross-domain generalization.
3. Autonomous UAV wildfire-monitoring performance in ArduPilot SITL.

The primary research deliverable is the **2×2 perception evaluation matrix**:

| Training  | Evaluation | Purpose                           |
| --------- | ---------- | --------------------------------- |
| Synthetic | Synthetic  | Same-domain synthetic performance |
| Synthetic | Real       | Synthetic → real generalization   |
| Real      | Synthetic  | Real → synthetic generalization   |
| Real      | Real       | Same-domain real performance      |

Level 3 then compares how the resulting perception models affect autonomous monitoring.

---

## Current state

### Level 1 — Neural-network perception

**Real dataset:** Boreal Forest dataset.

**Synthetic dataset:** Not yet completed. Planned approach is the project's 2.5D synthetic image generator using:

* ground-plane forest texture
* pinhole camera projection
* fire/smoke sprites placed at known ground coordinates
* camera position / altitude / angle variation
* lighting, haze, visibility, and color variation
* automatic ground-truth labels

**Models:**

* Model A — synthetic-trained YOLO
* Model B — real/Boreal-trained YOLO

**Current checkpoint:** A 3-epoch YOLO checkpoint exists for demonstration/testing only. It is not considered a final experimental result.

**Status:** In progress.

---

## Level 2 — Cross-domain evaluation

The evaluation harness and unified annotation schema are not yet complete.

Required final evaluation:

* fixed train/test splits
* unified annotation format
* same evaluation code path for all four matrix cells
* fixed seeds
* logged hyperparameters
* comparable training budgets
* precision
* recall
* mAP
* false-positive rate
* false-negative rate
* confidence
* localization error

**Status:** Not yet implemented. This is the highest-priority overdue component.

---

## Level 3 — Autonomous monitoring

The overall data flow is already substantially implemented:

```text
UAV Camera
    ↓
Neural Network
    ↓
Detection Probability
    ↓
Geographic Probability Map
    ↓
Priority Region
    ↓
UAV Coverage Decision
```

Existing infrastructure includes:

* ArduPilot / ArduCopter SITL
* wildfire simulation
* UAV swarm
* coverage/mission logic
* communication bus
* detection messages
* dashboard
* priority planning

The current perception scoring is still a placeholder and must eventually be replaced by the intended probability-map approach using Level 2 confusion-matrix results.

**Status:** Infrastructure substantially built; perception integration and final probability model remain incomplete.

---

## Synthetic generator

Planned location:

```text
synthgen/
```

Requirements:

* 2.5D rather than flat 2D compositing
* pinhole camera model
* ground-plane forest texture
* fire/smoke placement at known ground coordinates
* automatic labels
* controllable camera altitude and angle
* lighting/haze/color variation
* reproducible generation using fixed seeds

The projection implementation must be reusable by `sim/camera.py` so that synthetic image generation and UAV camera-to-ground projection use the same geometry.

**Status:** Not yet implemented.

---

## Important existing infrastructure

### Swarm / simulation

* `wildfire_sim.py` — FireSim cellular-automaton fire model
* `fog_tracker.py` — ground-station/fog entry point
* `fog/state.py` — world state and detection scoring
* `swarm_agent.py` — UAV control loop
* `mission/grid_mission.py` — coverage mission
* `mission/partitions.py` — static area assignment
* `mapping/bitmap.py` — coverage state
* `comms/schema.py` — message definitions
* `comms/gossip.py` — UDP multicast communication
* `launch_sitl_swarm.py` — SITL swarm launcher
* `launch_experiment.py` — experiment orchestration

### Perception

* `perception/base.py` — perception interface
* `NullPerception` remains the default fallback
* YOLO checkpoint already exists for demonstration/testing

### Dashboard

* `dashboard/server.py`
* vanilla HTML/CSS/JavaScript
* browser polling
* canvas visualization
* no FastAPI/WebSockets/Leaflet/Cesium

---

## Important design decisions

### Fire ownership

Fog owns the FireSim ground truth.

Agents must not run independent FireSim instances.

### Oracle mode

Oracle ground-truth hazard broadcasts must remain disabled for real perception experiments.

```text
fire_model.oracle_hazards: false
```

Oracle mode is only an upper-bound/perfect-information experiment.

### Detection scoring

Detections are scored when they are ingested, because the ground truth changes over time.

Primary Level 3 detection result files include:

```text
detections_<tag>.csv
experiment_end_*.txt
```

### Probability mapping

Current priority scoring is a placeholder based on detection confidence.

The intended replacement is a log-odds Bayesian probability grid using sensitivity/specificity obtained from the Level 2 confusion matrices.

### Coverage bitmap

The canonical map bit index is:

```text
i*N + j
```

### FOG_STATE

Belief state is sparse-encoded:

```text
[[i,j,p], ...]
```

above the belief floor rather than transmitting a dense grid.

---

## Current known issues / TODO

### Priority 1 — Unified evaluation system

Create:

* common annotation schema
* frozen train/test splits
* unified evaluation script
* configuration-driven matrix experiments
* fixed seeds
* metric logging

### Priority 2 — Fix `launch_experiment.py`

The launcher still contains old paths referring to:

```text
core/swarm_agent.py
core/fog_tracker.py
```

These should point directly to the current files in the repository.

### Priority 3 — Build `synthgen/`

Implement the 2.5D synthetic image generator and reusable projection math.

### Priority 4 — Build `sim/camera.py`

Use the same projection model as the synthetic generator.

### Priority 5 — Build `perception/detector.py`

Create the thin YOLO wrapper implementing the project's perception interface.

### Priority 6 — Build `mapping/probability_map.py`

Implement the log-odds probability map after the Level 2 confusion matrices are available.

### Priority 7 — Optional demo vehicle

Implement:

```text
sim/fake_vehicle.py
```

for dashboard demonstrations without requiring SITL.

This is useful for demos but is not required for the core research deliverable.

### Priority 8 — Formalize tests

Promote existing ad-hoc tests into:

```text
tests/
```

---

## Training / compute status

The development laptop has no GPU.

**Primary training environment:** RunPod cloud GPU.

Colab/Kaggle may be used as alternatives when convenient.

Cloud GPU spending is acceptable when it materially improves training speed or experiment reliability.

Training should not be artificially limited to a small number of epochs because the development machine lacks a GPU.

For important experiments, inspect:

* training curves
* validation curves
* loss
* precision
* recall
* mAP
* overfitting behavior

before deciding whether additional epochs, data changes, augmentation changes, or architecture changes are appropriate.

All important training runs should record:

* model/version
* dataset
* train/test split
* training configuration
* epochs
* metrics
* checkpoint
* random seed when applicable

The four Level 2 matrix cells should use comparable training methodology and documented training budgets.

---

## Git / repository state

Project repository:

```text
~/src/Ardupilot-UAV-YOLO
```

Reference-only upstream repository:

```text
~/src/ardupilot
```

Only `Ardupilot-UAV-YOLO` is the project repository.

Do not modify, commit, or push changes to the upstream ArduPilot repository unless explicitly instructed.

Claude should:

1. Inspect `git status` before significant work.
2. Keep changes organized into logical commits.
3. Run relevant tests/checks before completing major work.
4. Review `git diff`.
5. Commit completed work with a descriptive message.
6. Update this file when project state materially changes.
7. Never push to GitHub unless explicitly requested.

---

## How to continue after a context reset

Read:

```text
CLAUDE.md
docs/project-status.md
```

Then:

```bash
git status
git log --oneline -5
```

Determine:

1. What was completed.
2. What changed since the previous checkpoint.
3. What remains incomplete.
4. Whether any TODO above is stale.
5. What the highest-priority next task is.

Continue from the existing repository state rather than restarting or recreating completed work.

---

## Latest checkpoint

**Current highest-priority task:**

Build the unified annotation schema and evaluation harness before expanding the UAV/dashboard system further.

**Current research bottleneck:**

The project has substantial UAV/simulation/dashboard infrastructure, but the synthetic dataset, unified evaluation pipeline, and trustworthy 2×2 perception results are not yet complete.

**Next meaningful milestone:**

Produce the first reproducible Level 2 evaluation matrix using fixed datasets, fixed evaluation code, documented training configurations, and comparable model-training budgets.



---

## Training log

### 2026-10-04 — Model B (real/Boreal) v1 trained and evaluated

* YOLOv8s, 100 epochs, imgsz 640, batch 32, seed 0, deterministic, on RunPod 1x RTX 4090 (secure, US-IL-1).
* Frozen Boreal split at `/workspace/splits/boreal_v1/` (existing video-disjoint split; not rebuilt).
* Held-out test: P 0.937, R 0.933, mAP50 0.944, mAP50-95 0.669 (701 images, single class `smoke`).
* Checkpoint `real_boreal_best_v1.pt` and all outputs are on the `wildfire-datasets` volume under `/workspace/runs/`; full details in `models/MODELS.md`.
* Approx. cost ~ $0.60 total (Stage 1 inspection/smoke ~ $0.08, training+eval ~ $0.52). No pods left running.
* Caveats: Boreal has no fire class; test has no negative images; 5 unlabeled raw images recorded, not fixed.
* Not yet done: unified annotation schema, repo-side eval harness (`eval/`) running this same `(checkpoint, test-set)` path for all four matrix cells, Model A (synthetic) training with the identical recipe.
* Scripts used are archived in `eval/runpod/`. They are one-off RunPod launchers, not yet the config-driven harness.
