# Project Status — ECE 496 AI-Enabled Autonomous Wildfire Monitoring

**Last updated:** 2026-10-27 (documentation consolidation; no experiments since 2026-10-05)

This file is the primary source for current progress and immediate next tasks. `CLAUDE.md` holds the rules; `models/MODELS.md` is authoritative for checkpoint provenance. Always verify claims here against the repo (`git status`, `wc -l`, hashes) before acting.

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

**Synthetic dataset:** `synth_smoke_v1` generation is **COMPLETE and validated** (5,304 images, 2026-10-05; record in `docs/datasets/synth_smoke_v1.md`). Local copy: `local_data/datasets/synth_smoke_v1/`. The RunPod generation pod is deleted, the dataset is not yet on the RunPod volume, and **no Model A training has started**. It was built with the project's 2.5D synthetic image generator using:

* ground-plane forest texture
* pinhole camera projection
* fire/smoke sprites placed at known ground coordinates
* camera position / altitude / angle variation
* lighting, haze, visibility, and color variation
* automatic ground-truth labels

**Models:**

| Model | Checkpoint | State |
|---|---|---|
| Model A — synthetic-trained YOLO | `synth_smoke_best_v1.pt` (planned name, `docs/synthetic/synth_smoke_v1_spec.md` §5) | **Not trained.** Dataset ready; blocked on the edge-touch review gate. |
| Model B — real/Boreal-trained YOLO | `real_boreal_best_v1.pt`, SHA-256 `1dc35ef2…8355b6` | **Trained and tested once** (2026-10-04): test mAP50 0.944, mAP50-95 0.669. Details: `models/MODELS.md`, `docs/experiments/real_boreal_v1.md`. |

**Legacy "3-epoch demo checkpoint":** earlier versions of this file and `CLAUDE.md` said a 3-epoch demo checkpoint exists. No such file is in the working tree, and nothing in `models/MODELS.md` or the experiment record describes it. The only trace is git history: `models/real_boreal_bestv1.pt` (5,442,970 bytes, SHA-256 `05ef7d46…47a54`, commit `73dab87`, deleted in `23dc32a`) and a 0-byte `models/synthetic_best.pt` placeholder. The 5.4 MB blob is not Model B v1 (22.5 MB, different hash). Its epochs, architecture and data are **unrecorded**; do not call it Model A or B or cite it as a result. See "Open contradictions".

**Status:** Model B done; Model A pending.

---

## Level 2 — Cross-domain evaluation

The evaluation harness is implemented (2026-10-27) but not yet validated on real checkpoints. State of the pieces:

* De facto unified annotation format: YOLO txt (`cls cx cy w h`, normalized), single class `0: smoke`, enforced for both domains by `eval/validate_dataset.py`. There is no separate formal schema document or shared loader yet.
* Frozen splits: Boreal (`/workspace/splits/boreal_v1/` on the RunPod volume, hashes in `models/MODELS.md`) and `synth_smoke_v1` (hashes in `docs/datasets/synth_smoke_v1.md`). Test views `test_pos` and `test_all` are reported separately.
* Model B's test number came from one-off RunPod scripts (`eval/runpod/`), not the shared harness.
* Harness: `eval/metrics.py`, `eval/run_matrix.py`, `eval/datasets.yaml` (2026-10-27).

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

**Status:** Harness implemented and unit-tested 2026-10-27; not yet run on real checkpoints and the Boreal test set (Priority 1).

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

Location: `synthgen/` (`scene_params.py`, `render.py`, `annotate.py`, `generate_dataset.py`, `configs/domain_rand.yaml`), sharing `sim/camera.py`. Requirements it meets:

* 2.5D rather than flat 2D compositing
* pinhole camera model
* ground-plane forest texture
* fire/smoke placement at known ground coordinates
* automatic labels
* controllable camera altitude and angle
* lighting/haze/color variation
* reproducible generation using fixed seeds

The projection implementation must be reusable by `sim/camera.py` so that synthetic image generation and UAV camera-to-ground projection use the same geometry.

**Status:** Done. POC validated (`docs/synthetic/poc_report.md`); `synth_smoke_v1` frozen at `9cb0f1a` (tag `synth-v1`) and fully generated (`docs/datasets/synth_smoke_v1.md`). Tests: `tests/test_camera.py`, `tests/test_synthgen.py` (41 tests, all passing on 2026-10-27 in about 124 s; the suite is slow because it renders images). Known generator limitation: edge-touch rate 0.76 vs Boreal 0.30.

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
* `launch_experiment.py` — experiment orchestration (broken paths, see known issues)
* `custom_vehicle.py` (MAVLink wrapper), `swarm_strategies.py` (fallback velocity policies), `fog/run_log.py` (CSVs, `manifest.json`), `fog/config.py` (YAML/JSON loading), `mission/geo.py` (projection, Chebyshev distance)
* `exp_swarm_{solo,coop,fire}.yaml` — experiment configs

### Perception

* `perception/base.py` — perception interface
* `NullPerception` remains the default fallback; `from_config()` fails soft to it with a warning
* `perception/detector.py` (YOLO wrapper) is an empty stub, so no trained checkpoint is wired into the agents yet
* `perception/base.py`'s docstring example still names `models/synthetic_best.pt` (legacy name; code left unchanged)
* `sim/camera.py` — shared pinhole projection, round-trip tests; confirm `tilt_deg`/`fov_deg` conventions before Level 3 use
* `sim/frame_synth.py` (camera frame rendering for the agent) is an empty stub

### Dashboard

* `dashboard/server.py`: plain `http.server`, read-only bus subscriber (`rx_only`, cannot transmit)
* vanilla HTML/CSS/JavaScript in `dashboard/static/`, no build step or CDN
* browser polls `/api/state` every 250 ms, draws to a `<canvas>` grid
* no FastAPI/WebSockets/Leaflet/Cesium; a georeferenced map view would be a deliberate dashboard rebuild

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

### Edge-touch gate — CLEARED 2026-10-27 (baseline Model A only)

The owner cleared the `synth_smoke_v1` edge-touch review (about 0.76 / 0.76 / 0.77 vs Boreal train 0.30) for baseline Model A training, with conditions:

1. `synth_smoke_v1` is preserved exactly as frozen (tag `synth-v1`, `9cb0f1a`).
2. The edge-touch mismatch with Boreal is documented as an experimental limitation in every write-up.
3. Model A on `synth_smoke_v1` is the **baseline** experiment, not proof the synthetic domain is optimal.
4. Frozen train/val/test splits are not modified and no existing checkpoint is overwritten.
5. Synthetic dataset improvements are separately versioned (`synth_smoke_v2`, reported as post-hoc).
6. RunPod compute and storage charges are reconciled against the $100 autonomous allowance before paid resources are launched.
7. Ordinary authorized experiments proceed within the `CLAUDE.md` limits without further approval.

Still required before upload or training: the generated YAML/list files contain the generation pod's absolute paths, so use path-rewritten copies (`docs/datasets/synth_smoke_v1.md`); never edit the hashed originals. The pre-declared diagnostic (Model A on Boreal test, grouped by edge-touching vs not) still applies after training.

### Priority 1 — Unified evaluation system (implemented 2026-10-27; real-data validation pending)

Built: `eval/metrics.py` (pure-numpy metrics: mAP50, mAP50-95, P/R/F1 at max-F1 and fixed conf, FN/FP counts and rates, empty-scene image FP rate, mean confidence of TP/FP, IoU and centre-error localisation, recall split by edge-touching vs not), `eval/run_matrix.py` (config-driven runner; verifies dataset image-count and file-list hash pins and checkpoint SHA-256 pins, saves predictions, writes per-cell JSON with provenance plus `matrix.md`/`matrix.csv`), `eval/datasets.yaml` (the datasets, models and cells), and `tests/test_eval_metrics.py` + `tests/test_run_matrix.py` (13 tests, all passing; includes the frozen `synth_smoke_v1` test pins and a recompute-from-saved-predictions identity check). Conventions are documented in the `eval/metrics.py` docstring (a perfect detector scores 0.995, as in Ultralytics).

Not yet done: (a) run on the real Boreal test set and reproduce Model B's one-off numbers (mAP50 0.9442, mAP50-95 0.6688) within tolerance — this needs a pod, since the Boreal split lives on the RunPod volume; (b) pin Model A's checkpoint SHA-256 in `eval/datasets.yaml` after training; (c) the Boreal test set has no usable negatives, so the empty-scene FP rate comes only from `synth_test_all`.

### Priority 2 — Train Model A (gate cleared; follows Priority 1 per the harness-first rule)

Use Model B's recipe with only the differences pinned in `docs/synthetic/synth_smoke_v1_spec.md` §5. Requires uploading the dataset to the RunPod volume with rewritten paths. Stay within the budget in `CLAUDE.md`; record spend in the ledger below.

### Priority 3 — Fix `launch_experiment.py`

Lines 84–85 point at `HERE / "core" / "swarm_agent.py"` and `HERE / "core" / "fog_tracker.py"`; there is no `core/`. Change to `HERE / "swarm_agent.py"` / `HERE / "fog_tracker.py"`. Also `agent_cmd` is still defined twice (lines ~169 and ~176; the first is dead code). Not fixed as of 2026-10-27.

### Priority 4 — `perception/detector.py`

Thin YOLO wrapper implementing `observe()`; wire Model B (and later Model A) in via `perception.weights`. Needs `sim/frame_synth.py` to render frames from UAV pose using the shared projection.

### Priority 5 — `mapping/probability_map.py`

Log-odds belief grid; blocked on Level 2 confusion matrices. Then swap only `PriorityPlanner.score()`'s body.

### Priority 6 — Optional demo vehicle (`sim/fake_vehicle.py`)

A `SimVehicle` matching `CustomVehicle`'s public surface via a `sim:lat,lon,offset_n,offset_e,seed` connection string, for dashboard demos without SITL. Not required for the research deliverable.

### Priority 7 — Tests

`tests/` exists with `test_camera.py` and `test_synthgen.py`. Still to promote: grid round-trip, partition overlap, bitmap merge, fake-vehicle fly-through.

### Completed (history)

* `sim/camera.py` with round-trip tests (`9951947`).
* `synthgen/` POC, validator, and `synth_smoke_v1` full generation (`08c85e3` … `17e7f38`).
* Model B v1 training and evaluation (`1ed6f21`).
* Earlier fixes, kept so they don't come back: an older dashboard emitted `visited_cnt` but no `visited` list, so coverage drew nowhere (fixed by switching to a hex bitmap wire format); endgame stall when one cell remained, with every drone treating the same cell as another's claim (fixed by `mapping/bitmap.py: single_remaining()`); the agent-only `j*N + i` map index (now `i*N + j` everywhere).

### Open contradictions (unresolved, do not guess)

* **Legacy 3-epoch checkpoint:** claimed in older docs, no registry entry, provenance unrecorded (above). Needs the owner to say what it was, or to be dropped from the narrative.
* **Approval wording (resolved 2026-10-27):** the dataset doc and spec say upload and Model A training "need separate approval"; the owner has since cleared the gate and authorized ordinary experiments within the `CLAUDE.md` limits, which supersedes that wording.
* **Spend ledger (resolved 2026-10-27):** reconciled to billing; the undocumented 2026-10-04/05 pods (two with >$1 GPU charges) are still unexplained individually and the experiment docs' cost estimates are too low.
* **`tilt_deg`/`fov_deg` conventions** in `sim/camera.py` still need confirmation before Level 3 use.

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

### Autonomous RunPod budget ledger

Limits (from `CLAUDE.md`): $150 total project, $100 cumulative autonomous, $50 reserve, $20 per experiment, one paid GPU pod at a time.

**Reconciled 2026-10-27 against RunPod billing** (`list-billing`, `list-pod-billing`, `list-network-volume-billing`; the account has charges only in October 2026). All spend is counted against the allowance, including pre-policy runs.

| Item | Billed |
|---|---|
| Pods, 2026-10-04 (7 pods: Model B stage 1 / training / evaluation and short inspection pods) | $3.04 |
| Pods, 2026-10-05 (5 pods incl. CPU generation pod `ju9568emxzxr1a` $0.134) | $0.30 |
| Network volume `wildfire-datasets` (`sej40ez3ia`, 50 GB STANDARD, US-IL-1), 2026-10-01 to date | $2.78 (accrues about $0.10-0.12/day while it exists) |
| **Total billed** | **$6.12** |
| **Remaining autonomous allowance** ($100 - $6.12) | **$93.88** |

Corrections: the earlier per-experiment estimates in this file ("about $0.60" for Model B, "about $0.13" for generation) came from pod run time and undercount; billing shows about $3.04 for 2026-10-04 (two pods with $1.38 and $1.07 of GPU charges are not individually documented) and $0.30 for 2026-10-05. The experiment documents keep their original estimates; this ledger is authoritative for spend. Pod count at reconciliation: 0 running (`list-pods` empty).

Update this ledger after every paid action, and re-run the billing reads before launching anything.

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

**Update 2026-10-05 (end of session):** `synth_smoke_v1` generation is COMPLETE and validated; the RunPod CPU pod is deleted; no Model A training has started. **Next decision:** review the synthetic edge-touch distribution (about 0.76 train / 0.76 val / 0.77 test vs about 0.30 in Boreal train) before spending GPU money on Model A. The spec (Option A) had already accepted this gap as a documented limitation with a pre-declared diagnostic, so the review is about whether to proceed as-is or to make a post-hoc v2; do not change `synth_smoke_v1` itself. Also before any upload or training: the generated YAML/list files contain the pod's absolute paths and need path-rewritten copies. Details: `docs/datasets/synth_smoke_v1.md`.

**Update 2026-10-27:** `CLAUDE.md` consolidated into a short operating manual; stale claims in this file corrected (generator, camera and Model B status, 3-epoch checkpoint, launcher bug status). Checkpoint, dataset and code were not touched, and no paid resources were used.

**Current highest-priority task:**

Validate the harness on real data (reproduce Model B), then train Model A (gate cleared 2026-10-27) and run the 2x2.

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

### 2026-10-05 — Synthetic pipeline POC (no training, no full dataset)

* Inspected the Boreal reference (14 train frames + train/val label statistics): `docs/synthetic/boreal_visual_analysis.md`. Boreal pixels stay outside the repo.
* Implemented `sim/camera.py` (shared pinhole projection) and the `synthgen/` stubs (scene sampling, renderer, annotator, dataset writer), plus `eval/validate_dataset.py`. 31 unit tests pass (`python3 -m unittest discover -s tests -t .`).
* Generated and validated a 180-image POC (1280x720, smoke only, 7.5% background): `docs/synthetic/poc_report.md`. Measured about 0.96 img/s on one core; a Boreal-sized set (4,954 images) is about 85 min.
* Open items: edge-touching boxes (81% synthetic vs 30% Boreal), full-dataset size approval, `tilt_deg`/`fov_deg` convention for Level 3, eval harness (`eval/metrics.py`, `run_matrix.py`) still empty.

### 2026-10-05 — Phase 5: synth_smoke_v1 pre-scaling changes (candidate, not frozen)

* Implemented the approved changes: split-specific sprite banks (48/16/16), capped and decoupled plume drift, alternative box conventions in metadata, larger texture banks (16/6/6) with palette jitter, explicit positive/negative counts and separate `test_pos`/`test_all` views. Spec: `docs/synthetic/synth_smoke_v1_spec.md`.
* Fixed a sampling bias introduced during this work (retries re-drew archetype/size); archetype mix and box-area marginal are preserved.
* Edge-touch rate is NOT corrected (0.80 vs Boreal 0.30); evidence and options in the spec. Awaiting a decision before freezing.
* Verified Model B's optimizer: AdamW, lr0 0.002 (from `results.csv` and the Ultralytics 8.4.173 source). Model A will pin it explicitly.
* Small POC regenerated and validated; two runs are byte-identical. Full dataset not generated; Model A not trained.

### 2026-10-05 — synth_smoke_v1 full generation (complete, validated; no training)

* Generated on a RunPod CPU pod (`synth-v1-cpu-gen`, about $0.07/hr, about $0.13 total) from frozen commit `9cb0f1a`, seed 1001, 5,304 images (train 3,479 / val 774 / test 1,051) in 2,781 s. Pod deleted afterwards.
* First run aborted at `val_000737` (no valid scene in the default 40 attempts; faint `wide_steep` plume). Clean rerun with the existing runtime flag `--max-attempts 500` succeeded; max `attempts_used` = 66. No generator code changed. All 4,216 images from the partial run were byte-identical in the rerun.
* Validator passed on `data.yaml`, `data_test_pos.yaml`, `data_test_all.yaml`; tarball and all 10,616 file hashes verified after transfer to `local_data/datasets/synth_smoke_v1/`.
* Measured edge-touch fractions 0.76 / 0.76 / 0.77 vs Boreal 0.30: still open, to be reviewed before Model A training. Absolute-path caveat in the YAML/list files documented.
* Full record: `docs/datasets/synth_smoke_v1.md`. Spec status line updated.
