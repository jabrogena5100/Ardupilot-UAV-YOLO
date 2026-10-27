# Wildfire Monitoring Project — ECE 496 Fall 2026

AI-Enabled Autonomous Wildfire Monitoring. This file is the operating manual: rules and
invariants that change rarely. **Current progress, next tasks and the work log live in
`docs/project-status.md`. Verified checkpoint provenance lives in `models/MODELS.md`.**

**Research question:** does the training domain (synthetic vs. real imagery) change how well a
neural network perceives wildfire/smoke, and does that difference propagate into autonomous UAV
monitoring performance?

## Before starting any task (verify, do not trust docs)

Documentation lags the repo. Before acting on anything a doc says is done, missing, or next:
1. Run `git status` and `git log --oneline -10`.
2. Check the files the task touches (`ls`, `wc -l`, grep): an empty file is a stub, and a doc may be stale.
3. For models or datasets, check `models/MODELS.md` and the dataset record, then the hash on disk.
4. If docs and repo disagree, trust the repo, say so, and fix the doc in the same commit.
5. Never report a result that was not measured in this repo's recorded artifacts.

## Repositories

- `~/src/Ardupilot-UAV-YOLO` — the only repo Claude may modify, commit, or push to (never push unless asked).
- `~/src/ardupilot` — upstream, reference only. Do not modify or commit.
- `AGENTS.md` configures Codex as the read-only research auditor. Do not edit it unless asked.

## Three-level research structure

**Level 1 — Domain comparison.** Model A: trained on synthetic smoke images (auto-generated,
auto-labeled). Model B: trained on the real Boreal Forest dataset. Metrics: precision, recall, mAP,
false positive/negative rate, confidence, localization error.

**Level 2 — Cross-domain generalization.** 2×2 matrix (Synth→Synth, Synth→Real, Real→Synth,
Real→Real), every cell evaluated by the *same* code path. This matrix is the actual deliverable;
everything else is infrastructure in service of it.

**Level 3 — Perception → autonomous monitoring.** Closed loop on ArduPilot SITL:
`UAV Camera → Neural Network → Detection Probability → Geographic Probability Map → Priority Region
→ UAV Coverage Decision`. A known ground-truth fire is placed in sim; compare time to detection,
coverage efficiency and false-positive misdirection across Model A vs. Model B. The data flow exists
(`perception/base.py`, `comms/schema.py`, `planning/priority.py`); the probability math is a
placeholder (see Decisions).

## Model naming (authoritative: `models/MODELS.md`)

- **Model B** = `real_boreal_best_v1.pt` (YOLOv8s, Boreal, trained 2026-10-04, SHA-256 in the registry). Exists.
- **Model A** = `synth_smoke_best_v1.pt`, baseline trained 2026-10-27 on `synth_smoke_v1` (SHA in the registry). Baseline only.
- `models/synthetic_best.pt` and `models/real_boreal_bestv1.pt` are legacy git-history artifacts, not
  registry entries; do not treat either as Model A/B. Provenance of the old "3-epoch demo checkpoint"
  is unrecorded (see `docs/project-status.md`, open contradictions).
- Single class, `0: smoke`; there is no fire class in Boreal or in `synth_smoke_v1`.

## Two different "2D"s — do not conflate

1. **Fire-spread model** (`wildfire_sim.py::FireSim`): a 2D cellular automaton over the coverage grid.
   Intentional, built, stays 2D.
2. **Synthetic training imagery**: **2.5D** — a ground-plane forest texture projected through a
   pinhole camera at a requested position/altitude/angle, with smoke sprites placed at ground
   coordinates under the same projection. Not a flat 2D compositor and not Blender.
   "Make 2D wildfire images" means this.

**One projection, two uses:** the projection `synthgen/` uses to place a sprite is the code
`sim/camera.py` uses to turn a UAV footprint into ground cells. Never fork them. State the
limitation in any write-up: smoke is a flat billboard, not a volume; no real 3D occlusion.
Rationale: `docs/synthetic/poc_report.md`.

## Training pipeline discipline (the most important process rule)

If the matrix is not trustworthy, nothing downstream matters. The eval harness must have:
1. **Unified annotation format** across Boreal and synthetic data: same box format, same class names.
2. **One evaluation script**: `(checkpoint, test set) -> metrics`, same code path for all four cells.
3. **Frozen train/val/test splits** per domain, decided once, no leakage. Never change a test set to
   improve a score.
4. **Config-driven runs**: four `(model, test-set)` pairs fed into one harness, not four scripts.
5. **Fixed seeds, logged hyperparameters**, comparable training budgets for Model A and Model B
   (Model A reuses Model B's recipe in `models/MODELS.md`, with the differences pinned in
   `docs/synthetic/synth_smoke_v1_spec.md` section 5).

The harness is `eval/metrics.py` (numpy metrics), `eval/run_matrix.py` (runner) and `eval/datasets.yaml` (pinned
cells). Model B was trained and tested with one-off scripts (`eval/runpod/`), which are not the harness.
The harness reproduced Model B within tolerance and produced the first 2x2 (`docs/experiments/level2_v1.md`); current status is in `docs/project-status.md`.

## Decisions that matter

- **Fog owns the fire.** Agents never run their own `FireSim`; importing it in `swarm_agent.py` is a regression.
- **Oracle off by default** (`fire_model.oracle_hazards: false`). On = perfect-information upper bound,
  not a perception result. Real Level 3 runs must have it off. Fog prints a loud banner when on.
- **Detections are scored at ingest** (`Fog.ingest_detection()`), because truth changes over time.
  `detections_<tag>.csv` is the Level 3 result file; `experiment_end_*.txt` has `precision` and `detection_lag_s`.
- **MAP bit index is `i*N + j`** in fog, dashboard and `comms/schema.py`. Grep all three before changing it.
- **FOG_STATE belief is sparse**: `[[i,j,p], ...]` above `BELIEF_FLOOR` (0.02). Check packet size against
  `MTU_WARN_BYTES` (`comms/gossip.py`) before adding fields; fragmentation fails silently.
- **Perception scoring is a placeholder** (`planning/priority.py`: `max(confidence)` plus repeat bonus). The
  replacement is a log-odds grid in `mapping/probability_map.py`, written *after* the eval matrix produces
  sensitivity/specificity; swap `PriorityPlanner.score()`'s body only.
- **`perception.from_config()` fails soft** to `NullPerception` with a loud warning. Zero detections in a run?
  Check that warning before blaming the detector.
- **Run reproducibility:** `fog/run_log.py` writes `manifest.json` per run (weights path, size, mtime, `model_tag`).
- **Dashboard** is a read-only stdlib `http.server` subscriber polling `/api/state`, drawing to a canvas.
  Adding FastAPI/WebSockets/Leaflet is a deliberate rebuild, not a drop-in swap.

## Layout (details and status per file: `docs/project-status.md`)

- Flight/ground side: `fog_tracker.py` + `fog/`, `swarm_agent.py`, `mission/`, `mapping/bitmap.py`, `planning/`,
  `comms/`, `dashboard/`, `wildfire_sim.py`, `launch_*.py`, `exp_swarm_*.yaml`.
- ML side: `sim/camera.py`, `synthgen/`, `eval/` (`validate_dataset.py`, harness), `eval/runpod/` (archived one-off launchers), `models/`, `tests/`.
- Still empty stubs (verify with `wc -l`): `perception/detector.py`, `sim/frame_synth.py`, `mapping/probability_map.py`.
- Large local data (datasets, backups, POC outputs) is under gitignored `local_data/`; checkpoints (`*.pt`) are never committed.

## Autonomous Development and Compute Policy

Claude Sonnet is authorized to independently develop, test, train, evaluate, and debug the wildfire
monitoring system, including using paid RunPod resources.

### Compute Authorization
- Total project RunPod budget available: $150.
- Autonomous spending allowance: $100 cumulative.
- Reserve remaining $50 for later experiments.
- No individual experiment may exceed $20 estimated cost.
- No more than one paid GPU pod at a time.
- Check current rates before launching resources.
- Record cumulative spending, including storage charges (ledger: `docs/project-status.md`).
- Stop before exceeding the authorized allowance.
- Obtain approval to increase any budget limit.

### Permitted Actions
- Launch, stop, and terminate RunPod compute instances.
- Upload datasets and execute GPU training.
- Run evaluations and benchmark experiments.
- Retry failed jobs when justified.
- Modify code and experimental configurations.
- Commit implementation changes and experimental reports.
- Generate research artifacts for independent Codex review.

### Research Integrity
- Preserve frozen dataset versions and original splits.
- Preserve Model B's existing trained checkpoint.
- Never fabricate experimental measurements.
- Record software versions, seeds, configurations, checkpoint hashes, and experiment commands.
- Maintain sufficient logs for reproducibility.
- Back up important artifacts before removing resources.
- Do not change test datasets to improve reported scores.

### Cost and Resource Management
- Prefer cost-effective GPUs suited to each experiment.
- Terminate idle GPU resources.
- Verify that jobs have completed before terminating pods.
- Account for storage charges even when GPUs are stopped.
- Stop and report repeated unexplained experiment failures.

### Reporting
After each substantial experiment, report: purpose; GPU and environment; training/evaluation
configuration; measured results; artifact locations; actual or estimated spending; remaining
autonomous budget; unresolved issues. Codex independently reviews experimental evidence at research checkpoints.

### Training practice
- The laptop has no GPU (dev, tests, CPU inference); RunPod is the GPU environment. Do not cap epochs for lack of a local GPU.
- Inspect curves and metrics before recommending changes; if not undertrained, look at data, labels, splits,
  domain gap, resolution, augmentation, capacity and thresholds. Prefer reproducible runs over minimal GPU cost.

### Active gates (owner decisions that outrank the authorization above until cleared)
- **Edge-touch gate: CLEARED 2026-10-27 for baseline Model A training only**, on these conditions: `synth_smoke_v1`
  stays exactly as frozen (no edits to data, splits or generator); its edge-touch mismatch with Boreal (~0.76 vs ~0.30)
  is reported as an experimental limitation; Model A on `synth_smoke_v1` is a baseline, not evidence that the synthetic
  domain is optimal; never overwrite existing checkpoints or change splits; any dataset improvement is a separately
  versioned dataset (v2, post-hoc). Before launching paid resources, reconcile RunPod compute and storage charges
  against the $100 allowance. Ordinary authorized experiments need no further approval.
- Gates are listed and cleared in `docs/project-status.md`; if silent, ask.

## Scope — do not expand into

- fire-spread prediction beyond the existing 2D cellular-automaton ground truth
- reinforcement learning / sophisticated multi-agent RL
- cloud deployment, FastAPI, PostgreSQL, AWS
- fully realistic 3D wildfire environments (Blender volumetrics); the generator is 2.5D

## Commands

```bash
python3 fog_tracker.py --origin-lat 21.2970 --origin-lon -157.8170 --grid-miles 1.0 --grid-cells 10 --config exp_swarm_fire.yaml
python3 dashboard/server.py --port 8080      # http://localhost:8080
python3 swarm_agent.py --conn udp:127.0.0.1:14555 --origin-lat 21.2970 --origin-lon -157.8170 \
  --mission grid_frontier --config exp_swarm_fire.yaml
python3 -m unittest discover -s tests -t .   # ~2 min (renders images); run in background
```

PyYAML required; pymavlink only for real MAVLink. SITL need not run in real time (`--speedup` < 1).
`launch_experiment.py` has stale paths (see project-status known issues).

## Git rules

Before significant changes: inspect `git status`; work only in this repo; keep logical commits; do not push unless asked.
Before ending a major task: run relevant tests/checks, review `git diff`, commit with a descriptive message
(stage specific files, since the tree may hold the user's uncommitted work), and update `docs/project-status.md`.
Where historical detail belongs: dataset facts in `docs/datasets/`, experiments in `docs/experiments/`,
checkpoints in `models/MODELS.md`, generator design in `docs/synthetic/`, progress in `docs/project-status.md`.
Keep this file near 200 lines (policy block included); add a new doc only when no existing one fits.

## Continuation (session restart after context/token limits)

1. Read this file, then `docs/project-status.md`, then `models/MODELS.md` if models are involved.
2. Run `git status` and inspect the latest commits.
3. Apply "Before starting any task" above; trust the more recently updated source, and the repo over both.
4. Continue from the existing state rather than restarting or rebuilding completed work.
