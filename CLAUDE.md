# Wildfire Monitoring Project — ECE 496 Fall 2026

AI-Enabled Autonomous Wildfire Monitoring.

**Research question:** does the training domain (synthetic vs. real imagery)
change how well a neural network perceives wildfire/smoke, and does that
difference propagate into autonomous UAV monitoring performance?

## Repository

Repo: `~/src/Ardupilot-UAV-YOLO` — the only repo Claude should modify, commit,
or push to.

Reference-only: `~/src/ardupilot` — upstream dependency. Do not commit, push,
or modify unless explicitly instructed.

## Three-level structure

**Level 1 — Domain comparison.** Model A: trained on synthetic wildfire/smoke
images (auto-generated, auto-labeled). Model B: trained on the real Boreal
Forest dataset. Metrics: precision, recall, mAP, false positive/negative rate,
confidence, localization error.

**Level 2 — Cross-domain generalization.** A 2×2 matrix (Synth→Synth,
Synth→Real, Real→Synth, Real→Real), all four cells evaluated by the *same*
code path. See "Training pipeline discipline" below — this matrix is the
actual deliverable; everything else is infrastructure in service of it.

**Level 3 — Perception → autonomous monitoring.** Closed loop on ArduPilot
SITL: `UAV Camera → Neural Network → Detection Probability → Geographic
Probability Map → Priority Region → UAV Coverage Decision`. A known
ground-truth fire is placed in sim; UAVs fly and detect; detections accumulate
into a spatial probability map; probability drives priority (low → low
priority, high → investigate); priority drives where UAVs fly next. Compare
system-level outcomes (time to detection, coverage efficiency, false-positive
misdirection) across Model A vs. Model B.

This data-flow (camera → NN → probability → priority → coverage) is already
built: `perception/base.py` (camera → NN seam), `comms/schema.py`
(`DETECTION`/`BELIEF` messages), `planning/priority.py` (probability →
priority → target cell). What's still a placeholder is the probability math
itself — see TODO.

## Two different "2D"s — do not conflate them

This project has two unrelated things that are each "2D" for different
reasons. A past planning pass collapsed them into one scope bullet
("2D wildfire simulation... not fully realistic 3D"), which is ambiguous
enough to cause real damage if misread. Spelled out:

1. **Fire-spread model** (`wildfire_sim.py::FireSim`) — a 2D cellular
   automaton over the coverage grid (unburned/burning/burnt per cell, von
   Neumann or Moore neighbors). This is correct, intentional, and already
   built. It is a physics abstraction, not imagery. Keep it 2D.

2. **Synthetic training imagery** — NOT the same decision. The chosen
   approach (confirmed in prior planning) is **2.5D**: a ground-plane forest
   texture projected through a pinhole camera model at a requested
   position/altitude/angle, with fire/smoke sprites placed at ground
   coordinates under the same projection, plus haze/lighting/color variation.
   This is deliberately in between the two things scope excludes:
   - *not* naive flat 2D compositing — that can't vary camera altitude/angle
     and provides no 3D pose → geo-projection, which Level 3 needs directly
   - *not* full Blender/3D — physically accurate but slow on CPU-only
     hardware, and a weak volumetric scene risks making the Synth→Real
     result reflect renderer weakness rather than a real domain-transfer
     finding

   The payoff of 2.5D: exact labels and geo-locations for free (you placed
   the sprite, you know its ground truth), renders in milliseconds (thousands
   of images feasible on a laptop), and — this is the part that matters for
   Level 3 — **the same projection code the generator uses to place a sprite
   is the code `sim/camera.py` uses to turn a UAV's camera footprint back
   into ground cells.** One projection, two uses. Don't let these drift into
   two separate implementations.

   State the limitation explicitly in any write-up: smoke is a flat
   billboard, not a true volume; no real 3D occlusion.

If a future session is asked to "make 2D wildfire images" or similar, it
means (2), done as 2.5D — not a flat compositor, and not Blender.

## Repository layout

```
fog_tracker.py            ground station entry point
fog/state.py               world model: coverage, fire truth, detection scoring
fog/run_log.py             CSVs, manifest.json, run folder
fog/config.py              YAML/JSON loading

swarm_agent.py             one drone's control loop
mission/geo.py             projection, Chebyshev distance
mission/grid_mission.py    TRAVERSE coverage: frontier choice + edge sweeps
mission/partitions.py      static area assignment (quadrants/stripes/halves)
mapping/bitmap.py          visited/done bitmaps, neighbour merging
planning/priority.py       detections -> a cell worth investigating (placeholder scoring)
perception/base.py         NN seam: Pose in, Detection out. NullPerception is the default.

comms/schema.py            every message shape on the bus, with make_* producers
comms/gossip.py            UDP multicast bus (shared by agent/fog/dashboard)

dashboard/server.py        read-only HTTP+bus subscriber (rx_only — cannot transmit)
dashboard/static/          vanilla JS/CSS/HTML, no build step, no CDN deps

custom_vehicle.py          MAVLink wrapper
swarm_strategies.py        fallback velocity policies
wildfire_sim.py            FireSim cellular-automaton fire model (2D grid — see above)
launch_experiment.py       orchestrates fog + N agents from one YAML (has a bug — see TODO)
launch_sitl_swarm.py       spins up real ArduCopter SITL instances in tmux
exp_swarm_*.yaml           experiment configs
```

Not yet created: `sim/`, `synthgen/`, `eval/`, `models/`.

**Dashboard stack note:** this is a plain `http.server` subscriber with the
browser polling `/api/state` every 250ms, drawing to a `<canvas>` grid — not
FastAPI/WebSockets/Leaflet/CesiumJS. That's consistent with the Scope section
below (no FastAPI), zero new dependencies, and works fine on a CPU-only
laptop. If a georeferenced map view becomes a real requirement, that's a
deliberate rebuild of the dashboard layer, not a drop-in swap — decide that
explicitly before starting it.

## Training pipeline discipline (do this before more ML or more UAV work)

**This is the most important process rule in this file.** The 2×2 matrix is
the actual deliverable. Everything else — renderer fidelity, UAV flight
smoothness, dashboard polish — is infrastructure in service of that table. If
the table isn't trustworthy, nothing downstream matters.

Requirements for the eval harness to be genuinely reproducible:
1. **Unified annotation format** across Boreal and synthetic data — same box
   format, same class names — so one loader/eval script works on either
   without special-casing.
2. **One evaluation script**, not per-cell notebooks. `(checkpoint, test set)
   -> same metrics, same code path`, every time. Mixing code paths across the
   four matrix cells contaminates the comparison.
3. **Fixed, frozen train/test splits** per domain, decided once, reused for
   every experiment. No leakage.
4. **Config-driven runs** — the four experiments are four `(model, test-set)`
   pairs fed into one harness, not four scripts.
5. **Fixed seeds, logged hyperparameters**, so re-running with more epochs
   later is comparable to the first pass.

**Known status / departure from the recommended build order:** the original
plan was eval harness + even a crude placeholder generator *first*, with
ArduPilot/dashboard work starting only once a real checkpoint per domain
existed to plug in. What's actually been built is the reverse — fog, the
swarm agent, mission logic, and the dashboard are substantially done, while
the unified annotation schema, the eval harness, and the synthetic generator
have not been started. This was a reasonable call to get something demoable,
but it means the harness and schema are now overdue relative to the plan's
own stated reasoning, not just the next item on a list. Do this next, before
expanding UAV/dashboard features further.

## Decisions that matter

**Fog owns the fire. Agents never run their own FireSim.** Two independent
RNGs means no single ground truth. If `FireSim` is ever imported in
`swarm_agent.py`, that's a regression.

**The oracle is off by default** (`fire_model.oracle_hazards: false`). When
true, fog broadcasts ground-truth `HAZARD_UPDATE` to every agent — that's the
perfect-information upper bound, not a perception result. Real Level 3 runs
must have this off, or YOLO is decoration. Fog prints a loud banner whenever
it's on.

**Detections are scored at ingest, not after the run.**
`Fog.ingest_detection()` checks truth the instant a `DETECTION` message
arrives, because truth changes over time (a cell correct at t=40s may be
burnt out by t=90s). `detections_<tag>.csv` is the Level 3 result file.
`experiment_end_*.txt` carries `precision` and `detection_lag_s`.

**MAP bit index is `i*N + j`**, matching fog, the dashboard, and
`comms/schema.py`. A prior version used `j*N + i` in the agent only — it was
self-consistent but would silently mirror coverage about the diagonal the
moment anything else read an agent's map. Grep all three files if this ever
needs to change again.

**FOG_STATE belief is sparse-encoded**: `[[i,j,p], ...]` above `BELIEF_FLOOR`
(0.02), not a dense grid. A dense grid at N=10 alone exceeded the single-UDP-
datagram budget (`MTU_WARN_BYTES` in `comms/gossip.py`). Check packet size
against that constant before adding fields to FOG\_STATE — fragmentation
degrades silently, not loudly.

**Perception scoring is a placeholder.** `planning/priority.py` scores a cell
as `max(confidence)` plus a small repeat bonus. The real replacement is a
log-odds Bayesian grid in `mapping/probability_map.py`, which needs
sensitivity/specificity from the Level 2 confusion matrices — write it
*after* the eval matrix runs, then swap `PriorityPlanner.score()`'s body only.

**`perception/base.py.from_config()` fails soft** — missing weights, missing
torch, bad config all fall back to `NullPerception` with a loud warning
instead of crashing agents mid-flight. Zero detections in a run? Check this
warning before assuming the detector is broken.

**Reproducibility:** `fog/run_log.py` writes `manifest.json` per run (model
weights path, size, mtime, `model_tag`). Level 2's matrix is only meaningful
if both models get comparable training budgets — log that into
`models/MODELS.md` once it exists.

## Training / Compute Policy

The project is not limited to CPU-only training.

The development laptop is CPU-only, but cloud GPU resources may be
used for substantial YOLO training runs. The user is willing to pay
for cloud GPU compute when it materially improves the quality or
reliability of the experiment.

Do not artificially restrict training to a small number of epochs
because the development machine has no GPU.

When evaluating a model, inspect training/validation curves and
metrics before recommending architectural changes.

If a model appears undertrained, recommend additional training when
supported by the evidence.

If additional epochs are unlikely to solve the problem, investigate
other causes such as:
- dataset quality
- class imbalance
- annotation quality
- train/validation split
- domain gap
- image resolution
- augmentation
- model capacity
- confidence/IoU thresholds

For important experimental results, prioritize reproducible,
well-documented training runs over minimizing GPU cost.

Record relevant training configuration, dataset split, model version,
epochs, and metrics so experiments can be compared fairly.

## Hardware / environment constraints

- No GPU on the development laptop — CPU-bound for inference and the 2.5D renderer.
- RunPod is the primary cloud training environment. Colab/Kaggle may be used as alternatives when convenient. The user is willing to pay for GPU compute when it materially improves training speed or experiment reliability.
- Dev environment: WSL2 (Ubuntu) + VS Code via the WSL extension. Keep the
  repo inside the Linux filesystem (`~/src/...`), not `/mnt/c/...`.
- SITL does not need to run in real time — slow it down (`--speedup` < 1) to
  give render + inference latency room per frame.
- Already have: Boreal Forest dataset in use; a 3-epoch YOLO checkpoint
  (demo-only — not representative of a final Level 1/2 result).

## Fixed bugs (so they don't come back)

- `launch_experiment.py` defines `agent_cmd` twice (dead first copy, ~line 176).
- An older dashboard emitted `visited_cnt` but never a `visited` list, so
  coverage drew nowhere. Fixed by switching the wire format to a hex bitmap.
- Endgame stall: with one cell left, every drone's frontier search returned
  the same cell and treated it as someone else's claim, idling until timeout.
  `mapping/bitmap.py: single_remaining()` detects this and force-targets it.

## Outstanding TODO (rough priority order)

1. **Unified annotation schema + eval harness — overdue, start here.** Common
   box format and class names across Boreal and synthetic data; one
   `(checkpoint, test-set) -> metrics` script; frozen splits; config-driven
   four-cell runs; fixed seeds and logged hyperparameters.
2. **`launch_experiment.py` path fix.** Lines ~26–27 point at
   `core/swarm_agent.py` / `core/fog_tracker.py`; there is no `core/`. Change
   both to `HERE / "swarm_agent.py"` and `HERE / "fog_tracker.py"`. Direct
   invocation of each file already works and is how everything above was
   tested.
3. **`synthgen/`** — the 2.5D generator (pinhole projection, ground-plane
   texture, sprite placement, auto-labels). This is the heaviest remaining
   piece — open-ended content generation, not a spec-checkable implementation
   like the rest of this list. Build the projection math so it is directly
   reusable by (4).
4. **`sim/camera.py`** — UAV pose + altitude + FOV → ground footprint in grid
   cells, using the *same* projection as (3).
5. **`perception/detector.py`** — thin YOLO wrapper implementing `observe()`.
   Lightest item on this list; the trained checkpoint already exists.
6. **`mapping/probability_map.py`** — log-odds belief grid. Blocked on (1)'s
   confusion matrices.
7. **Demo vehicle** (`sim/fake_vehicle.py`) — a `SimVehicle` matching
   `CustomVehicle`'s public surface via a `sim:lat,lon,offset_n,offset_e,seed`
   connection string, for showing the dashboard live without SITL running.
   Not required for the research deliverable; useful for demos.
8. Promote ad-hoc test scripts (grid round-trip, partition overlap, bitmap
   merge, full fly-through against a fake vehicle) into a committed `tests/`
   directory.

## Scope — do not expand into

- fire-spread prediction (beyond the existing 2D cellular-automaton ground
  truth)
- reinforcement learning / sophisticated multi-agent RL
- cloud deployment, FastAPI, PostgreSQL, AWS
- fully realistic 3D wildfire environments (full Blender volumetrics) — the
  synthetic generator is 2.5D, see above, not this

## Commands

```bash
# ground station + dashboard (two terminals)
python3 fog_tracker.py --origin-lat 21.2970 --origin-lon -157.8170 \
  --grid-miles 1.0 --grid-cells 10 --config exp_swarm_fire.yaml
python3 dashboard/server.py --port 8080      # http://localhost:8080

# one real/SITL drone (needs pymavlink + a MAVLink endpoint)
python3 swarm_agent.py --conn udp:127.0.0.1:14555 \
  --origin-lat 21.2970 --origin-lon -157.8170 --mission grid_frontier \
  --config exp_swarm_fire.yaml
```

PyYAML is required (`pip install pyyaml`); pymavlink only for real MAVLink
connections.

## Git rules

Before making significant changes:
1. Inspect `git status`.
2. Work only inside `Ardupilot-UAV-YOLO`.
3. Keep changes in logical commits.
4. Do not push to GitHub unless explicitly requested.

Before ending a major task:
1. Run relevant tests/checks.
2. Review `git diff`.
3. Commit with a descriptive message.
4. Update `docs/project-status.md` (create it if it doesn't exist yet).

## Continuation (session restart after context/token limits)

1. Read this file.
2. Read `docs/project-status.md`.
3. Run `git status`.
4. Inspect the latest commits.
5. Determine what was completed and what remains (cross-check against
   "Outstanding TODO" above — it may be stale if `docs/project-status.md`
   disagrees; trust the more recently updated one).
6. Continue from the existing project state rather than restarting.
