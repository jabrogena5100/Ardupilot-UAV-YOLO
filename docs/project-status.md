# Project Status — ECE 496 AI-Enabled Autonomous Wildfire Monitoring

**Last updated:** 2026-11-09 (Level 3 final evaluation complete, awaiting Codex audit; earlier: Codex REVISE accepted: Boreal split found NOT video-disjoint, Level 2 v2 report corrected, Level 3 observation contract drafted; Level 3 not started; awaiting second Codex review)

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

**Synthetic dataset:** `synth_smoke_v1` generation is **COMPLETE and validated** (5,304 images, 2026-10-05; record in `docs/datasets/synth_smoke_v1.md`). Local copy: `local_data/datasets/synth_smoke_v1/`; a hash-verified copy is on the RunPod volume at `/workspace/synth_smoke_v1/` (uploaded 2026-10-27). Model A was trained on it 2026-10-27 (below). It was built with the project's 2.5D synthetic image generator using:

* ground-plane forest texture
* pinhole camera projection
* fire/smoke sprites placed at known ground coordinates
* camera position / altitude / angle variation
* lighting, haze, visibility, and color variation
* automatic ground-truth labels

**Models:**

| Model | Checkpoint | State |
|---|---|---|
| Model A — synthetic-trained YOLO | `synth_smoke_best_v1.pt` (seed 0, SHA `1da10c48…009564`) + `_seed1`, `_seed2` | **Three seeds trained** (2026-10-27/28, 100 epochs each). Synthetic test mAP50 0.995 (saturated); Boreal test mAP50 0.375 +- 0.059 (mean +- SD, n=3). `docs/experiments/level2_v2.md`, `models/MODELS.md`. |
| Model B — real/Boreal-trained YOLO | `real_boreal_best_v1.pt`, SHA-256 `1dc35ef2…8355b6` | **Trained and tested once** (2026-10-04): test mAP50 0.944, mAP50-95 0.669. Details: `models/MODELS.md`, `docs/experiments/real_boreal_v1.md`. |

**Legacy "3-epoch demo checkpoint":** earlier versions of this file and `CLAUDE.md` said a 3-epoch demo checkpoint exists. No such file is in the working tree, and nothing in `models/MODELS.md` or the experiment record describes it. The only trace is git history: `models/real_boreal_bestv1.pt` (5,442,970 bytes, SHA-256 `05ef7d46…47a54`, commit `73dab87`, deleted in `23dc32a`) and a 0-byte `models/synthetic_best.pt` placeholder. The 5.4 MB blob is not Model B v1 (22.5 MB, different hash). Its epochs, architecture and data are **unrecorded**; do not call it Model A or B or cite it as a result. See "Open contradictions".

**Status:** Three seeds of each family exist and are evaluated through the harness. Edge-touch mismatch remains a stated limitation.

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

**Status:** Harness implemented, validated against Model B (reproduces its Ultralytics test mAP50 0.9442 / mAP50-95 0.6688 as 0.9511 / 0.6730), and used for the 3-seed 2x2 (2026-10-28, `docs/experiments/level2_v2.md`; seed 0 = `level2_v1`):

| train -> test | mAP50 mean +- SD (n=3) | per seed | mAP50-95 mean | recall@0.25 mean | FP/img@0.25 mean |
|---|---|---|---|---|---|
| Synth -> Synth | 0.995 +- 0.000 | 0.995, 0.995, 0.995 | 0.986 | 1.000 | 0.000 |
| Synth -> Real | 0.375 +- 0.059 | 0.429, 0.384, 0.313 | 0.164 | 0.519 | 0.436 |
| Real -> Synth | 0.759 +- 0.107 | 0.753, 0.656, 0.869 | 0.390 | 0.627 | 0.093 |
| Real -> Real | 0.954 +- 0.003 | 0.951, 0.956, 0.953 | 0.675 | 0.947 | 0.070 |

**Correction 2026-10-29 (Codex REVISE):** the Boreal split is not video-disjoint (3 clips shared train/test, 2 train/val, via 197 background-only `Empty-Images` train frames); no metric value changed, and a test-subset sensitivity shows no consistent inflation of Real -> Real (`docs/experiments/level2_v2_audit.md`). The "seed variability dominates sampling uncertainty" claim is withdrawn (synthetic test scenes share 6 texture slots; Boreal test has 5 videos). "Optimizer steps" were minibatches; the optimizer made about 5,532 updates per run.

SD is across seeds; bootstrap CIs (images; plus source videos for Boreal) are test-set sampling only and are reported separately. Synth -> Synth is saturated (cannot discriminate). Synth-trained models fire on 14-50% of real empty scenes (Boreal `Empty-Images`, train-split diagnostic) while giving 0% on synthetic negatives, so synthetic negatives alone overstate specificity. Boreal test has effectively no negatives. Protocol audit (equal epochs/steps/schedule, runtime gap is a data/host effect), bootstrap verification and limitations are in the experiment report.

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
* `perception/detector.py` (`YoloPerception`: SHA-256-verified weights, Ultralytics CPU, box -> grid cell) and `perception/frames.py` (HTTP frame client) exist; wired and demonstrated with one UAV on branch `integration/yolo-perception-prep` (2026-11-02, wiring only, no Level 3 result)
* `perception/base.py`'s docstring example now names the Boreal v2 seed 0 default; `perception.strict: true` makes a failed init raise instead of falling back
* `sim/camera.py` — shared pinhole projection, round-trip tests; confirm `tilt_deg`/`fov_deg` conventions before Level 3 use
* `sim/frame_synth.py` (virtual nadir camera over the unchanged synthgen renderer) and `sim/camera_service.py` (fog-side frame service; truth never leaves fog) exist; settings (nadir, conf 0.25, plume model) are provisional

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

### Edge-touch gate — CLEARED 2026-10-27 (baseline Model A only; Model A baseline now trained)

The owner cleared the `synth_smoke_v1` edge-touch review (about 0.76 / 0.76 / 0.77 vs Boreal train 0.30) for baseline Model A training, with conditions:

1. `synth_smoke_v1` is preserved exactly as frozen (tag `synth-v1`, `9cb0f1a`).
2. The edge-touch mismatch with Boreal is documented as an experimental limitation in every write-up.
3. Model A on `synth_smoke_v1` is the **baseline** experiment, not proof the synthetic domain is optimal.
4. Frozen train/val/test splits are not modified and no existing checkpoint is overwritten.
5. Synthetic dataset improvements are separately versioned (`synth_smoke_v2`, reported as post-hoc).
6. RunPod compute and storage charges are reconciled against the $100 autonomous allowance before paid resources are launched.
7. Ordinary authorized experiments proceed within the `CLAUDE.md` limits without further approval.

The generated YAML/list files contain the generation pod's absolute paths, so runs use path-rewritten copies (`data_rw.yaml`; the harness reads image directories directly); never edit the hashed originals. The pre-declared diagnostic (Model A on Boreal test, grouped by edge-touching vs not) was run: recall 0.873 edge-touching (n=142) vs 0.497 not (n=571), reported as consistent with, not proof of, the mismatch.

### Priority 1 — Independent review of the Level 2 protocol (next; blocks Level 3)

**2026-10-29 update:** Codex returned REVISE; the correction phase is done (`docs/experiments/level2_v2_audit.md`): split memberships reconstructed from a pod scan of the source dataset (shared clips found, no exact duplicates), report corrections applied, backup manifest repaired, provenance limits documented, `docs/experiments/level3_observation_contract.md` drafted (proposal only, nothing implemented). **Next: second Codex review of that package.** Open decisions for the owner: (a) report Level 2 with the documented overlap and sensitivity, or build a clip-level version-2 Boreal split and retrain Model B (about $0.9 per seed on a 4090; the frozen split itself must not be edited, a v2 would be a new versioned split); (b) real-domain negatives and cell-level truth for Level 3 (contract sections 3 and 6). Do not implement `mapping/probability_map.py` before the contract's section 5 is measured.

Original plan (kept for context):

Level 2 evidence is now: 3 seeds per family, pinned datasets/checkpoints, saved predictions and ground truth, image and video-cluster bootstrap CIs, a negatives diagnostic, and a documented protocol audit (`docs/experiments/level2_v2.md`, artifacts in `docs/experiments/level2_v2/`, hashes in `models/MODELS.md`). Per the owner's instruction Level 3 does not start until this protocol is validated. Next steps:
1. **Codex review** of `docs/experiments/level2_v2.md` and its artifacts (checkpoint hashes, recompute from `local_data/backups/level2_v2_seeds/level2_v2/{predictions,gts}`, the audit claims). Address findings before anything else.
2. Open methodological items to decide with the owner: (a) the Boreal test set has no usable negatives and the frozen split must not change, so real-world specificity can only be probed on the train-split `Empty-Images` diagnostic (valid for Model A only); (b) n = 3 seeds is small, so SDs are rough and no significance claim is made; more seeds would be a cheap extension (about $0.5 per Model A seed, about $0.9 per Model B seed) if the reviewer wants tighter variance; (c) Real -> Synth is highly seed dependent (0.656-0.869) and deserves per-archetype/per-size breakdowns from `metadata.jsonl` rather than more data changes.
3. Only after review: derive sensitivity/specificity (per family, with seed spread and the negatives caveat) for `mapping/probability_map.py` (Priority 5).

Built and tested: `eval/metrics.py`, `eval/run_matrix.py`, `eval/aggregate_seeds.py`, `eval/datasets.yaml`, `tests/test_eval_metrics.py` (13), `tests/test_run_matrix.py`, `tests/test_aggregate_seeds.py`, pod scripts `eval/runpod/level2_*.sh`. A perfect detector scores 0.995 (Ultralytics convention).

### Priority 2 — Model A improvements are a separate dataset version

Anything that changes the synthetic data (for example reducing the edge-touch rate) is `synth_smoke_v2`, generated, hashed and reported as post-hoc, with its own Model A. `synth_smoke_v1` and its splits stay frozen.

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
* **Runtime gap (resolved 2026-10-28):** the 2.3x A-vs-B wall-time gap is a data and host effect (4096x2160 Boreal frames, 20x the bytes of the 1280x720 synthetic frames, CPU-bound dataloader; the same Model A recipe also ran 4.6x slower on one slow host); epochs, optimizer steps and schedule are identical (`docs/experiments/level2_v2.md` section 1.2).
* **Boreal "video-disjoint" (found 2026-10-29, unresolved):** earlier docs said 0 shared videos; with the site prefix normalised, 5 source clips are shared across splits (`level2_v2_audit.md` section 1). Pixel-level confirmation and a v2 split are open.
* **Inference provenance (2026-10-29):** pod cells record `git.commit: null` (ran a non-git copy); exact `predict()` source of the saved predictions is unrecorded (v1 and v2).
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
| Network volume `wildfire-datasets` (`sej40ez3ia`, 50 GB STANDARD, US-IL-1), 2026-10-01 to 2026-10-27 | $2.79 (accrues about $0.10-0.12/day while it exists) |
| Level 2 v1 pod `h88h3d09647945` (22:57-about 23:27 UTC 2026-10-27) | posted: about $0.51 (2026-10-27 GPU charge) |
| Level 2 v2 pods `3yf63rbjjtl6x2` (slow host, aborted, about 33 min), `kp3n3sqb685hiy` (slow host again, aborted, about 9 min), `kn5e6x2mo3r52x` (about 120 min, completed) | about $2.4 by running time; billing had posted only about $1.6 when read |
| Boreal integrity scan pod `n5kg8sii9fcrp5` (RTX 4090 $0.89/h, US-IL-1, created 03:37 UTC 2026-10-29, scan about 20 s, terminated minutes later; not yet billed at last read) | not posted; allowance $0.25 (pod time under 17 min) |
| **Billed October total at last read (2026-10-29)** | **$9.08** (pods GPU $5.98 + CPU $0.15 + disk $0.03, storage $2.92); earlier read 2026-10-28: $8.21 |
| **Conservative cumulative** (billed + scan pod allowance) | **about $9.3** |
| **Remaining autonomous allowance** | **about $90.7** (billing-only figure: $90.92) as of the 2026-10-29 morning read, **before the Boreal v2 experiment** |
| Boreal v2 experiment, 2026-10-29 (4 attempts + 7 diagnostic/probe pods; estimated from pod start/stop times x $0.89/h) | **$2.93** estimated; billing had posted $11.27 total at 22:50Z (lags by hours) |
| Billing read 2026-11-02 | **$12.35** total billed (posts hours late; includes the Boreal v2 pods and storage); no pod running |
| **Conservative cumulative after Boreal v2** | **about $12.3** (9.3 + 2.93 + about $0.1 storage); remaining autonomous allowance **about $87.7**. Re-read billing before the next launch; no pod is running (all EXITED; 7 stopped probe/diagnostic pods and 4 attempt pods remain, removable by hand) |
| Checkpoint backup pod `g02e1nlir1927z` (RTX 4090 secure, $0.89/h, US-IL-1, created 01:35:05 UTC 2026-11-02, deleted after the transfer; run time about 2 min, no billing read yet) | allowance $0.05 (estimate under $0.04); **cumulative about $12.4, remaining autonomous about $87.6**; volume untouched, pod deleted, 0 running |

Corrections: the earlier per-experiment estimates in this file ("about $0.60" for Model B, "about $0.13" for generation) came from pod run time and undercount; billing shows about $3.04 for 2026-10-04 (two pods with $1.38 and $1.07 of GPU charges are not individually documented) and $0.30 for 2026-10-05. The experiment documents keep their original estimates; this ledger is authoritative for spend. Pod count after the 2026-10-29 scan: 0 running (`list-pods` empty). Next session: re-read billing (it lags by hours) and replace the estimate with the billed figure. Storage keeps accruing about $0.12/day for the 50 GB volume.

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

**Update 2026-10-27 (later):** gate cleared by the owner; evaluation harness built and validated against Model B; Model A baseline trained and the first 2x2 produced (`docs/experiments/level2_v1.md`). Next: confidence intervals and seed variance (Priority 1).

**Update 2026-10-27:** `CLAUDE.md` consolidated into a short operating manual; stale claims in this file corrected (generator, camera and Model B status, 3-epoch checkpoint, launcher bug status). Checkpoint, dataset and code were not touched, and no paid resources were used.

**Current highest-priority task:**

Quantify uncertainty of the first 2x2 (bootstrap CIs, then extra seeds) before drawing or using any conclusion.

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

### 2026-10-29 — RunPod orchestration tooling (no training, no paid pod, $0 spent)

Branch `cloud/runpod-orchestration`; details `docs/runpod/orchestration.md`, `docs/handoffs/from_cloud_2026-10-29.md`. Notifications dropped; status
is manual (`launch_job.py status`). Built: launcher + budget gate, in-pod supervisor/watchdog, stateless off-pod watchdog
(`watchdog.py` + GitHub Actions every 5 min, deadline encoded in the pod name, append-only `runpod-logs` branch), disposable lifecycle job.
Verified: API key, reads, volume, stop endpoint authorization, live `enforce` (no pods), 49 mock tests. **Not verified (blocks "ready")**: GitHub watchdog
armed (needs merge to `main` + `RUNPOD_API_KEY` secret), live stop on a real pod, create-pod fields. Lifecycle test prepared, worst case $0.56, awaiting owner approval.
Boreal v2 split + Level 2 evaluation remain the next scientific task, after the lifecycle test passes.

### 2026-10-29 — Boreal v2 (clip-disjoint split): COMPLETE (3 Model B seeds trained, Level 2 re-evaluated)

Records: `docs/experiments/boreal_v2.md` (split), `docs/experiments/boreal_v2_results.md` (results and inventory), artifacts `docs/experiments/boreal_v2/results/`, handoff `docs/handoffs/from_cloud_2026-10-29.md` Update 6.
Split frozen first (31 clips; train 3462 / val 748 / test 744; video-disjoint; 22 residual near-duplicate train/test frame pairs documented). Model B v2 seeds 0-2 trained with the Level 2 v2 protocol plus a documented
speed-only deviation (`cache=ram`, pixel-identical checked); Model A not retrained. Revised 2x2 mAP50 (mean of 3 seeds): Synth->Synth 0.995 (saturated), Synth->Real 0.512 +- 0.095, Real->Synth 0.612 +- 0.104, Real->Real 0.899 +- 0.027.
Clip-clustered intervals (8 clips) are wide; real empty-scene FP rate (65 held-out frames): Model A 0.14-0.26, Model B-v2 0.015. Not comparable with Boreal v1 numbers. Estimated GPU spend $2.93 (4 attempts + 7 diagnostic pods); pods all EXITED.
Infrastructure learned: pod-injected key cannot REST-stop (403) but GraphQL `podStop` works (now the supervisor fallback, verified in 3 real runs); RunPod rejects a 300 KB env (60 KB ok); host speed varies 6x (shared CPU).
Checkpoints are on the volume only (hashes in `models/MODELS.md`); local backup is an owner step (`docs/handoffs/checkpoint_backup.md`).
**Post-experiment (2026-11-02):** self-audit PASS with documented caveats (`docs/experiments/boreal_v2_audit.md`, 49/49 checks, `eval/boreal_v2_audit.py`); branch `science/boreal-v2` pushed for preservation (not merged); ArduPilot integration plan written, nothing implemented (`docs/handoffs/ardupilot_integration_plan.md`).

### 2026-11-02 — Perception integration v0 (no training; $0 beyond the backup pod)
Branch `integration/yolo-perception-prep`. Virtual camera + YOLO adapter + fog-side frame service + paired-run planner (`eval/level3_pairs.py`) implemented and tested; CPU benchmark of all six registry checkpoints on this VM (about 0.3 s YOLO + 0.5 s render per frame, about 740 MB peak); one single-UAV end-to-end run (51 detections scored at ingest). **No Level 3 result and no claim about coverage, detection time or mission performance.** This 1 vCPU / 2 GB VM cannot carry multiple UAVs with in-process models. Record, findings and next step: `docs/experiments/perception_integration_v0.md`.

### 2026-10-27 — Level 2 v1: harness validation, Model A baseline, first 2x2

* Gate cleared by the owner (7 conditions, above). Built `eval/metrics.py`, `eval/run_matrix.py`, `eval/datasets.yaml`, 13 tests.
* RunPod 1x RTX 4090 (US-IL-1, $0.89/hr, about 30 min, pod terminated and verified): dataset hashes verified (10,616 files, before and after), harness reproduced Model B (mAP50 0.9511 vs 0.9442), Model A trained with Model B's recipe (100 epochs, 996 s, SHA `1da10c48…009564`), full 2x2 produced.
* First launch failed before training on PEP 668 (`pip` refused); fixed and rerun from a clean state (log kept in `docs/experiments/level2_v1/`).
* Results, interpretation and limitations: `docs/experiments/level2_v1.md`. Checkpoint backed up locally with hashes (`local_data/backups/synth_smoke_a_v1/`). Model B untouched.

### 2026-10-28 — Level 2 v2: seed replication, audits, uncertainty

* Pre-GPU audits: billing reconciled; training protocol audited for all six runs (equal epochs, steps, schedule, no early stopping; runtime gap explained); bootstrap verified by tests (point estimate equals full metric, resample equals full recompute) and its assumptions documented; background-image audit (Boreal test has 1 unlabeled frame, synthetic test has 350 true negatives).
* Trained seeds 1 and 2 for Model A and Model B (identical protocol, new checkpoint names, existing checkpoints untouched, dataset hashes verified before and after). Two launches landed on a slow host (45 s/epoch) and were aborted; a speed guard was added to the pod script. Third pod completed all four runs (about 1000 s per Model A, 2180 s per Model B).
* All 24 cells evaluated through the harness; metrics and 1,000-resample bootstraps recomputed offline from saved predictions and ground truth. Results, interpretation and limits: `docs/experiments/level2_v2.md`. Pod terminated; artifacts backed up with hashes.
* Not done on purpose: Level 3, any change to frozen splits or `synth_smoke_v1`.

### 2026-10-29 — Level 2 v2 correction phase (Codex REVISE)

* Boreal split integrity: read-only scan of all 4,954 source images on a short GPU pod (no CPU pod existed in US-IL-1; about $0.25 allowance, terminated). Scan reproduces the pinned file-list hashes. Found 3 train/test and 2 train/val shared source clips (197 background-only `Empty-Images` train frames), 0 MD5 duplicates, no same-clip near-duplicates; sensitivity on affected/unaffected test videos shows no consistent inflation for Model B. Frozen splits, test sets and all checkpoints untouched.
* Report corrections in `docs/experiments/level2_v2.md` (optimizer updates 5,532 vs 10,900 minibatches; bootstrap claim withdrawn, texture-slot cluster bootstrap added; reproducibility narrowed to what was re-verified: offline recompute reproduces all 1,602 committed values; inference provenance limitation documented); stale backup `SHA256SUMS` repaired (old copy kept); correction notes added to `real_boreal_v1.md` and `models/MODELS.md`. `synth_smoke_v1_spec.md` line 25 carries the same 10,900 wording and was not touched (uncommitted owner edits there).
* `docs/experiments/level3_observation_contract.md`: proposal only; no probability-map code written.
* New code/tests: `eval/boreal_split_audit.py`, `eval/boreal_leak_sensitivity.py`, `eval/synth_cluster_bootstrap.py`, `tests/test_boreal_split_audit.py`.

**Level 3 calibration (2026-11-02, branch `integration/yolo-perception-prep`):** methodology, held-out/calibration split, 60 deg camera convention, per-arm thresholds (F0.5 rule), frozen fire settings, deterministic planner with investigate cooldown, start gate and run provenance are implemented and tested (226 tests OK). Calibration pilot (`docs/experiments/level3_methodology.md` section 10) is **non-informative**: 0 true detections in every arm, identical coverage, because the coverage baseline never reaches the fire in 600 s. Blocker: scenario geometry/duration must be revised and re-piloted on calibration scenarios before any held-out run. No Level 3 performance claim.

**Level 3 corrected pilot (2026-11-02):** geofence sign bug fixed (it had pinned the baseline outside the grid), calibration ignition v2 chosen by camera geometry, Model B thresholds re-derived on an extended grid (0.01/0.01/0.02), six-run pilot (baseline, Model A seed 0, Model B v2 seed 0 x scenarios 1001, 1002) is non-trivial: detections, false alarms and detection-driven route changes differ by arm. n = 1 per cell, about 20 s timing jitter, no performance claim. Open: held-out ignition re-derivation (B1), repeatability (B2). Details: `docs/experiments/level3_methodology.md` sections 12-14.

**Level 3 replay dashboard (2026-11-02):** read-only browser view of saved pilot runs (`dashboard/replay_server.py` on port 8081, loader `dashboard/replay_data.py`, UAV paths cached from `sitl_work/mav.tlog` by `dashboard/replay_build.py` into gitignored `local_data/level3/_replay_cache/`). Lists completed / interrupted / not-run separately, replays path + fire + coverage + detections, compares baseline / A0 / B2s0 per scenario; missing data is shown as missing. Reads files only (no SITL, no YOLO, ~22 MB RSS). Pilot 2 state at this point: scenarios 1001 and 1002 complete for all three arms (6 runs); 1003 and 1005 (6 runs) not yet flown. Path timing: t=0 re-derived from the tlog start-gate rule, median 3 s later than fog's first-visit times, a few cells off by up to ~80 s (unexplained). Smoke-tested via HTTP and a JS parse only; no browser was available on the VM to confirm rendering. Doc note: the pilot 1 `runs/` dir holds a completed baseline-1001 (`080445`) that `level3_methodology.md` section 10 says does not exist (pilot 1 is superseded either way).

**Level 3 final paired evaluation (2026-11-09, corrected after the read-only audit):** 56 held-out runs (8 scenarios x 7 arms, T = 600 s, oracle off) plus 2 repeats (baseline and A0, scenario 2001), all clean (exit 0, no leftovers). **Provenance:** 26 runs are `git_dirty: false` (commit `0c2f6c6`, sequential execution) and 32 are `git_dirty: true` (30 main runs and the 2 repeats, 2-slot parallel execution; the dirty edits, committed in `661c72d`, are infrastructure only: ports, slots, tagged cleanup). The A0 repeat ran on an intermediate harness state that was never committed and cannot be recovered; the first baseline-repeat attempt failed after 1 s (cause unknown). An earlier version of this entry said every run was dirty; that was wrong. **Statistics:** the 8 scenarios use only 4 ignition cells, so the cell is the primary cluster (n = 4); the smallest possible p is 0.125, nothing is claimed significant, direction counts are the evidence. **Censoring:** 15 of 56 runs have no observed ignition-cell visit by 600 s (B 14 of 24, A 1 of 24, baseline 0); times are lower bounds. **Policy:** the planner is a reactive detection-triggered investigation policy (`max(conf)` evidence, investigate-first, 60 s cooldown); the probability map (`mapping/probability_map.py`, empty) was never built. Descriptive findings (A vs B, same direction in 4/4 cells): Model A detects sooner (177 vs 297 s lower bound), has far fewer false detection observations (0.3 vs 55.7 per run; distinct false cells 0.2 vs 6.4), reaches the ignition cell sooner and covers more; neither perception arm beat the patrol at reaching the ignition cell. `det_false` counts repeated observations, not incidents. Repeat runs are confounded with execution mode (both repeats +13-14 s) and are not a jitter estimate. Full account: `docs/experiments/level3_final.md`; generated tables, linkage and figure in `docs/experiments/level3_final/`. Status: prepared for independent Codex review, not declared validated; spend none. Next: Codex review of the corrected package (`docs/handoffs/level3_final_audit.md`). **Second-audit corrections (2026-11-09): a fire-clock confound was found and documented (not fixed, no rerun):** fog schedules fire steps by wall clock (`fog_tracker.py:311-314`), so the realized fire differs across arms in the 2-slot parallel scenarios (ignition-time ratio vs baseline 1.003-1.004 sequential, about 1.05-1.06 parallel, up to 1.09; burned sets differ in scenarios 2014 and 2005) although the initial fire configuration is identical; sensitivity by execution era shows A-vs-B directions agree in both eras but A-vs-baseline ignition-cell visit does not (3/3 sequential, 2/4 parallel scenarios). Also documented: radius-one truth matching, observed-only censoring bias, INVESTIGATE log lines as an approximate time proxy; baseline first detection is `NA` (not 600 s) in `seed_summary.csv`. New calculation tests on synthetic inputs (`tests/test_level3_analyze.py`, `tests/test_level3_report_calcs.py`); tests do not establish scientific validity. See `docs/experiments/level3_final.md` section 2a, `level3_methodology.md` section 16, `docs/handoffs/level3_final_audit.md`. **Third audit (Codex) corrections (2026-11-09):** capped-time censoring now read correctly (a capped difference is a bound only when censoring is confined to the later arm); unsupported fire-clock bias claims withdrawn (only measured associations kept: tick rate vs fire ratio, inconsistent within-scenario link to first detection); parallel-only subset covers 3 cells, not 4; figure captions corrected and made deterministic; 9 detection records stamped after 600 s in `l3-B2s0-2014` excluded (`det_true` 76 -> 67; `stats.json` unchanged).

**Baseline freeze and external-evaluation preparation (2026-11-09):** third-audit corrections committed locally (`b45ccc5` code and tests, `adc539f` regenerated outputs and docs; 299 tests pass; tables byte-identical and figures pixel-identical on regeneration from the raw runs). Levels 1-3 frozen as `baseline-L1-3-v1` (local tag, not pushed): `docs/baseline/baseline_v1.json` pins the code commit, 9 checkpoint hashes (all in `models/MODELS.md`), 15 configuration files, 56 Level 3 run configs, 209 result artifacts and 290 raw Level 3 files; check with `python3 eval/freeze_baseline.py --verify`. Level 3's Model B is the **Boreal v2** seeds, not `real_boreal_best_v1` (verified from every run manifest). External dataset desk audit and a draft cross-dataset protocol are in `docs/external_eval/` (separate from the baseline): FLAME has no smoke boxes/masks, so it is **not** used for box-level evaluation; FASDD_UAV is the only candidate with smoke boxes, but its license, formats, splits and sources are **unverified** (page unreadable by fetch, preprint withdrawn), so evaluation is gated on G1-G5 (license, label compatibility, Boreal overlap, sufficiency, RGB type). Nothing downloaded, no inference run, no spend. Next: owner reads the FASDD record and approves a download; then gates, then freeze `eval/datasets_external.yaml`, then eval-only runs. The 24-page report waits for the external evaluation and independent review.

**FASDD_UAV readiness assessment (2026-11-09):** official Science Data Bank record read (V9, CC BY-SA 4.0 verified, 25,097 UAV images, 17,222 smoke instances, YOLO/COCO/VOC/TDML, random 1/2:1/3:1/6 split; the published paper could not be fetched). Not verifiable without a login: `FASDD_UAV.zip` size and any checksum. Free disk is 17 GB against an estimated ~21 GB need, so **storage blocks the download**. Tooling added and tested (`eval/external_prep.py`, `tests/test_external_prep.py`: class remap, Boreal exact+dHash overlap using the saved scan, deterministic 100-image screening sample with 30 spot-checks); nothing downloaded, no inference, no spend, baseline `--verify` still passes. Details, gates and the exact next action: `docs/external_eval/readiness.md`.

**FASDD_UAV storage re-audit (2026-11-10):** `/` now has 98 GB free (144 GB volume), so the storage gate G0 passes (`docs/external_eval/readiness.md` section 2a; the 17 GB figure above is superseded). The download remains blocked: Science Data Bank returns `User Not Login` for the file list and the owner has no session in this environment. Nothing downloaded, no inference, no spend. Next: owner signs in and downloads `FASDD_UAV.zip` into `local_data/external_eval/raw/` (or provides a download URL/cookie-free link and the listed size), then gates G0b-G5.

**FASDD_UAV external-eval readiness (2026-11-10):** archive verified on the RunPod volume, official test split (4,181 images) extracted and checked, smoke-only label set built, G4 sufficiency passes (2,149 smoke-positive test images, 1,282 dHash clusters at radius 4). The v1 Boreal overlap flags were an artefact of an inverted dHash comparison in the stored scan; with identical hashing on both sides no genuine overlap was found, but G3 awaits the owner's visual check, and G2 (100-image screening, `local_data/external_eval/screening_v2/`) and G5 are open. `eval/external_prep.py` prefix mapping fixed with regression tests. Compute for these five short CPU pods is estimated at under $0.10 in total (billing not re-read; the volume storage charge is separate), no GPU, no training or evaluation. Details and evidence: `docs/external_eval/readiness.md` sections 2b-2f.

**FireMan-UAV-RGBT candidate and volume cleanup (2026-11-11):** read FireMan by HTTP range requests (no 33 GB download): CVAT XML boxes in 34 RGB videos (92.8% of 69,871 frames carry a smoke box), mixed resolutions, whole-plume boxes (median area 0.43, 74% edge-touching), official test split is effectively one video; overlap with Boreal (Ruokolahti) untested. Details: `docs/external_eval/dataset_audit.md`. Volume `sej40ez3ia` held 47.15 GB of 100 GB; after the owner approved it, the FASDD_UAV.zip (14.89 GB) was deleted by a guarded pod job (only after the sealed extracted test set, hashes and both screening folders were verified); 32.26 GB used now. The extracted FASDD test set, screening v1/v2, hashes and all Boreal, synthetic and run data are untouched. `boreal/` is 26.1 GB, of which `boreal_yolo` (13.07 GB) and the five per-event image folders (13.07 GB) look like duplicates (not verified, not touched). Two more CPU pods (cleanup 2 vCPU, $0.06/hr, about 1 minute; FireMan was read from the laptop at no cost); billing not re-read.

**External-validation strategy and FireMan checks (2026-11-11):** dataset roles fixed by the owner: Boreal and `synth_smoke_v1` = training and evaluation; FASDD_UAV and FireMan-UAV-RGBT (RGB) = independent external tests only; FLAME2-DT dropped. FireMan (34 RGB videos, 69,871 frames) hashed from 1,962 sampled frames: no MD5 match, no frame within 4 bits of Boreal, 22 within 8 (likely low-texture collisions, unverified); 30 scene clusters (the official split is not scene-disjoint); box labels are 96-99% interpolated; screening sample (100 frames) built, not reviewed. Proposal for the statistics (cluster bootstrap, paired B-A differences, pre-declared reading) and the blocker list: `docs/external_eval/fireman_protocol_proposal.md`. No inference, no pods, no spend for this step (FireMan read from the laptop).

**External-evaluation preparation (2026-11-11):** committed and tested `eval/fireman_prep.py` (CVAT XML to YOLO, deterministic sampling, pinned manifests) and `eval/external_stats.py` (cluster bootstrap and tests); pinned `eval/fireman_frames_primary_v1.json` (5 videos dated 2023-08-26 from thermal on-screen dates, 750 frames) and `..._secondary_v1.json` (29 unverified videos, 4,350 frames). Boreal's per-site dates were read (Evo 2022-05-24, Heinola 2022-06-28, Karkkila 2022-07-06, Ruokolahti 2022-08-15); FireMan lists 2022-07-06 and 2022-08-16, so unverified FireMan videos are excluded from the primary set. FireMan has no hash or pixel duplicate of Boreal or FASDD images. FASDD screening (screener = Claude, owner spot-check pending): 74 of 100 sampled images are fixed tower cameras (G5 fails for UAV-only) and about 10% of empty-label images show unlabeled smoke (not usable as negatives). Details: `docs/external_eval/readiness.md` 2g-2h and `fireman_protocol_proposal.md`. One CPU pod (about $0.035, EXITED), no inference.

**External protocol finalized as a proposal (2026-11-11):** `eval/datasets_external.yaml` (status `proposed_not_frozen`; same schema, settings and metric path as the 2x2; the six frozen checkpoints with their baseline hashes; FireMan primary = 5 videos dated 2023-08-26 (750 frames, descriptive only: 1 burn event, 2-5 scene clusters, about 23 effective frames), FireMan secondary = 29 provenance-unverified videos (4,350 frames, not in default cells), FASDD = 2,149 smoke-positive images as a mixed-platform set; cluster gate: interval and replication reading need at least 20 clusters, exact test at least 8). FASDD platforms separated by image size and visually validated (tower 718, UAV nadir 306, unknown-platform video 1,125); G5 (UAV-only) fails, G2 open. New tested code: `eval/external_materialize.py` (harness views, pins), `tests/test_external_pins.py` (all pins and manifests reproduce). Visual-review summary with the exact files for the owner: `docs/external_eval/visual_review_summary.md`. No inference, no pods this stage; CPU inference of the six checkpoints can run on the laptop.



## 2026-11-11 FireMan primary inference
Six frozen checkpoints run on the 750 pinned FireMan primary frames (local CPU, ~2 min/cell); baseline verifier passed after. Result and limits: `docs/external_eval/fireman_primary_v1_findings.md` (descriptive, one burn event). FASDD evaluation BLOCKED: test images exist only on the RunPod volume; needs `FASDD_UAV.zip` on the VM (`eval/external_scripts/extract_fasdd_positive.py`) or one approved CPU pod. Owner visual review of screening sheets: PENDING. No paid spend this step.

## 2026-11-11 FASDD smoke-positive primary inference and report (supersedes the 'FASDD BLOCKED' line above)
The FASDD test images were copied from the RunPod volume export (pod job `eval/runpod/fasdd_pod_job.py`), the six frozen checkpoints ran on the 2,149 pinned smoke-positive images (local CPU; baseline verifier passed before and after), and the report was generated. **B - A mAP50 = -0.184 pooled (cluster bootstrap -0.241, 95% [-0.270, -0.213], 680 clusters): Model A beats Model B, the reverse of Level 2 and FireMan.** Driven by the 720p ground/low-video group (-0.44); UAV nadir is a tie (-0.014); tower cameras about 0. G5 fails (mixed platforms), G2/G3 owner review still PENDING. Details: `docs/external_eval/fasdd_smoke_positive_v1_findings.md`. Fixed an out-of-memory bug in `eval/external_stats.py::sign_flip_test` (chunked Monte Carlo; this is what crashed the laptop). No paid spend this step; RunPod pod/volume state not re-read this session. Next: owner review of the visual-review sheets, Codex audit of the external results, then the 24-page report.
