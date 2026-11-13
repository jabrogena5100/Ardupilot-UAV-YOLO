# Professor / TA guide: set up, demonstrate and reproduce the wildfire-monitoring project

ECE 496, AI-Enabled Autonomous Wildfire Monitoring. This guide is for someone who has never opened this repository.
It was written by auditing the repository as it is (not older README text) and every command marked **tested** in
section 9 was run on the project machine (Ubuntu 24.04, Python 3.10.18) while writing it.

**If you only want to see the project, read [QUICKSTART.md](../QUICKSTART.md): three commands, no ArduPilot, no GPU.**

## 0. What you can do, and what each thing needs

| Tier | What you do | Needs | Time |
|---|---|---|---|
| **1. Replay** | Watch the 56 recorded final experiments (baseline, Model A, Model B) in a browser | Python 3.9+. Nothing else. Works from a fresh clone. | 2 min |
| **2. Recompute** | Re-derive the report's tables and figures from the saved run files | Tier 1 + the **data bundle** (raw runs, 2.8 GB, not in git) + numpy | minutes |
| **3. Fresh run** | Fly a **new** simulated mission in ArduPilot SITL for the baseline, Model A or Model B | Linux, ArduPilot SITL, tmux, 2 GB RAM; arms A/B also need the **model weights** (not in git) + PyTorch (CPU) | about 12 min per run |
| **4. Retrain / re-evaluate models** | Reproduce Level 2 training and the 2x2 matrix | A GPU (the project used RunPod), the Boreal and synthetic datasets (not in git) | hours, paid GPU |

**Replay vs new simulation.** The replay (tier 1) plays back files that already exist. It starts no simulator, loads no neural
network and writes nothing. A fresh run (tier 3) actually launches ArduPilot, the planner and the detector and produces new
files. The two are different programs with different ports (replay 8081, live dashboard 8080) and the page itself says
"replay" at the top.

### What a fresh clone does and does not contain

| Item | In a fresh `git clone`? |
|---|---|
| All code, configs, frozen thresholds/scenarios (`eval/level3_*.json`), reports, committed result tables (`docs/experiments/`) | Yes |
| Single-file replay of the 56 final runs (`docs/demo/level3_final_replay.html`, 2.6 MB) | **Yes** |
| Model weights (`*.pt`) | **No** (`.gitignore`). Not publicly hosted. See 2.3. |
| Raw Level 3 runs (`local_data/level3/final/`, 2.8 GB) and path cache | **No** (gitignored). See 2.4. |
| Datasets (Boreal, `synth_smoke_v1`) | **No**. Needed only for tier 4. |
| ArduPilot itself | **No**. Separate project; see 2.2. |

## 1. Requirements

* **Tier 1 (replay):** any OS with Python 3.9 or newer (tested: 3.10 on Ubuntu 24.04). A web browser. No pip packages are
  needed for the committed snapshot. No network after cloning. About 50 MB RAM.
* **Tiers 2 to 3:** Linux (Ubuntu 22.04/24.04 recommended; the runner uses `/proc` and `tmux`, so macOS/Windows are not
  supported for fresh runs; on those, use a Linux VM). **At least 2 GB RAM** (the final runs used a 1-2 vCPU, 2 GB VM, one run
  at a time; one run peaks near 1.4 GB), 10 GB free disk for ArduPilot plus about 4 GB for PyTorch (CPU). No GPU is used or needed.
* **Tier 4:** a CUDA GPU. Out of scope for a laptop; see section 6.4.

## 2. Install

### 2.1 Clone and check (everyone)

```bash
git clone --branch integration/yolo-perception-prep https://github.com/jabrogena5100/Ardupilot-UAV-YOLO.git   # or the URL/zip you were given
cd Ardupilot-UAV-YOLO
python3 check_setup.py
```

`check_setup.py` is read-only. It prints `[ ok ]`, `[warn]` or `[FAIL]` for each tier with the command that fixes
each warning, and ends with "tier 1 (replay) READY". Only the `[FAIL]` lines and the tier you want matter.

> **Branch note.** The final work (replay, guides, helper scripts) lives on the branch `integration/yolo-perception-prep`, **not on
> `main`** (verified: `main` has no `demo.py`). Always clone with `--branch integration/yolo-perception-prep`, or `git checkout` it after
> cloning. Verify with `ls demo.py docs/demo/level3_final_replay.html`. If you were given a zip instead of a URL, unzip it and start at `cd`.

For tier 1 you are finished: go to section 3.

### 2.2 Tier 3 only: Python environment and ArduPilot SITL

**(a) Python environment** (a virtual environment keeps this away from your system Python):

```bash
sudo apt update && sudo apt install -y python3-venv python3-dev tmux git build-essential
python3 -m venv ~/venvs/wildfire
source ~/venvs/wildfire/bin/activate              # run this in every new terminal
pip install --upgrade pip
pip install -r requirements-sim.txt                # baseline arm: PyYAML, numpy, OpenCV, Pillow, pymavlink, MAVProxy
pip install -r requirements-yolo.txt               # only for Model A / Model B: CPU PyTorch + ultralytics (about 4 GB)
```

**(b) ArduPilot SITL.** This is the upstream flight-software simulator; this repository only drives it. Build it once
(about 20 to 40 minutes on a small VM):

```bash
git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git ~/src/ardupilot
cd ~/src/ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y     # then log out and back in (or: . ~/.profile)
echo 'UH=21.296868,-157.816036,100,270' >> Tools/autotest/locations.txt   # the University of Hawaii start location the project uses
./waf configure --board sitl && ./waf copter                # builds build/sitl/bin/arducopter
echo 'export PATH=$PATH:$HOME/src/ardupilot/Tools/autotest' >> ~/.bashrc && source ~/.bashrc
which sim_vehicle.py                                        # must print a path
```

* The `UH=` line matters: `launch_sitl_swarm.py` starts the vehicle with `-L UH`, and stock ArduPilot has no such location.
* The project machine used ArduPilot `4.6.0-beta1-4159-g4b436becb7` (`git checkout 4b436becb7` reproduces it). A newer
  checkout probably works but is **untested**.
* `sim_vehicle.py` must find `mavproxy.py`, which the `pip install` above puts in the virtual environment's `bin/`. Keep the
  virtual environment activated when you run an experiment.

Re-run `python3 check_setup.py`: the "Tier 3" block should now be all `[ ok ]` (the arm-weight lines depend on 2.3).

### 2.3 Model weights (Models A and B): **not in git, not publicly hosted**

`*.pt` files are gitignored and no public download exists. **This is a limitation:** a fresh clone cannot run Model A or
Model B, and the baseline arm and the replay do not need them. To run arms A/B you must obtain the files from the project
team (the primary copy is on the team's RunPod network volume, a verified backup on the team's laptop; see
`models/MODELS.md`). Ask the team for the weights bundle, which is the six files below (about 22.5 MB each; Level 3's "Model B"
is the **Boreal v2** retraining, not `real_boreal_best_v1.pt`).

Place each file at exactly this path **relative to the repository root** (these are the paths `eval/level3_pairs.py` reads):

| Arm | Path | SHA-256 (first 16 hex) |
|---|---|---|
| A, seed 0 | `local_data/backups/synth_smoke_a_v1/synth_smoke_a_v1/synth_smoke_best_v1.pt` | `1da10c4814a905d9` |
| A, seed 1 | `local_data/backups/level2_v2_seeds/synth_smoke_a_v1_seed1/synth_smoke_best_v1_seed1.pt` | `31c4528e37c54f77` |
| A, seed 2 | `local_data/backups/level2_v2_seeds/synth_smoke_a_v1_seed2/synth_smoke_best_v1_seed2.pt` | `3d5a31689ab21243` |
| B (v2), seed 0 | `local_data/backups/boreal_v2/real_boreal_v2_best_seed0.pt` | `090c0f4cbe2718d2` |
| B (v2), seed 1 | `local_data/backups/boreal_v2/real_boreal_v2_best_seed1.pt` | `5a9616cefddd3bca` |
| B (v2), seed 2 | `local_data/backups/boreal_v2/real_boreal_v2_best_seed2.pt` | `c4d443a8af919c6b` |

```bash
sha256sum local_data/backups/boreal_v2/real_boreal_v2_best_seed0.pt     # compare with the table; full hashes: models/MODELS.md, docs/baseline/baseline_v1.json
python3 check_setup.py | grep weights                                    # every line should say [ ok ]
```

The detector itself refuses weights whose hash it does not recognise, so a wrong file fails loudly rather than silently.
You need only seed 0 of each family for a single comparison.

### 2.4 Data bundle for tier 2 (raw runs): not in git

`local_data/level3/final/` (56 run directories, 2.8 GB) is gitignored. Ask the team for it and unpack it so that
`local_data/level3/final/results/` contains 56 directories named `l3-<arm>-<scenario>`. Everything in tiers 1 and 3 works
without it. If you have it, `demo.py` automatically serves from it (the "full" source) instead of the snapshot.

### 2.5 Confirm the installation

```bash
python3 check_setup.py                 # reads your machine; tier 1 must say READY
python3 demo.py --no-browser &         # starts the replay (Ctrl+C or `kill %1` to stop)
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8081/    # expect 200
kill %1
python3 -m unittest discover -s tests -t .     # optional, 385 tests, about 5 min on a 1.9 GB VM (renders images)
```

## 3. Quick-start demonstration (replay)

```bash
python3 demo.py
```

Then open **http://localhost:8081** (the script tries to open it for you). The terminal prints
`REPLAY of recorded Level 3 runs (not a live simulation)`. Leave the terminal open; the page is served from it.

* **What appears:** a 10 x 10 grid map per arm (Baseline, Model A, Model B) side by side on one clock, a time slider,
  Play/Restart, a scenario drop-down, layer checkboxes and, below, a table of recorded results.
* **Start / use:** press **Play** (or Space), set speed 30x, drag the slider; pick another scenario (try 2001 or 2004); press
  **UAV's view (hide truth)** to hide the fire; the **Jump to** buttons go to each arm's first true detection.
* **Stop:** press **Ctrl+C** in the terminal (tested: it exits cleanly, code 0, and leaves no process behind).
* **Reset:** press **Restart** on the page, or reload the page. A replay has no state, nothing to clean up.
* **Options:** `--port 9000` (another port), `--host 0.0.0.0` (reachable from outside a VM), `--no-browser`,
  `--source snapshot` or `--source full` (see 2.4). The data source it chose is printed.

**Blank or broken page?**

1. Reload with Ctrl+Shift+R. Open `http://127.0.0.1:8081` instead of `localhost` (some systems resolve `localhost` to IPv6 only).
2. The terminal must still be running `demo.py`; if it printed `port 8081 is already in use`, run `python3 demo.py --port 8082`.
3. Run `python3 check_setup.py`; tier 1 must be READY. If `docs/demo/level3_final_replay.html` is missing, your checkout is on the wrong branch or incomplete (`git status`).
4. In a VM, see 7.4.
5. A blank map with a populated table usually means a very old browser; use a current Firefox or Chrome.

## 4. Understanding the dashboards

There are two pages. The **replay** (port 8081) is for looking at finished runs. The **live dashboard** (port 8080,
`dashboard/server.py`) is the same idea for a run in progress. Layers:

| What you see | Colour / shape (replay) | What it is | Who knows it |
|---|---|---|---|
| UAV position and trajectory | white dot with a trail | Real MAVLink position samples (1 Hz) from the SITL log, no interpolation | UAV and ground station |
| Visited / unvisited cells | blue = visited, dark = not yet | Cells the UAV's footprint has covered | UAV and ground station |
| Ground-truth fire | orange = burning, grey = burnt out, yellow outline = ignition cell | The fire simulator's true state (`wildfire_sim.py`, a 2-D cellular automaton) | **Ground station / evaluation only. The UAV never sees it.** |
| Smoke detections | green circle = scored true, red circle = false alarm | One YOLO detection mapped to a grid cell, then scored against truth at the moment it arrived | The UAV makes it; **the colour is judged by the ground station** |
| AI-triggered investigation targets | dashed boxes: cyan = fire was near, magenta = no fire there | Cells the planner sent the UAV to check after a detection | The UAV's decision; the cyan/magenta colouring is judged in hindsight by the ground station |
| Metrics table | numbers under the maps | First true detection, first arrival at the ignition cell (capped at 600 s, shown as "at least 600 s"), coverage, true/false detection records, investigation time | Computed offline from the saved files |

**What is visible only to the ground station vs available to the UAV.** The UAV (`swarm_agent.py`) gets: its own telemetry,
its own camera frames, and what its detector reports. It is never given the fire map. The fire simulator runs only inside
the "fog" process (`fog_tracker.py`), which plays the role of the ground station: it holds the true fire, scores every
detection when it arrives, writes the CSV results and feeds both dashboards. The camera image the UAV sees is rendered
by a service inside fog from the burning cells, but the service returns only a JPEG and exposes no fire state. A config
flag `fire_model.oracle_hazards` would give the UAV perfect fire knowledge; it is **off in every recorded run** (a test checks
this) and fog prints a loud banner if it is turned on. "UAV's view (hide truth)" in the replay shows roughly what the UAV could know.

**Honest limits of what the page shows**

* **Single UAV.** All 56 recorded runs fly one UAV. Multi-UAV code exists, but multi-UAV behaviour was **not** experimentally validated.
* **No Bayesian probability map.** `mapping/probability_map.py` is an empty file. Planning is a reactive rule: a detection
  above the model's calibrated confidence threshold triggers an investigation (`max(confidence)` evidence, 20 s lifetime, 60 s
  cooldown). The live dashboard's "P(fire)" layer is that placeholder score, **not** a calibrated or Bayesian probability.
* **Not a real fire or a real camera.** Smoke is a flat billboard on a 2.5-D rendered ground, not a volume. The fire is a
  cellular automaton on a 10 x 10 grid (1 mile square).
* **Reporting window = 0 to 600 s.** Playback, charts and the table use the window of the published analysis. One run (`l3-B2s0-2014`) also has
  9 detection records stamped 600.12 to 600.16 s (all scored true); they are kept in the data but not replayed, and the table shows the published window
  totals (67 true / 28 false; the full log has 76 / 28). The page says so under the table and in "Data notes and caveats".
* **In "UAV's view (hide truth)"** detections and investigation boxes turn neutral, because telling a true detection from a false one needs the hidden fire.
* **Investigation boxes are untimed** (the agent log carries no timestamps), so they summarise the whole run and are not placed on the timeline.
* The fire clock runs on wall time in fog, so the realised fire differs slightly between arms of one scenario
  (`docs/experiments/level3_final.md` section 2a). 8 scenarios on 4 ignition cells: results are descriptive; no significance claims.

More detail: [docs/demo.md](demo.md).

## 5. Run a new experiment (tier 3)

**Prerequisites:** 2.2 done, venv active, `python3 check_setup.py` Tier 3 block all `[ ok ]`; for A/B also 2.3. Stop other
simulations first (the runner refuses to start if it finds SITL/fog processes).

### 5.1 One command per arm

```bash
python3 run_fresh.py --arm baseline --dashboard      # no neural network: the planner just patrols
python3 run_fresh.py --arm A --dashboard             # Model A: synthetic-trained YOLOv8s (seed 0)
python3 run_fresh.py --arm B --dashboard             # Model B: real (Boreal v2) YOLOv8s (seed 0)
python3 run_fresh.py --arm all --dashboard           # the three, back to back, about 40 minutes
```

* Add `--dry-run` first: it lists what would run and every missing prerequisite, and starts nothing.
* `--dashboard` also starts the live view at **http://localhost:8080**. It shows "waiting for fog" until the UAV is
  airborne (about 1 to 2 minutes in), then the grid fills with blue visited cells, the white UAV marker and (as ground truth) the fire.
* All three use **scenario 2001** by default (fire seed 2001, ignition cell (3,2), 600 s, one UAV at 120 m). Choose another with
  `--scenario` (held-out: 2001, 2002, 2004, 2005, 2006, 2012, 2014, 2015; calibration: 1001, 1002, 1003, 1005).
  `--seed 0|1|2` picks which of the three trained copies of A or B to use.
* **Fairness.** The wrapper calls the project's own planner `eval/level3_pairs.py`, which builds one config per arm from the
  frozen `eval/level3_fire.json`, `eval/level3_scenarios.json` and `eval/level3_thresholds.json`, and **refuses** to write a
  set in which the arms differ in anything except the perception keys (enabled, weights, hash, tag, confidence threshold).
  Same fire, same ground texture, same flight altitude/speed, same planner, same agent seed. Each detector uses the
  confidence threshold that was calibrated for it on separate calibration scenarios, not a hand-picked value.
* Each run takes about 12 minutes of wall clock (45 s SITL boot, start gate, 600 s mission). With `--arm all` they run one after another.
* **Stop:** Ctrl+C in the terminal. The runner stops fog, the agent, MAVProxy, the SITL process and its tmux session. Check with
  `pgrep -af "arducopter|mavproxy|fog_tracker|swarm_agent"` (nothing should print).

### 5.2 Where the outputs are

Everything goes to a new directory (default `local_data/fresh_runs/<timestamp>-<scenario>/`; change with `--out DIR`); the
recorded runs are never touched.

```
<out>/plan/l3-<arm>-<scenario>.yaml          the exact config that was run (and matrix.json with hashes of the frozen inputs)
<out>/results/l3-<arm>-<scenario>/
    experiment_end_*.txt   headline result: reason, time_s, visited_frac, det_true, det_false, precision, first_detection_t, detection_lag_s
    detections_*.csv       every detection, scored true/false at the moment it arrived (the Level 3 result file)
    coverage_t_*.csv, fog_cells_*.csv, fire_cells_*.csv, fire_t_*.csv     coverage and ground-truth time series
    agent_A.log            the UAV's decisions (INVESTIGATE lines)
    manifest.json          weights path/size/SHA-256, camera settings, oracle flag, python and package versions, git commit
    run_meta.json          start/end time, exit code, leftover-process check, git commit and whether the tree was dirty
    fog.log, sitl.log, sitl_work/mav.tlog     raw logs
<out>/plan/runs/<timestamp>-l3-<arm>-<scenario>/   fog's own copy of the result files (the runner copies them into results/)
```

Compare with the recorded run of the same arm and scenario: `docs/experiments/level3_final/per_run.csv`, row `l3-<arm>-<scenario>`.
(Recorded Model B means a Boreal v2 seed: arm id `B2s0`.)

To look at a fresh run in the replay viewer, the replay build tool (`dashboard/replay_build.py`) reads the final runs only; viewing fresh
runs there is **not supported**. Use the live dashboard during the run and the CSVs afterwards.

### 5.3 Random seeds and reproducibility

* **What is seeded:** the fire (`fire_seed`, per scenario), the rendered environment (`scenario_seed`), the UAV's own random
  choices (`scenario_seed * 100 + k + 1`), `PYTHONHASHSEED=0`, and the detectors use fixed weights. They are
  written into the config and `manifest.json`. They are the same for every arm of a scenario, so arms differ only by perception.
* **Seeds 0/1/2 of Model A and B** are three independently trained networks (different training seeds). They measure training
  variance. The report quotes mean +/- SD over those 3, never one.
* **What is not repeatable:** ArduPilot SITL and the fire clock run against the wall clock, MAVLink and camera calls take
  variable time, and CPU load changes it. The same config therefore gives the same **scenario** but **not** identical trajectories,
  detection counts or times. In this project two repeats of recorded runs differed by about 13 to 14 s in a timing measure, and that
  was confounded with execution mode; see `docs/experiments/level3_final.md` section 2a. **Expect the same qualitative behaviour
  and numbers of similar size, not equality**, and never compare a single fresh run against a recorded mean as if it were a test.
* A detector arm has two sources of variation (network seed, run timing); the baseline has one.

## 6. Reproducing the published tables and figures

Be explicit about the four things people mean by "reproduce". All commands run from the repository root.

### 6.1 Replay an existing run (CPU only, fresh clone)

`python3 demo.py` (section 3). Reproduces nothing new; it shows the recorded runs. Needs nothing beyond Python.

### 6.2 Read the committed tables and figures (CPU only, fresh clone)

The report's numbers are committed as text: `docs/experiments/level3_final/per_run.csv` (56 rows),
`seed_summary.md`, `cluster_stats.md`, `censoring.md`, `linkage.md`, `provenance.md`, `fire_clock.md`; figures
`fig_level3_scenarios.png`, `fig_trajectories.png` in the same directory; Level 2 in `docs/experiments/boreal_v2/results/`;
the report draft is `docs/report/final_report.md`. Level 3 narrative and the reading rule: `docs/experiments/level3_final.md`.
Models and their hashes: `models/MODELS.md`. A frozen snapshot of all hashes: `docs/baseline/baseline_v1.json`.

### 6.3 Recompute metrics from the recorded runs (CPU only, needs the data bundle, 2.4)

```bash
OUT=/tmp/l3_recomputed; mkdir -p $OUT
python3 eval/level3_analyze.py local_data/level3/final/results/* --out $OUT     # per-run metrics (per_run.csv), stats.json, figure; about 7 s
mkdir -p /tmp/l3_tables && cp $OUT/per_run.csv /tmp/l3_tables/                    # the table script reads per_run.csv from its --out directory
python3 eval/level3_report_tables.py --out /tmp/l3_tables                         # seed/cluster/censoring/linkage/provenance tables + trajectory figure; about 8 s
diff $OUT/per_run.csv docs/experiments/level3_final/per_run.csv && echo "per_run.csv identical"
diff -rq /tmp/l3_tables docs/experiments/level3_final                              # see the note below
python3 eval/freeze_baseline.py --verify                                           # re-hash code, configs, checkpoints, results (needs the weights too)
```

On the project machine this reproduced `per_run.csv` and every generated table **byte for byte** and both figures **pixel for pixel**
(PNG file bytes differ because of metadata). `diff -rq` will list `matrix.json`, `repeats.json`, `stats.json` and `fig_level3_scenarios.png`
as "Only in docs/..." (those come from `level3_analyze.py`, not the table script) and may list the two PNGs as "differ" (compare pixels, not bytes).

Use `--out` to a scratch directory so the committed tables are not overwritten. Without the data bundle these commands cannot run
(the raw runs are not in git); use 6.2 instead. `freeze_baseline.py --verify` was **not** run for this guide.

### 6.4 Run a new simulation (CPU, needs ArduPilot; weights for A/B)

Section 5. A fresh run is a new sample, not a recomputation: do not expect equality (5.3). To reproduce the **full** 56-run
evaluation, `python3 eval/level3_pairs.py plan --split heldout --final-evaluation --out DIR` then
`python3 eval/level3_queue.py --plan DIR ...` (see the docstring in `eval/level3_queue.py`) is the mechanism that was used; it takes about
11 hours sequentially (56 runs x about 12 minutes) and was **not** re-run for this guide.

### 6.5 Retrain models or redo the Level 2 matrix (needs a GPU and datasets)

* Training used a rented RTX 4090 on RunPod: Model A about 17 minutes, Model B about 36 minutes per seed. Recipes and
  environments: `models/MODELS.md`, `docs/synthetic/synth_smoke_v1_spec.md` section 5, `docs/experiments/level2_v2/training/`.
* The datasets are not in git: the real Boreal Forest smoke dataset (provenance in `docs/experiments/real_boreal_v1.md`; the project holds its own copy and I did not verify a public download link) and the generated
  `synth_smoke_v1` (regenerable with `synthgen/` and recorded in `docs/datasets/synth_smoke_v1.md`; the recorded pins and
  manifests, not the images, are in git).
* The evaluation harness `eval/run_matrix.py` + `eval/datasets.yaml` computes the 2x2 matrix from `(checkpoint, test set)`. It can run on
  CPU, slowly, **if** you have the weights and the test sets. **Not tested for this guide**; neither training nor the 2x2 re-evaluation
  was re-run.

| Result | CPU-only laptop | Needs extra |
|---|---|---|
| Replay of Level 3 | yes | none |
| Read tables, figures, report | yes | none |
| Recompute Level 3 tables from raw runs | yes | data bundle |
| Fresh Level 3 baseline run | yes (Linux) | ArduPilot SITL |
| Fresh Level 3 run with Model A/B | yes (Linux, slow, 12 min/run) | + weights + CPU PyTorch |
| Level 2 matrix re-evaluation | possible, slow | weights + test sets |
| Model training | no | GPU + datasets (about $1 per seed on RunPod) |

## 7. Troubleshooting

### 7.1 Python environment / missing dependencies
`ModuleNotFoundError: yaml|cv2|pymavlink|numpy` means the virtual environment is not active or not installed:
`source ~/venvs/wildfire/bin/activate && pip install -r requirements-sim.txt`. `python3 check_setup.py` names the missing module.
Python older than 3.9 is not supported (`python3 --version`). `ultralytics`/`torch` missing: `pip install -r requirements-yolo.txt`
(the `+cpu` wheels come from the extra index inside that file; no GPU is needed). The replay needs none of this.

### 7.2 SITL does not launch
* `sim_vehicle.py: command not found` or `run_fresh.py` reports it: add `~/src/ardupilot/Tools/autotest` to `PATH` (2.2).
* `Unknown location UH`: you missed the `UH=` line in `Tools/autotest/locations.txt`.
* `arducopter` not found / `--no-rebuild` failure: build it, `cd ~/src/ardupilot && ./waf configure --board sitl && ./waf copter`.
* `tmux: command not found`: `sudo apt install tmux`.
* `refusing to start ... processes already running`: a previous run is still alive. `tmux kill-server; pkill -f arducopter; pkill -f mavproxy`.
* Start gate never opens ("start gate" in the error): the vehicle was slow to get a GPS fix on a slow VM. The runner retries once
  by itself; otherwise close other programs and retry. Look at `<out>/results/<run>/sitl.log`.

### 7.3 MAVLink connection failures
The agent connects to `udp:127.0.0.1:14555`, which MAVProxy (started inside SITL's tmux session) forwards. If `agent_A.log` repeats
"waiting for heartbeat": SITL did not boot (see `sitl.log`), another program holds UDP 14555 (`ss -ulpn | grep 14555`), or a
firewall blocks loopback (rare). Do **not** start QGroundControl on the same ports during a run.

### 7.4 Dashboard not reachable; VirtualBox networking
* On the same machine open `http://127.0.0.1:8081` (replay) or `:8080` (live).
* **Inside a VirtualBox VM, browser on the host:** start the server on all interfaces and forward the port.
  1. In the VM: `python3 demo.py --host 0.0.0.0 --no-browser` (live view: `python3 run_fresh.py ... --dashboard --dashboard-host 0.0.0.0`).
  2. In VirtualBox: VM Settings, Network, Adapter 1 (NAT), Advanced, **Port Forwarding**, add a rule: Protocol TCP, Host IP 127.0.0.1,
     Host Port 8081, Guest IP empty, Guest Port 8081 (and again for 8080 if you use the live view).
  3. On the host open `http://127.0.0.1:8081`.
  Alternative: Bridged Adapter, then browse to `http://<VM-IP>:8081` (`hostname -I` in the VM).
* If the VM has a desktop, simply open a browser inside it; that needs no forwarding.
* "Address already in use": pick another port with `--port`. "Connection refused": the server is not running or is bound to 127.0.0.1 while you are outside the VM.
* The dashboards have **no authentication**; use `--host 0.0.0.0` on a trusted network only.

### 7.5 Missing model weights
`weights for ...: missing` (from `check_setup.py` or `run_fresh.py`): see 2.3. The path must match exactly. If a detector arm
runs but shows **zero detections**, check `fog.log`/`agent_A.log` for a loud warning that perception fell back to `NullPerception`
before blaming the detector (a missing or hash-mismatched weights file fails soft in some configurations).
Baseline and replay never need weights.

### 7.6 Missing datasets or saved artifacts
* `--source full needs the raw runs ...`: you do not have the data bundle; use `--source snapshot` (the default on a fresh clone).
* `docs/demo/level3_final_replay.html is missing`: wrong branch or incomplete clone (`git status`, `git checkout integration/yolo-perception-prep`).
* `eval/level3_analyze.py` finds no runs: the bundle is not unpacked at `local_data/level3/final/results/`.
* Datasets are only needed for tier 4 and are never needed for tiers 1 to 3.

### 7.7 Empty dashboard or no smoke detections
* Replay: pick a scenario and press Play; the baseline has **no** detections by design (it has no detector, shown as n/a).
* Live dashboard: "waiting for fog" for the first 1 to 2 minutes is normal. It stays empty if the dashboard and the run use different
  gossip ports (default 5005 for both) or if the run was started by hand without the fog process.
* Few or no detections in a detector arm can be genuine: the UAV may not have flown near the fire yet (Model B reached the ignition cell
  in only about 10 of 24 recorded runs within 600 s). Check `det_total` in `experiment_end_*.txt` and the warning mentioned in 7.5.

### 7.8 Insufficient RAM or slow execution
One fresh run peaks near 1.4 GB, so a 2 GB VM suffices for **one** run at a time; do not run two, and close browsers inside the VM.
Add swap if the machine freezes (`free -m`). A detector run is slower than the baseline (about 0.3 s YOLO + 0.5 s rendering per frame on one core)
and is paced by wall clock, so a slower CPU gives the UAV fewer camera frames per mission; that changes results, so use comparable
hardware for all arms. The replay needs about 50 MB.

## 8. Where to go next

* `docs/project-status.md`: current state and history. `models/MODELS.md`: checkpoints and hashes.
* `docs/experiments/level3_final.md`: the Level 3 study, with its reading rule and limits.
* `docs/report/final_report.md`: the report draft. `CLAUDE.md`: the project rules (research integrity, frozen artifacts).
* Frozen items you should not edit: test splits, `synth_smoke_v1`, checkpoints, `eval/level3_*.json`, the recorded runs.

### Manual browser checklist (2 minutes; use this if you want to confirm the page on your own machine)

1. `python3 demo.py`; the page heading is "Autonomous Wildfire Monitoring — Experiment Replay" and a yellow "REPLAY OF SAVED RUNS - NOT LIVE" badge is visible.
2. Three map panels (Baseline, Model A, Model B) with a dark grid. Press F12, Console tab: there should be no red errors.
3. Press **Play** at 30x: the clock `t = ... s / 600` counts up, blue cells appear, a white dot with a trail moves, orange fire cells appear.
4. Drag the slider to the end: green/red circles with numbers appear on the Model A and B maps (none on the baseline); the table below lists all three arms.
5. Click **UAV's view (hide truth)**: the orange/grey cells disappear and "fire ground truth" is unticked. Click **Show everything**: they return.
6. Choose scenario 2004 from the drop-down; click a **Jump to** button (the clock jumps). Click **Single run**, then **Restart** (clock back to 0).
7. Ctrl+C in the terminal: the server stops.

## 8b. Four-UAV demonstration (optional; separate from the research)

**What it is.** A live demonstration that the project's multi-UAV architecture works: four ArduPilot SITL copters fly a grid-coverage mission together over a
simulated wildfire. **It is not an experiment and produces no result.** All research results (the 56 recorded runs and the report) are single-UAV and are unchanged.
It is a *new* simulation (like section 5), not a replay; the replay on port 8081 is a different program and is unaffected.

```bash
source ~/venvs/wildfire/bin/activate          # the environment from 2.2: needs ArduPilot SITL, tmux and requirements-sim.txt (this puts mavproxy.py on PATH); no weights, no GPU
python3 demo_multi.py --drones 4              # then open  http://localhost:8080/multi.html
```

* **Time:** about 2 minutes of startup (four SITL boots), then the mission (`--duration`, default 900 simulated seconds, about 17 minutes of wall clock). The page shows
  "Starting up" until the UAVs connect.
* **Stop:** Ctrl+C (or `kill` the launcher). It stops fog, the four agents, four SITL/MAVProxy processes, the tmux session and the dashboard; tested: exit code 0 in about 6 s with no leftover process.
  When the mission ends the SITL processes stop and the dashboard keeps showing the final state until you press Ctrl+C (`--exit-when-done` quits instead).
* **Options:** `--port`, `--host 0.0.0.0` (VM, see 7.4), `--no-browser`, `--dry-run` (prerequisite check), `--ignite i,j`, `--duration`. Output files go to a new `local_data/demo_multi/<timestamp>/`.
* **Resources (measured on the 1.9 GB / 4 vCPU project VM):** about 1.25 GB in use while running, 40 MB of swap, so it fits, with little headroom; close other programs. On a 1-vCPU VM it is untested.
  Four SITL instances is the supported maximum here; if it is too heavy for your machine the lighter alternative is the single-UAV runs of section 5 or the replay.

**What you see:** a red-amber banner "MULTI-UAV ARCHITECTURE DEMONSTRATION ...", the 10 x 10 map with four numbered, coloured UAVs (cyan 1, yellow 2, violet 3, green 4) and their trajectories,
each UAV's assigned quadrant (dashed border in its colour), the cell each is flying to (solid square), visited cells (blue), the simulated fire spreading (orange burning, grey burnt),
the elapsed simulation time and coverage percentage, a legend with each UAV's live telemetry, and a "what is real / what is illustrative" panel.

| Element | Source |
|---|---|
| UAV positions, altitude, heading, speed, mode | **Real ArduPilot SITL telemetry** (MAVLink), one SITL instance per UAV (MAVLink system ids 1 to 4) |
| Trajectories | The positions the dashboard actually received (kept by `dashboard/server.py`, `/api/trails`); nothing interpolated or scripted |
| Quadrant assignment, target choice, coverage map, run clock | The project's own code: `mission/partitions.py` (quadrants), `swarm_agent.py` (`grid_frontier`), `fog_tracker.py` |
| Fire and its spread | **Simulated** (the 2-D cellular automaton in `wildfire_sim.py`), owned by the ground station; the UAVs never see it and no detector runs, so it does not influence their flight |
| Smoke detections | **None.** No detector runs in this demo, so the panel says so; a detection would appear only if the system produced one |
| Fire parameters, duration, altitude 60 m, speed 15 m/s | Illustrative display settings (the fire rate equals the Level 3 setting, copied, not edited) |

**Not validated:** no performance claim is made; multi-UAV behaviour was not experimentally validated; no probability map; all four UAVs start at the same SITL home point and nothing models collisions.
The trails show real flight, including turns, transit across other regions and overshoot; the quadrant is the set of cells a UAV may *choose as targets*, not a fence.
"Region cells visited" counts cells visited by any UAV. The coverage percentage is the fog tracker's shared map.

## 9. Verification record

Tested on the project machine (Ubuntu 24.04, Python 3.10.18 venv, 1.9 GB RAM, 4 vCPU), 2026-11-13:

* **Tested:** `git clone` of the committed repository into an empty directory, then `python3 -I demo.py --no-browser --port N`: the
  snapshot is served (HTTP 200, 2.6 MB page, correct title), source printed as `snapshot`; `check_setup.py` from the clone ends "tier 1 READY".
* **Tested:** `python3 demo.py` in the working tree (source `full`): HTTP 200; Ctrl+C (SIGINT to the process group) exits with code 0 and no
  `replay_server` process remains.
* **Tested:** `run_fresh.py` error handling. In a clean clone with a stripped `PATH` it reports, without starting anything, each missing item with its fix
  (`sim_vehicle.py`, `mavproxy.py`, Python modules, `ultralytics`/`torch`, each missing weights file; it also checks the SITL build, the `UH` location,
  tmux, leftover SITL processes and busy ports) and exits 1 (dry run) or 2; an unknown scenario lists the valid ones; a run that fails to launch prints the tail of
  `sitl.log` and `fog.log` plus likely causes and exits 1 (that last path was checked with a simulated failure, not a real SITL crash).
* **Tested (browser):** the replay snapshot from a clean clone was driven in headless Chromium (Playwright): no console errors, page errors or failed
  requests, no request leaves localhost; the scenario menu lists all 8 scenarios and each of the three seeds; Play at 30x advanced the clock, the slider reached 600 s, the three
  maps and the four charts changed and were non-blank, "UAV's view (hide truth)" unticks the fire layer, "Jump to" moved the clock, Single run and Restart worked.
  The embedded data was checked against the sources: 56 runs, every run's recorded metrics equal `per_run.csv`, every detections list equals the run's `detections_*.csv`,
  and re-exporting from the raw runs gives a file **byte-identical** to the committed snapshot.
* **Tested:** a **fresh baseline run**, `python3 run_fresh.py --arm baseline --dashboard --python ~/venvs/yolo-cpu/bin/python`, scenario 2001: it
  planned the config, booted SITL, flew 600 s, wrote the files listed in 5.2, exited with code 0 after 705 s, and left no SITL, MAVProxy,
  fog, agent, dashboard or tmux process behind. The live dashboard answered at http://localhost:8080 during the run (HTTP 200; `/api/state` showed
  `fog_connected: true`, the UAV, visited cells and the fire). The generated config's SHA-256 equals the recorded run's, and the end-of-run
  values equal the recorded baseline-2001 run (`visited_frac` 0.22, `burnt_cnt` 3, 0 detections). One sample only: this shows the setup works, it is not a
  statement that fresh runs always match.
* **Tested:** a **fresh Model A run** (seed 0, scenario 2001, same command with `--arm A`): exit 0, 12 min, no leftover processes,
  `manifest.json` records `oracle_hazards: false` and `model_tag synth_smoke_v1_seed0`. It was **similar to, not equal to**, the recorded run: first true detection
  106 s vs 101 s recorded, 17 true / 4 false detection records vs 14 / 2, coverage 0.17 vs 0.21. This is the timing nondeterminism described in 5.3 and is why the guide
  does not promise equality. (Model B uses the same code path; it was not separately run.)
* **Tested:** recompute (6.3): `level3_analyze.py` on the 56 recorded runs and `level3_report_tables.py` reproduced the committed tables byte for byte and the figures pixel for pixel; the
  committed research files were unchanged afterwards (`git status` on `docs/experiments`, `models`, `eval` shows nothing modified).
* **Not tested:** rendering in a real browser by an automated check (HTTP status and file contents were checked; no
  browser was available to the automated checks, so please report anything that looks wrong on screen); a fresh Model B run (same code path as the tested Model A run); installing ArduPilot
  on a clean machine (the steps in 2.2 are the standard upstream procedure plus the project's `UH` location and were not re-run from scratch);
  Windows/macOS; VirtualBox port forwarding on a clean VM; the full 56-run reproduction; training and Level 2 re-evaluation.
* **Known limitations:** weights, raw runs and datasets are not public (2.3, 2.4); fresh simulations are not bit-reproducible (5.3);
  single UAV only; no probability map; the replay viewer cannot show fresh runs; the live dashboard's "P(fire)" layer is a placeholder score;
  `launch_experiment.py` and the older `exp_swarm_*.yaml` swarm configs are from earlier phases and are **not** the supported path.
