# AI-Enabled Autonomous Wildfire Monitoring (ECE 496)

Does the training domain of a smoke detector (synthetic vs. real imagery) change how well it perceives wildfire smoke,
and does that difference carry through to how an autonomous UAV monitors a fire? The project trains YOLOv8s detectors on
synthetic smoke (Model A) and on the real Boreal dataset (Model B), compares them on both domains (Level 2) and flies
them in ArduPilot SITL against a no-detector patrol baseline (Level 3).

## Start here

| I want to... | Go to |
|---|---|
| **See the project working in 2 minutes** (browser replay of the 56 recorded experiments; Python only) | [QUICKSTART.md](QUICKSTART.md): `git clone --branch integration/yolo-perception-prep ...`, `cd ...`, `python3 demo.py`, then http://localhost:8081 |
| Install everything, run a **new** simulation, understand the dashboard, reproduce results, troubleshoot | [docs/PROFESSOR_GUIDE.md](docs/PROFESSOR_GUIDE.md) |
| Check what works on my machine | `python3 check_setup.py` |
| Run a new baseline / Model A / Model B mission | `python3 run_fresh.py --arm baseline\|A\|B\|all --dashboard` (needs ArduPilot SITL; see the guide) |
| Read the results | [docs/experiments/level3_final.md](docs/experiments/level3_final.md), [docs/report/final_report.md](docs/report/final_report.md) |
| See the current status and decisions | [docs/project-status.md](docs/project-status.md), [CLAUDE.md](CLAUDE.md), [models/MODELS.md](models/MODELS.md) |

## What a fresh clone contains

Code, frozen configs, reports, committed result tables and a single-file replay of the final runs. It does **not**
contain model weights, raw run data, datasets or ArduPilot; these are not public. The guide
(section 2) says which tier of the project each missing item blocks and how to obtain it.

## Honest scope

Level 3 uses a **single UAV** and a **reactive** detection-triggered planner. There is no Bayesian probability map
(`mapping/probability_map.py` is empty) and multi-UAV behaviour was not experimentally validated. The replay shows
recorded runs; fresh simulations are not bit-reproducible.
