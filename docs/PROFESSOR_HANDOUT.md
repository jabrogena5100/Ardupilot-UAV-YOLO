# Handout: running the wildfire-monitoring demonstrations yourself

ECE 496, AI-Enabled Autonomous Wildfire Monitoring. Three ways to see the project, from easiest to most involved.
Everything below was tested; "Not tested" items are marked. Full details: [PROFESSOR_GUIDE.md](PROFESSOR_GUIDE.md).

## Option 1 (recommended, 2 minutes): replay of the 56 recorded experiments

Needs only **Python 3.9 or newer** and a web browser. No GPU, no ArduPilot, no internet after cloning.

```bash
git clone --branch integration/yolo-perception-prep https://github.com/jabrogena5100/Ardupilot-UAV-YOLO.git
cd Ardupilot-UAV-YOLO
python3 demo.py
```

Open **http://localhost:8081** (it usually opens by itself). Press **Play**, set the speed to 30x, choose a scenario (try 2001 or 2004) and
press **UAV's view (hide truth)** to hide the fire. Stop with **Ctrl+C**.

This is a **replay of saved runs**: three arms (baseline patrol, Model A trained on synthetic images, Model B trained on real images), one UAV,
8 scenarios, 600 s each. It starts no simulator and runs no neural network. The page says so at the top.

## Option 2 (about 20 minutes): live four-drone demonstration

Four simulated ArduPilot drones fly a grid-coverage mission over a simulated wildfire while you watch in the browser.
**This is a demonstration of the multi-UAV architecture, not a research experiment**; the research results are single-UAV.

**Prerequisites:** Linux (Ubuntu 22.04/24.04), about 2 GB free RAM, ArduPilot SITL built, `tmux`, and the Python packages in `requirements-sim.txt`.
Setting these up from scratch takes 30 to 60 minutes: follow PROFESSOR_GUIDE.md section 2.2. If a student or the project team has already prepared a machine, skip that.

```bash
cd Ardupilot-UAV-YOLO
source ~/venvs/wildfire/bin/activate        # the environment from the guide (it provides mavproxy.py)
python3 demo_multi.py --drones 4
```

Open **http://localhost:8080/multi.html**. Wait about 2 minutes ("Starting up" is normal), then watch about 15 minutes of flight.
Stop with **Ctrl+C**. `python3 demo_multi.py --dry-run` only checks the prerequisites.

**Watching from another computer** (for example the demo runs in a VM): start it with
`python3 demo_multi.py --drones 4 --host 0.0.0.0 --no-browser`, read the address it prints, and open `http://<that address>:8080/multi.html`.
With VirtualBox NAT, add a port-forward rule for TCP 8080 (guide section 7.4).

You will see four numbered, coloured drones with trails, each drone's assigned quadrant, visited cells, the simulated fire spreading,
elapsed time and coverage percentage. **Driven by ArduPilot SITL:** each drone's real position, altitude, speed and mode.
**Simulated:** the fire (the drones never see it). **Not shown, by design:** smoke detections, because no detector runs.
No performance claim is made.

## Option 3: run a new single-UAV experiment (baseline, Model A, Model B)

About 12 minutes per run. Needs the same setup as Option 2. Models A and B additionally need the trained weights, which are **not public and not in git**
(ask the team; guide section 2.3) and CPU PyTorch (`requirements-yolo.txt`). The baseline needs no weights.

```bash
python3 run_fresh.py --arm baseline --dashboard      # live view: http://localhost:8080
python3 run_fresh.py --arm A --dashboard
python3 run_fresh.py --arm B --dashboard
```

`--dry-run` lists anything missing. All arms use the same scenario and settings. A new simulation is **not** bit-identical to a recorded run
(wall-clock timing); expect similar behaviour, not equal numbers.

## What is and is not claimed

* Single UAV only in all recorded results. Multi-UAV behaviour was **not experimentally validated**.
* The UAV plans reactively (a detection above a threshold triggers an investigation). There is **no probability map**.
* The wildfire is a simulation; smoke is a flat rendered billboard, not a volume.
* Results are descriptive: 8 scenarios on 4 ignition cells, so no significance claims. See `docs/experiments/level3_final.md` and the report in `docs/report/`.

## If something goes wrong

| Symptom | Fix |
|---|---|
| Blank page / "connection refused" | Same machine: use `http://127.0.0.1:PORT`. Other machine: start with `--host 0.0.0.0`. Port busy: add `--port 9000`. |
| `mavproxy.py not found`, `sim_vehicle.py not found` | Activate the virtual environment; add `~/src/ardupilot/Tools/autotest` to `PATH` (guide 2.2). |
| "Starting up" for more than 4 minutes | Not enough RAM or SITL did not boot: close other programs, read `sitl.log` in the output folder named in the terminal. |
| `demo.py` missing after cloning | You are on `main`; clone with `--branch integration/yolo-perception-prep` as above. |

Not tested: Windows/macOS, a from-scratch ArduPilot install on a clean machine, and a 1-CPU machine for Option 2.
