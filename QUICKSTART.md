# Quick start: watch the wildfire-monitoring demo (about 2 minutes)

This opens a **replay of the 56 completed Level 3 experiments** (baseline, Model A, Model B; one UAV; 8 scenarios).
It is a recording, not a new simulation. It needs only Python 3.9+ (3.10 tested). No GPU, no ArduPilot, no network.

```bash
git clone --branch integration/yolo-perception-prep https://github.com/jabrogena5100/Ardupilot-UAV-YOLO.git   # or the URL you were given
cd Ardupilot-UAV-YOLO
python3 demo.py
```

That is **three commands**. Your browser opens **http://localhost:8081**. If it does not, type that address yourself.
The `--branch` matters: the demo is not on `main`. (If `git clone` is not possible, unzip the repository and start at `cd`.)

## What you should see

* Left: a 10 x 10 grid map for **Baseline**, **Model A** and **Model B** side by side, all on one clock.
* Blue cells and a white dot with a trail: where the UAV has flown. Orange cells: the actual fire (ground truth).
* Green/red circles: smoke detections (green = fire really was nearby, red = false alarm). Dashed boxes: cells the
  UAV was sent to investigate.
* Below the maps: a table of the recorded results. Legend and plain-language explanation are on the page.

## Use it

1. Press **Play** (or the space bar). Set the speed to 30x. Drag the slider to move through the 600 s run.
2. Choose another **Scenario** from the drop-down. Try 2001 or 2004.
3. Click **UAV's view (hide truth)** to hide the fire, leaving only what the UAV could know.
4. Click the **Jump to** buttons to go to each arm's first true detection.

## Stop, restart, reset

* Stop: press **Ctrl+C** in the terminal. Start again with `python3 demo.py`.
* Reset a replay: the **Restart** button. There is no state to clean up; the replay never writes anything.
* Another port: `python3 demo.py --port 9000`. Running inside a VM: `python3 demo.py --host 0.0.0.0` (see the guide).

## Blank page?

Reload; try `python3 demo.py --no-browser` and open the address by hand; confirm with `python3 check_setup.py`.
More in [docs/PROFESSOR_GUIDE.md](docs/PROFESSOR_GUIDE.md) section 7.

## Optional: four-drone live demonstration (needs ArduPilot SITL; see the guide, section 8b)

```bash
python3 demo_multi.py --drones 4        # then open http://localhost:8080/multi.html ; Ctrl+C stops it
```

Four simulated ArduPilot copters cover a grid over a simulated fire (about 2 min startup, then about 15 min). It is a **demonstration of the multi-UAV architecture,
not a research experiment**, and is separate from the replay above. Needs the setup of guide section 2.2; no weights or GPU.

## Honest limits

Replay only. Single UAV; multi-UAV behaviour was **not** experimentally validated. There is **no probability map**; the
planner is reactive (a detection above a threshold triggers an investigation). The fire is simulated ground truth the UAV
cannot see. Full guide, including running a **new** experiment: [docs/PROFESSOR_GUIDE.md](docs/PROFESSOR_GUIDE.md).
