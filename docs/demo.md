# Browser demonstration: Level 3 replay

A local web page that replays the **completed, frozen Level 3 experiments** (56 held-out runs: 8 scenarios x 7 arms,
single UAV, 600 s, oracle off). It is a **replay of saved files, not a live simulation**: it starts no SITL, loads no
model and never writes to a results directory. Every frame is read from artifacts the experiments already wrote.

## Launch

```bash
cd ~/src/Ardupilot-UAV-YOLO
python3 dashboard/replay_server.py --host 127.0.0.1 --port 8081
# open  http://localhost:8081
```

Needs only the Python standard library plus PyYAML. Memory is a few MB. `--host 0.0.0.0` (the default) lets another
machine on the network open it. The live dashboard (`dashboard/server.py`, port 8080) is a separate program for running
SITL experiments and is not needed for the demo.

One-time, only if the UAV paths are missing (they are cached under gitignored `local_data/level3/_replay_cache/`):

```bash
python3 dashboard/replay_build.py        # needs pymavlink; reads sitl_work/mav.tlog, writes only the cache
```

A single self-contained HTML file (no server, about 3 MB) for a laptop without the data:

```bash
python3 dashboard/replay_export.py --pilot final --out replay_final_standalone.html
```

## Using it

1. **Experiment**: pick *FINAL held-out evaluation* (default). Pilot sets are kept for the record and are flagged.
2. **Scenario**: pick one of the 8 scenarios (2001, 2002, 2004, 2005, 2006, 2012, 2014, 2015). Every arm in a scenario
   starts from the same ignition cell.
3. **Compare arms** (default tab): baseline, Model A and Model B side by side on **one shared clock**; choose which
   seed of A and B to show. **Single run** shows one arm with its detection list.
4. **Play / Pause**, **Restart**, speed 1x to 120x, the timeline slider, Space (play/pause), left/right arrows (5 s), Home.
   The *Jump to* buttons go to each arm's first true detection and the moment the UAV reached the ignition cell.
5. **UAV's view (hide truth)** hides the fire ground truth so only what the UAV has covered and detected remains;
   **Show everything** restores it. Individual layers have checkboxes.

Suggested story for a visitor: scenario 2001 or 2004, *Compare arms*, press Play at 30x, then jump to "first true
detection" for each arm and read the table below the maps.

## What is on screen

| Layer | Source file | Meaning |
|---|---|---|
| Orange / grey cells, yellow outline | `fire_cells_*.csv` | **Fire ground truth** (burning / burnt out / ignition cell). Not visible to the UAV. |
| Blue cells, white dot + trail | `fog_cells_*.csv`, `sitl_work/mav.tlog` | **UAV coverage and position** (1 Hz real MAVLink samples, no interpolation). |
| Green / red circles | `detections_*.csv` | **Detector observations** scored at ingest: green = fire within 1 cell at that moment, red = false positive. |
| Cyan / magenta dashed boxes | `agent_A.log` | **Reactive priority decisions**: cells the UAV was sent to investigate. Cyan = fire nearby (judged in hindsight), magenta = no fire (false-priority diversion). |

Table values come from the run files and the audit-corrected `docs/experiments/level3_final/per_run.csv`:
first true detection and first arrival at the ignition cell (capped at 600 s, shown as "at least 600 s"), coverage,
fire cells overflown, true/false detection observations, investigation time and distinct false-priority targets.
A missing value is shown as `n/a` with the reason; nothing is filled in. The baseline's first detection is `n/a`
(no detector), not 600 s.

## Scientific limits shown in the UI

* **Single UAV.** Multiple UAVs were not experimentally validated.
* **Reactive policy.** Detection above threshold triggers an investigation (`max(conf)` evidence, 20 s lifetime, 60 s
  cooldown). There is **no probability map**; `mapping/probability_map.py` is an empty stub and the page does not show one.
* **Truth vs knowledge.** The fire is ground truth from the fire simulator. The UAV learns about it only through its
  detector (oracle off in every run; checked by the tests).
* **Investigation targets are untimed.** The agent log has no timestamps, so these boxes are a whole-run summary and
  are not placed on the timeline. Investigation time is a count of HUD lines (about 1 per second, approximate).
* **Fire clock.** The initial fire is identical across arms, but fog steps the fire on the wall clock, so the realized
  spread differs slightly between runs (`level3_final.md` section 2a).
* **Path timing.** t = 0 for the UAV path is re-derived from the tlog; against fog's first-visit times the median offset
  is about 3 s (one cell per run differs by up to about 85 s, unexplained). Read path timing as plus or minus a few seconds.
* 8 scenarios on 4 ignition cells, 3 seeds per detector family: descriptive, no significance claims; Model A is tested
  in-domain on synthetic imagery, Model B is not. Smoke is a flat billboard, not a volume.

## Tests

```bash
python3 -m unittest tests.test_replay_final tests.test_replay_data
```

`tests/test_replay_final.py` checks, against the recorded data, that the run list is exactly the 56 matrix runs, that
displayed metrics equal the recorded table and the raw detection/fire/coverage files, that capped values are flagged and
not imputed, that investigation counts agree with the analysis, that the UAV timeline is monotone and its coordinates
map to the same cells fog recorded, and that every run is single-UAV with the oracle off. The real-data tests skip when
`local_data/` is absent.
