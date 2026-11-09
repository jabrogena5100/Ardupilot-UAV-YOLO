# Fire-clock check (generated)

**Why this exists.** `fog_tracker.py` schedules each fire step at `time.time() + dt_seconds` after the previous step, so the simulated fire advances in wall-clock time plus loop latency, not in run-clock seconds. When fog's tick loop is slowed (two SITL runs sharing one vCPU, perception load), the fire advances fewer steps per run second. The **initial fire configuration** (ignition cell, fire seed, `p_ignite`, `burn_steps`) is identical across arms; the **realized fire trajectory** is not.

`fire ratio` = median over cells burned in both the run and the same-scenario baseline of ignition time(run) / ignition time(baseline); 1.00 means the same progression, 1.05 means the run's fire was about 5% slower. `ticks/s` is fog's logged tick rate (nominal 2.0).

| mode | arm group | runs | fire ratio mean (min-max) | fog ticks/s mean (min) | worst tick gap (s) |
|---|---|---|---|---|---|
| seq | baseline | 4 | reference | 1.99 (1.99) | 0.5 |
| seq | A | 12 | 1.003 (1.001-1.006) | 1.98 (1.97) | 0.7 |
| seq | B | 10 | 1.004 (1.001-1.007) | 1.98 (1.97) | 0.6 |
| par | baseline | 4 | reference | 1.91 (1.90) | 1.4 |
| par | A | 12 | 1.058 (1.005-1.086) | 1.79 (1.73) | 21.4 |
| par | B | 14 | 1.050 (0.999-1.085) | 1.81 (1.75) | 22.3 |

**Burned-cell counts at 600 s that differ across arms within a scenario** (final burned set is also sensitive to the clock):

- scenario 2005: A0 25, A1 25, A2 25, B2s0 25, B2s1 23, B2s2 25, baseline 25
- scenario 2014: A0 16, A1 16, A2 15, B2s0 16, B2s1 16, B2s2 16, baseline 21

Scenarios with identical burned sets across all 7 arms: [2001, 2002, 2004, 2006, 2012, 2015]. In the sequential scenarios the fire ratio is within about 1% of the baseline; in the parallel scenarios it reaches about 1.09, with later ignitions in the perception arms (which also load the VM more) than in the baseline. Consequences: arm contrasts in parallel scenarios compare unequal realized fires; the first true detection, the true/false labelling of detections and diversion targets, and the burned-set-dependent counts are affected; the ignition cell burns from t = 0 in every run, so its first-visit time does not depend on the fire clock but the UAV's route can.
