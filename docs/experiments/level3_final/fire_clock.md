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

Scenarios with identical burned sets across all 7 arms: [2001, 2002, 2004, 2006, 2012, 2015].

**What was measured.**

- sequential runs: 22 of 22 perception runs have a fire ratio above 1.0 (their ignitions are later than the same-scenario baseline's); Spearman correlation between fog tick rate and fire ratio = -0.84 (n = 22).
- parallel runs: 25 of 26 perception runs have a fire ratio above 1.0 (their ignitions are later than the same-scenario baseline's); Spearman correlation between fog tick rate and fire ratio = -0.87 (n = 26).
- Family difference in the fire ratio (mean A minus mean B) in scenarios where both families ran in parallel mode: 2006: +0.007, 2012: +0.018, 2014: +0.010, 2015: -0.007.
- Within-scenario Spearman correlation between a perception run's fire ratio and its first true detection time (parallel scenarios with at least 5 runs): 2006: +0.09 (n = 6); 2012: +0.37 (n = 6); 2014: -0.66 (n = 6); 2015: +0.03 (n = 6).

**What these data do and do not support.** Supported: the realized fire differs across arms in the parallel scenarios (later ignitions in the perception runs than in the baseline's, different burned sets in scenarios 2014 and 2005), so contrasts there compare unequal fires; fog's tick rate and the fire ratio are strongly rank-correlated, which is consistent with the wall-clock scheduling mechanism (association, not proof of the mechanism); the perception runs' lower tick rate is consistent with, but does not prove, a heavier load as its cause. **Not established:** the size or direction of any bias this introduces in the arm contrasts. The within-scenario association between fire ratio and first true detection time has inconsistent sign (small n), so no effect on detection timing is demonstrated or excluded; the family difference in the fire ratio is small and mixed in sign over 4 scenarios, so these data show no family-specific fire-clock shift but cannot exclude one. The ratio is relative to the baseline run, whose own clock is also affected, so it is not an absolute clock error. The ignition cell burns from t = 0 in every run, so the first-visit time of the ignition cell does not depend on the fire clock directly (the UAV's route can).
