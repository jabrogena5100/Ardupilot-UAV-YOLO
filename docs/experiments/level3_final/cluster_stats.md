# Contrasts with the four ignition cells as clusters (generated)

Unit of analysis: the **ignition cell** (4 clusters, each flown in 2 scenarios with different fire seed and texture). For each contrast the scenario differences (family mean over seeds, per scenario) are averaged within a cell, giving 4 cell differences.
`CI` is a percentile bootstrap that resamples the 4 cells (10,000 draws, seed 0); with 4 clusters it is a crude, unstable interval and is **descriptive only**. `p` is the exact sign-flip p over the 4 cell differences: its **minimum is 0.125**, so nothing can reach 0.05; Holm adjustment over all 16 contrasts is shown for completeness. Time-to-event contrasts use values censored at 600 s (lower bounds), see `censoring.md`. **Direction counts (cells/scenarios with the same sign) are the supported evidence.**

| metric | contrast | mean cell diff | cell CI (descriptive) | cells + / - | scenarios + / - | p over cells | Holm p |
|---|---|---|---|---|---|---|---|
| first true detection (s) | A-B | -119.5 | [-170.9, -68.0] | 0/4 | 0/8 | 0.125 | 1.000 |
| first visit of ignition cell (s) | A-B | -219.7 | [-238.7, -207.5] | 0/4 | 0/8 | 0.125 | 1.000 |
| first visit of ignition cell (s) | A-baseline | 46.6 | [18.3, 98.3] | 4/0 | 6/2 | 0.125 | 1.000 |
| first visit of ignition cell (s) | B-baseline | 266.3 | [230.3, 337.1] | 4/0 | 8/0 | 0.125 | 1.000 |
| grid coverage at 600 s | A-B | 0.035 | [0.023, 0.048] | 4/0 | 8/0 | 0.125 | 1.000 |
| grid coverage at 600 s | A-baseline | -0.026 | [-0.031, -0.020] | 0/4 | 0/8 | 0.125 | 1.000 |
| grid coverage at 600 s | B-baseline | -0.061 | [-0.070, -0.052] | 0/4 | 0/8 | 0.125 | 1.000 |
| true detection observations / run | A-B | -13.9 | [-39.8, 11.2] | 1/3 | 2/6 | 0.500 | 1.000 |
| false detection observations / run | A-B | -55.4 | [-83.8, -32.6] | 0/4 | 0/8 | 0.125 | 1.000 |
| distinct cells with a false detection / run | A-B | -6.2 | [-7.7, -4.5] | 0/4 | 0/8 | 0.125 | 1.000 |
| INVESTIGATE log lines on burned-adjacent targets (~s) | A-B | -63.4 | [-176.3, 26.4] | 2/2 | 3/5 | 0.500 | 1.000 |
| INVESTIGATE log lines on burned-adjacent targets (~s) | A-baseline | 173.7 | [106.2, 240.3] | 4/0 | 8/0 | 0.125 | 1.000 |
| INVESTIGATE log lines on burned-adjacent targets (~s) | B-baseline | 237.1 | [168.8, 295.0] | 4/0 | 8/0 | 0.125 | 1.000 |
| INVESTIGATE log lines on false targets (~s) | A-B | -79.0 | [-122.9, -35.1] | 0/4 | 1/7 | 0.125 | 1.000 |
| INVESTIGATE log lines on false targets (~s) | A-baseline | 1.6 | [0.0, 4.8] | 1/0 | 2/0 | 1.000 | 1.000 |
| INVESTIGATE log lines on false targets (~s) | B-baseline | 80.6 | [38.2, 122.9] | 4/0 | 7/0 | 0.125 | 1.000 |

**Scenario-level differences (descriptive; scenarios sharing a cell are not independent)**

| metric | contrast | 2001 | 2002 | 2004 | 2005 | 2006 | 2012 | 2014 | 2015 |
|---|---|---|---|---|---|---|---|---|---|
| first true detection (s) | A-B | -104 | -69 | -248 | -22 | -228 | -77 | -84 | -124 |
| first visit of ignition cell (s) | A-B | -144 | -109 | -326 | -273 | -112 | -321 | -172 | -301 |
| first visit of ignition cell (s) | A-baseline | 21 | 65 | 65 | 25 | 101 | -32 | 182 | -54 |
| first visit of ignition cell (s) | B-baseline | 165 | 174 | 391 | 298 | 213 | 290 | 353 | 247 |
| grid coverage at 600 s | A-B | 0.040 | 0.030 | 0.057 | 0.060 | 0.020 | 0.010 | 0.033 | 0.033 |
| grid coverage at 600 s | A-baseline | -0.020 | -0.013 | -0.020 | -0.017 | -0.027 | -0.053 | -0.033 | -0.023 |
| grid coverage at 600 s | B-baseline | -0.060 | -0.043 | -0.077 | -0.077 | -0.047 | -0.063 | -0.067 | -0.057 |
| true detection observations / run | A-B | -48 | -56 | -24 | -49 | 44 | 29 | -2 | -5 |
| false detection observations / run | A-B | -61 | -15 | -158 | -32 | -52 | -37 | -35 | -53 |
| distinct cells with a false detection / run | A-B | -8 | -2 | -10 | -4 | -6 | -5 | -7 | -8 |
| INVESTIGATE log lines on burned-adjacent targets (~s) | A-B | -202 | 58 | -63 | -276 | 88 | 24 | -72 | -64 |
| INVESTIGATE log lines on burned-adjacent targets (~s) | A-baseline | 50 | 278 | 227 | 91 | 182 | 258 | 198 | 105 |
| INVESTIGATE log lines on burned-adjacent targets (~s) | B-baseline | 252 | 220 | 290 | 367 | 95 | 234 | 270 | 169 |
| INVESTIGATE log lines on false targets (~s) | A-B | -47 | -42 | -169 | 6 | -143 | -57 | -51 | -128 |
| INVESTIGATE log lines on false targets (~s) | A-baseline | 6 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| INVESTIGATE log lines on false targets (~s) | B-baseline | 54 | 42 | 169 | 0 | 143 | 57 | 51 | 128 |

**Sensitivity by execution era (fire-clock confound).** In the 2-slot parallel runs fog's wall-clock-scheduled fire advanced more slowly and differently per arm (`fire_clock.md`), so contrasts there compare unequal realized fires. Direction counts over the sequential-only scenarios [2001, 2002, 2004] (3 cells), the parallel-only scenarios [2006, 2012, 2014, 2015] (all 4 cells), with the mixed-mode scenario(s) [2005] in neither:

| metric | contrast | sequential scen + / - | sequential cells + / - | parallel scen + / - | parallel cells + / - |
|---|---|---|---|---|---|
| first true detection (s) | A-B | 0/3 of 3 | 0/3 of 3 | 0/4 of 4 | 0/3 of 3 |
| first visit of ignition cell (s) | A-B | 0/3 of 3 | 0/3 of 3 | 0/4 of 4 | 0/3 of 3 |
| first visit of ignition cell (s) | A-baseline | 3/0 of 3 | 3/0 of 3 | 2/2 of 4 | 2/1 of 3 |
| first visit of ignition cell (s) | B-baseline | 3/0 of 3 | 3/0 of 3 | 4/0 of 4 | 3/0 of 3 |
| grid coverage at 600 s | A-B | 3/0 of 3 | 3/0 of 3 | 4/0 of 4 | 3/0 of 3 |
| grid coverage at 600 s | A-baseline | 0/3 of 3 | 0/3 of 3 | 0/4 of 4 | 0/3 of 3 |
| grid coverage at 600 s | B-baseline | 0/3 of 3 | 0/3 of 3 | 0/4 of 4 | 0/3 of 3 |
| true detection observations / run | A-B | 0/3 of 3 | 0/3 of 3 | 2/2 of 4 | 2/1 of 3 |
| false detection observations / run | A-B | 0/3 of 3 | 0/3 of 3 | 0/4 of 4 | 0/3 of 3 |
| distinct cells with a false detection / run | A-B | 0/3 of 3 | 0/3 of 3 | 0/4 of 4 | 0/3 of 3 |
| INVESTIGATE log lines on burned-adjacent targets (~s) | A-B | 1/2 of 3 | 1/2 of 3 | 2/2 of 4 | 2/1 of 3 |
| INVESTIGATE log lines on burned-adjacent targets (~s) | A-baseline | 3/0 of 3 | 3/0 of 3 | 4/0 of 4 | 3/0 of 3 |
| INVESTIGATE log lines on burned-adjacent targets (~s) | B-baseline | 3/0 of 3 | 3/0 of 3 | 4/0 of 4 | 3/0 of 3 |
| INVESTIGATE log lines on false targets (~s) | A-B | 0/3 of 3 | 0/3 of 3 | 0/4 of 4 | 0/3 of 3 |
| INVESTIGATE log lines on false targets (~s) | A-baseline | 1/0 of 3 | 1/0 of 3 | 0/0 of 4 | 0/0 of 3 |
| INVESTIGATE log lines on false targets (~s) | B-baseline | 3/0 of 3 | 3/0 of 3 | 4/0 of 4 | 3/0 of 3 |

Where the two columns agree the direction does not depend on the execution mode; where a sequential-only count rests on 3 cells and 3 scenarios it is weak evidence on its own. The sequential-only subset is the less confounded one for contrasts against the baseline.

Cells: (0,1) = scenarios [2006, 2015]; (1,0) = scenarios [2004, 2014]; (2,1) = scenarios [2002, 2012]; (3,2) = scenarios [2001, 2005]
