# Right-censoring and coverage (generated)

A run is **censored** for a time-to-event metric when the event was not observed within the 600 s window. The true time is then only known to exceed 600 s; it is never an exact time. Means that substitute 600 s are lower bounds.

| arm | runs | ignition cell visited by 600 s | no observed visit (censored) | observed-only median visit (s) | true detection observed | no true detection (censored) | coverage mean +- SD over scenarios |
|---|---|---|---|---|---|---|---|
| baseline | 8 | 8 | 0 | 207 | n/a (no detector) | n/a | 0.209 +- 0.010 |
| A0 | 8 | 7 | 1 | 260 | 8 | 0 | 0.184 +- 0.023 |
| A1 | 8 | 8 | 0 | 247 | 8 | 0 | 0.180 +- 0.026 |
| A2 | 8 | 8 | 0 | 236 | 8 | 0 | 0.185 +- 0.026 |
| B2s0 | 8 | 4 | 4 | 427 | 8 | 0 | 0.149 +- 0.031 |
| B2s1 | 8 | 3 | 5 | 363 | 7 | 1 | 0.140 +- 0.028 |
| B2s2 | 8 | 3 | 5 | 247 | 8 | 0 | 0.154 +- 0.013 |

Total runs with no observed ignition-cell visit by 600 s: **15 of 56**. The observed-only median is conditional on a visit having occurred and is biased low for arms with many censored runs; do not read it as the arm's typical time.
Censored runs by scenario (ignition-cell visit): 2001: 0, 2002: 1, 2004: 2, 2005: 0, 2006: 4, 2012: 2, 2014: 3, 2015: 3.
Runs with no observed true detection (perception arms): l3-B2s1-2004.

Direction of the bias: substituting 600 s for an unobserved time understates that run's true time, and the arm with more censored runs (B for the ignition-cell visit) is understated more, so time contrasts between arms are lower bounds on the size of the difference, not estimates of it. No survival model was fitted (4 clusters, 8 scenarios).
