# Handoff to Codex: Level 3 final evaluation, final corrected package (2026-11-09)

Read-only review request. Do not edit results. Verify, do not trust: run `git status` and `git log --oneline -10`.
**Prepared for independent review; not declared scientifically validated.** Passing tests show the code does what it is specified to do on known inputs; they do not establish scientific validity. The probability-map planner proposed in `CLAUDE.md` is **not implemented** (`mapping/probability_map.py` is 0 lines); what ran is a reactive detection-triggered investigation policy.
History: first handoff `86bb2df` (contained a provenance error); first correction pass `28a2faa`; two read-only audits (the second recomputed metrics from raw files); this is the final targeted correction pass.

## Resolved findings
| finding | resolution | evidence |
|---|---|---|
| Provenance said every run was dirty | 26 clean (sequential, `0c2f6c6`) + 32 dirty (30 main + 2 repeats, 2-slot parallel); dirty edits are infrastructure only; A0 repeat ran on an unrecoverable intermediate harness | `provenance.md`, `level3_final.md` s2 |
| Scenarios treated as independent | 4 ignition cells are the clusters; p floor 0.125; no significance claimed; direction counts | `cluster_stats.md`, s3 |
| Censoring | 15/56 ignition-cell visits and 1 detection censored; times are lower bounds; observed-only limits documented | `censoring.md`, s6, s7a |
| `det_false` read as incidents | repeated observations; distinct-cell counts added | s7, `per_run_extra.csv` |
| Seed SD | SD over 3 seed values, ddof = 1 | `seed_summary.md` |
| Planner over-described | reactive policy; map not implemented | s1 |
| **Fire clock** (second audit) | documented: initial fire configuration identical, realized fire trajectory is not (wall-clock fire scheduling); measured per run; sensitivity by execution era; wrong "identical/deterministic" wording corrected | `fire_clock.md`, `level3_final.md` s2a, `level3_methodology.md` s16, `cluster_stats.md` |
| Baseline first detection shown as 600 s in CSV | now `NA` in `seed_summary.csv` | `seed_summary.csv`, test |
| Sensitivity only dropped scenario 2005 | replaced by sequential-only vs parallel-only direction counts across all affected scenarios | `cluster_stats.md` |
| Radius-one matching, observed-only censoring, log-line proxy | documented with consequences | `level3_final.md` s7a |
| Analysis code untested | `tests/test_level3_analyze.py` (13 tests, synthetic run directories, hand-computed answers) and `tests/test_level3_report_calcs.py` (16 tests: SD ddof, cell averaging, sign-flip counting, Holm, era subsets, censoring bookkeeping, fire-clock ratio, linkage seed mapping, regeneration) | see Test evidence |
| Tick-rate estimator off by one | `fog_ticks_per_s` now uses intervals per second; tables regenerated | `eval/level3_report_tables.py` |

## Unresolved experimental limitations (cannot be fixed without new runs)
- **Fire clock:** realized fires differ across arms in the parallel scenarios (2006, 2012, 2014, 2015, parallel runs of 2005), most against the baseline. A small sequential rerun of the parallel scenarios would bound it; none was run.
- Execution mode (sequential vs parallel on one vCPU) is confounded with scenario and cell; repeats (+13-14 s) are confounded and are not a jitter estimate; the A0 repeat's harness state is lost.
- 4 ignition cells, 8 scenarios, 3 seeds per family: crude intervals, p floor 0.125; right-censoring (B 14/24 ignition-cell visits); one-cell truth radius; flat-billboard smoke; one UAV; Model A in-domain vs Model B cross-domain; per-arm thresholds differ from the Level 2 operating point.
- Reactive policy only: no probability map, no confidence-aware trade-off, no separation of policy effect from detector effect. No monitoring benefit of perception is shown.

## Conclusions that remain supported (descriptive, with the limits above)
- Model A detects sooner, raises far fewer false detection observations (0.3 vs 55.7 per run; distinct false cells 0.2 vs 6.4) and covers more than Model B, with the same direction in all 4 cells and in both the sequential-only and parallel-only scenarios. Model B's ignition-cell delay is a lower bound.
- Both perception arms cover less than the patrol baseline (4/4 cells, both eras). B reaches the ignition cell later than the baseline in both eras. **A versus baseline on ignition-cell visit is not consistent across eras** (3/3 sequential, 2/4 parallel scenarios) and is not claimed.
- Detections drive investigation (every INVESTIGATE target is a detected cell in all 48 perception runs; none in the baseline; oracle off).
- Not supported: significance statements, a monitoring benefit, real-flight performance, a quantity for the B delay, a probability-map result.

## Test evidence (what it does and does not show)
Full suite: 286 tests, OK (255 before this pass + 13 analyzer + 16 report-calculation + 2 documentation-consistency tests), including the regeneration check against the raw runs. Mutation checks on a scratch copy (not part of the repo; run with bytecode caching off after a stale-`.pyc` artifact contaminated a first attempt): 20 targeted code mutations (flipped correctness flag, widened ignition filter, truth radius 1 to 2, family map swap, censor value, ddof, sign-flip tie handling, cell mean to median, Holm, fire ratio, `NA` baseline, pre-gate filter) are each caught by at least one test. The cell mean/median mutation is only distinguishable with a patched 3-scenario cell because every real cell has 2 scenarios. These show the formulas are pinned on synthetic inputs, not that the experiment design is valid. Regeneration from raw artifacts is byte-identical (`TestRegeneration`; skipped when raw runs are absent). I did not recompute metrics from the tlogs.

## Questions for the reviewer
1. Is the fire-clock treatment (document, report by-era sensitivity, do not rerun) adequate for a 24-page report, or is a small sequential rerun of the parallel scenarios needed before any statement about contrasts against the baseline?
2. Is the cell-cluster framing (4 clusters, floor 0.125, direction counts) acceptable?
3. Does any sentence in `level3_final.md` sections 4, 5 or 9 still exceed what the data support?

## Reproduce
`python3 eval/level3_analyze.py local_data/level3/final/results/* --out OUT` (unchanged), then `python3 eval/level3_report_tables.py` (writes tables and figure into `docs/experiments/level3_final/`, fixed bootstrap seed). Raw runs (2.8 GB) are gitignored under `local_data/level3/final/`.
