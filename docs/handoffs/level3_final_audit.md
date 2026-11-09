# Handoff to Codex: Level 3 final evaluation, corrected audit package (2026-11-09)

Read-only review request. Do not edit results. Verify, do not trust: run `git status` and `git log --oneline -10`.
**The package is prepared for independent review; it is not declared scientifically validated.** The first version of this handoff (commit `86bb2df`) contained a factual error, corrected below.

## What to review
`docs/experiments/level3_final.md` (corrected), the generated tables in `docs/experiments/level3_final/`, and `eval/level3_report_tables.py` / `eval/level3_analyze.py`. Protocol: `docs/experiments/level3_methodology.md` sections 6-9 and 15. A first read-only audit (by the same model that produced the analysis, so not independent) produced the corrections.

## Corrections made and the evidence for each
| # | issue | correction | evidence to check |
|---|---|---|---|
| 1 | Docs said every run is `git_dirty: true`. | 26 runs clean (`0c2f6c6`, sequential), 32 dirty (30 main + 2 repeats, 2-slot parallel). | `run_meta.json` per run; `provenance.csv`; mtimes of `eval/level3_pairs.py` (09:47:45), `launch_sitl_swarm.py` (09:17:43), `eval/level3_queue.py` (09:18:43). |
| 2 | Dirty edits not characterised. | `git diff 0c2f6c6 661c72d -- eval/level3_pairs.py launch_sitl_swarm.py`: slots, ports, tagged cleanup only; `scientific_config_identical` true in all 32 slot runs; thresholds/fire/scenario hashes equal in all 58. | the diff; run_meta fields. |
| 3 | A0 repeat provenance. | It started 09:35:11, before the last `eval/level3_pairs.py` edit: it ran on an intermediate, uncommitted version that cannot be recovered. Its failed predecessor (09:34, rc=1, 1 s) is uninvestigated. | `queue.log`, `results_repeat/l3-A0-2001/run_meta.json`. |
| 4 | Scenarios treated as independent (p = 0.0078). | Primary cluster = the 4 ignition cells; minimum p = 0.125; nothing claimed significant; Holm shown; direction counts are the evidence; scenario-level kept as descriptive. | `cluster_stats.{md,json}`, `eval/level3_scenarios.json` (cell pairing). |
| 5 | Censored times read as means. | 15/56 runs have no ignition-cell visit by 600 s (B 14/24, A 1/24); 1 run has no true detection; times labelled lower bounds. | `censoring.md`, `per_run.csv` (empty fields). |
| 6 | `det_false` read as incidents. | It counts repeated detection observations; distinct-cell counts added (`per_run_extra.csv`); INVESTIGATE columns are log lines, not exact seconds. | `fog/state.py::ingest_detection`, `detections_*.csv`. |
| 7 | "Seed spread" was the SD over scenarios. | SD over the 3 seed values (each = mean over 8 scenarios), ddof = 1. | `seed_summary.md/csv`. |
| 8 | Planner over-described. | Reactive detection-triggered investigation policy; probability map not implemented (`mapping/probability_map.py` is 0 lines). | `planning/priority.py`, `swarm_agent.py` step 5. |
| 9 | Repeats called "jitter". | Both repeats ran parallel (A0 on the lost harness) vs sequential originals and are +13-14 s in the same direction: confounded, not a jitter estimate. | `repeats.json`, `level3_final.md` section 8. |
| 10 | Weights provenance asserted without check. | Verified per run: every manifest weights SHA-256 is in `models/MODELS.md`, one hash per model tag. | `manifest.json`, `models/MODELS.md`. |

## Remaining uncertainties and limitations (not fixable without new runs)
- Execution mode (sequential vs parallel on one 1-vCPU VM) is confounded with scenario and cell. A sensitivity check without scenario 2005 (the only scenario mixing modes) keeps all direction counts. No scenario was flown in both modes except the two repeats.
- Four ignition cells, 8 scenarios, 3 seeds per family: all intervals are crude; the p floor is 0.125.
- Right-censoring at 600 s; no survival model; Model B's ignition-cell delay is a lower bound.
- Model A is in-domain on synthetic imagery, Model B cross-domain; this largely restates the Level 2 gap. Per-arm thresholds differ from the Level 2 operating point (linkage is descriptive, n = 6 checkpoints).
- Reactive policy only; no probability map; no monitoring benefit of perception is shown; policy effects and detector effects are not separated.
- One-cell truth radius; time-agnostic diversion truth; flat-billboard smoke; one UAV; SITL timing not bit-reproducible.
- The trajectory figure is illustrative; scenarios 2001 and 2004 were picked after the aggregate analysis. Metrics were not recomputed from the tlogs (HUD cell log used for the figure; `fog` CSVs for metrics).
- The first-attempt failure of the baseline repeat is uninvestigated.

## Specific questions for the reviewer
1. Is the cell-cluster analysis (4 clusters, exact sign-flip floor 0.125, direction counts) an adequate framing, or should the report use a different summary (e.g. per-cell tables only)?
2. Is the mtime-based reconstruction of the dirty state acceptable, or should the dirty runs be re-run in sequential mode for a subset to bound the execution-mode effect?
3. Does any sentence in `level3_final.md` section 4 or 5 still exceed what the data support?
4. Are the linkage table's caveats (different operating points, n = 3 per family) sufficient?

## Reproduce
`python3 eval/level3_analyze.py local_data/level3/final/results/* --out OUT` (unchanged), then `python3 eval/level3_report_tables.py` (writes the tables and figure into `docs/experiments/level3_final/`; fixed bootstrap seed). `tests/test_level3_report_tables.py` checks the generated tables against `stats.json` and the committed per-run data. Raw runs (2.8 GB) are gitignored under `local_data/level3/final/`.
