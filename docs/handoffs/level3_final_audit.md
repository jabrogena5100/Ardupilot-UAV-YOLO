# Handoff to Codex: Level 3 final evaluation audit package (2026-11-09)

Read-only audit request. Do not edit results. Verify, do not trust: run `git status`, `git log --oneline -10`.

## What to audit
`docs/experiments/level3_final.md` claims (56 paired runs, 8 held-out scenarios x 7 arms, scenario-level statistics, 2 repeats). Protocol: `docs/experiments/level3_methodology.md` sections 7-9 and 15.

## Evidence
- Committed: `docs/experiments/level3_final/{per_run.csv,stats.json,repeats.json,fig_level3_scenarios.png,matrix.json}`; code `eval/level3_analyze.py`, `eval/level3_queue.py`, `eval/level3_pairs.py`.
- Local only (gitignored, 2.8 GB): `local_data/level3/final/{results,results_repeat,queue.log,queue_logs}`. Each run dir has `run_meta.json` (config/fire/threshold/scenario hashes, git commit, dirty flag), `manifest.json` (weights SHA-256, oracle flag), fog CSVs, agent log, tlog.
- Reproduce: `python3 eval/level3_analyze.py local_data/level3/final/results/* --out OUT`; compare with `stats.json`.

## Specific checks requested
1. Every `run_meta.json` has `fog_exit_code` 0, empty `leftover_after_cleanup`, oracle off, weights hashes matching `models/MODELS.md`, thresholds/fire/scenario hashes equal across runs.
2. Provenance gap: all runs have `git_dirty: true` at base commit `0c2f6c6` (uncommitted harness edits to `eval/level3_pairs.py`, `launch_sitl_swarm.py`, new `eval/level3_queue.py`, committed afterwards). Judge whether this weakens reproducibility claims.
3. Censoring: first-detection and ignition-cell first-visit are censored at 600 s; B2s1 scenario 2004 has 0 true detections and 305 false ones. Does censoring bias the A-B contrast?
4. Unit of analysis (scenario, seed-mean) and sign-flip p (min 0.0078, no multiplicity correction); overclaiming in the interpretation section?
5. Repeat estimate is n = 1 per config (baseline and A0, scenario 2001; 13-14 s shifts). Failed first baseline-repeat attempt (rc=1, 1 s) is uninvestigated.
6. Confounds: Model A is in-domain on synthetic imagery; Model B thresholds sit at the detector floor for 2 of 3 seeds; one-cell truth radius; time-agnostic diversion truth.
7. Anything in the doc not traceable to the artifacts above.

## Process note
A VM reboot occurred after the queue finished (queue log end 13:05 UTC; audit 21:43 UTC); no run overlapped it.
