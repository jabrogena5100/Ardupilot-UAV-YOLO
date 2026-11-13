# Verified baseline v1: Levels 1-3 (frozen 2026-11-09)

Machine-readable record: `baseline_v1.json`. Check it with `python3 eval/freeze_baseline.py --verify` (recomputes every hash; exit 1 on any mismatch).
"Frozen" means these inputs and outputs are pinned and any later change is detectable. It does **not** mean the results are scientifically validated:
Level 3 is prepared for independent review and has the limits listed in `docs/handoffs/level3_final_audit.md`.

## What is pinned
| item | where | count |
|---|---|---|
| Code commit | `code_commit` = `adc539fe75a43f7d231556ed8482e72dd18deda3` (final analysis code and regenerated Level 3 outputs after the third audit). The commit adding this directory is tagged `baseline-L1-3-v1` (local tag, not pushed) | 1 |
| Checkpoints (SHA-256, each cross-checked against `models/MODELS.md`) | `checkpoints` | 9 |
| Scientific configuration (eval pins, scenarios, fire, thresholds, swarm configs, domain-randomisation config, metrics/runner/analysis code) | `config_files`, `config_digest` | 15 |
| Level 3 per-run configs | `level3_run_configs` | 56 |
| Dataset records (`synth_smoke_v1` manifest and data.yaml); other pins are the file-list hashes inside `eval/datasets*.yaml` | `datasets` | 2 |
| Result artifacts under `docs/experiments/` (level2_v1, level2_v2, boreal_v2, level3_final + the report docs) | `result_artifacts`, `result_digest` | 209 |
| Raw Level 3 per-run files the analysis reads (manifest, run_meta, experiment_end, detections, fire_cells; gitignored) | `level3_raw_artifacts` | 290 |

## Which checkpoints belong to which level
- **Level 1/2 (v1 Boreal split):** Model A = `synth_smoke_best_v1` seeds 0-2; Model B = `real_boreal_best_v1` seeds 0-2.
- **Level 2 on the clip-disjoint split (`boreal_v2`):** Model B = `real_boreal_v2_best_seed0..2` (not comparable with the v1 B checkpoints; different train and test populations).
- **Level 3:** A = `synth_smoke_best_v1` seeds 0-2; **B = `real_boreal_v2_best_seed0..2`**, not the v1 Boreal checkpoints. Verified: every Level 3 run's `manifest.json` records the SHA-256 of the checkpoint it loaded, and all of them are in `baseline_v1.json` (A and B v2 seeds 8 runs each, A seed 0 has 9 including the repeat; 9 baseline runs load none).
- Checkpoints live outside git (`*.pt`); local copies under `local_data/backups/`, paths in the JSON.

## Provenance caveats carried by the baseline (details: `docs/experiments/level3_final/provenance.md`)
26 Level 3 runs were produced from a clean tree at `0c2f6c6`, 32 from a dirty tree (infrastructure edits only) in the 2-slot parallel harness; the A0 repeat's harness state is unrecoverable. The raw hashes pin the artifacts, not the harness state of the dirty runs.

## Re-verification done at freeze
Third-audit findings C1-C5 resolved (`docs/handoffs/level3_final_audit.md`); 299 tests pass; tables regenerate byte-identically and figures pixel-identically from the raw runs.
