# Baseline v1 amendments (documentation only)

`baseline_v1.json` is not rewritten. After the freeze, three pinned documentation files were edited on 2026-11-11 (final Codex audit corrections). `python3 eval/freeze_baseline.py --verify` therefore reports exactly these three mismatches and nothing else (no checkpoint, config, dataset, level-3 run config or raw artifact differs).

| file | frozen sha256 | current-at-amendment sha256 | change |
|---|---|---|---|
| `models/MODELS.md` | 7116b553b6d782ac2aa20cf13f9a78d0801a1041fab8722375d64ff9b9474bc6 | 3988e7c4ef78b75bc60a4827ff65ffef3da1228944ad35012749e38f7872d887 | Boreal-v2 local backup now documented (hashes re-verified) |
| `docs/experiments/boreal_v2_results.md` | d9aca368b38bff0b41a4e0f939da6312164330f8945143768d4694685814ade0 | 2332c35189d4c4e8bd3c4a8853ad178f1a0114b68c6dc9d69e4619f37bd3d637 | same backup correction |
| `docs/experiments/level3_final.md` | 00061ede3435dc444c83fda3d03c297850a5ce162c594a45102a887f03d70891 | 3444722eb4db88829432cf4edf01fbb55419f85bb4bcbf0e15c59adb14005b46 | added a reading rule (no causal / planner-benefit / improved-monitoring claim); no number changed |
