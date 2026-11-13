# Boreal v2: post-experiment scientific audit (read-only)

Date: 2026-11-02. Auditor: Claude (cloud session), **not independent of the author** (same session lineage): treat as a self-audit to be confirmed by the owner/Codex.
Scope: the completed Boreal v2 experiment at commit `b80a465` (branch `science/boreal-v2`). Nothing was retrained, re-run on a GPU, edited or re-scored; the audit read committed files only.
Re-runnable: `python3 -m eval.boreal_v2_audit [--recompute]` (committed result: `docs/experiments/boreal_v2/audit.json`, 49 checks, 49 VERIFIED, 0 FAILED, bootstrap recompute included).

## Verdict: PASS WITH DOCUMENTED CAVEATS
No claim in `boreal_v2.md` / `boreal_v2_results.md` was contradicted. The experiment is internally consistent and reproducible offline for every detection metric. It is **not** evidence for a tightly estimated domain gap (8 test clips, 3 seeds), and training reproducibility is demonstrated only for 4 epochs. Boreal v1 and the historical Level 2 v2 baseline are untouched.

## A. VERIFIED (recomputed or read from a recorded artifact in this audit)
| # | Claim | Evidence (audit check id) |
|---|---|---|
| 1 | Three Model B v2 checkpoints, hashes `090c0f4c...` / `5a9616ce...` / `c4d443a8...`, recorded in each `train_config.json` | `ckpt.seed{0,1,2}.hash_recorded` |
| 2 | Each completed 100/100 epochs (`complete: true`, 100 `results.csv` rows) | `ckpt.seed*.complete_100_epochs` |
| 3 | Training args equal the protocol (YOLOv8s COCO, 100 epochs, patience 100, imgsz 640, batch 32, deterministic, workers 8, AdamW 0.002 / 0.9 / 0.0, seeds 0/1/2, data = `boreal_v2`), ultralytics 8.4.173, torch 2.8.0 | `ckpt.seed*.protocol_args`, `.env`, `.data_is_boreal_v2` |
| 4 | `cache: ram` appears in every seed's recorded `args.yaml` and is documented in `PROTOCOL_DEVIATION.json`, the results doc, `MODELS.md`, the handoff and the status file; the non-determinism warning and the 4-epoch evidence limit are stated | `ckpt.seed*.cache_ram_recorded_in_args`, `deviation.*` |
| 5 | Checkpoint paths (`/workspace/runs/boreal_v2_try3/.../seed0`, `boreal_v2_try4/.../seed{1,2}`) and that the harness hashed exactly those files when evaluating (5 cells per model, all pinned, all equal to the recorded hash) | `ckpt.seed*.volume_path`, `.evaluated_file_hash_matches` |
| 6 | 30 cells = 6 models x 5 datasets; dataset pins and settings (imgsz 640, conf 0.001, NMS 0.7, max_det 300, report 0.25) match `eval/datasets_boreal_v2.yaml`; Model A = the existing pinned checkpoints | `cells.*` |
| 7 | **Detection metrics reproduce exactly offline** from the saved predictions and ground truth: 678 values (incl. all 18 clip-cluster bootstrap interval sets) with 0 differences; 40-107 localisation statistics differ at the 1e-7..4e-6 relative level (NumPy version drift: pod 2.1.2, sandbox 2.5.3) | `metrics.detection_metrics_recomputed_exact`, `results/recompute_check.json` |
| 8 | Frozen split hashes (train `5b23b199...`, val `8ef9e902...`, test `9fa6dbc1...`), full coverage of the 4,954 scanned files exactly once, scan pinned (`8a4db032...`), harness views pinned | `split.*` |
| 9 | Split is video-disjoint (31 clips, all intersections empty) and has 0 exact (MD5) duplicates across partitions, recomputed from per-file records | `split.video_disjoint_recomputed`, `split.no_exact_duplicates_across_partitions` |
| 10 | Near-duplicate findings reproduce: 22 train/test frame pairs within 4 dHash bits, 6 distinct test frames, 2 clip pairs (`heinola_dji_0028`/`0048`, `0048`/`ruokolahti_dji_0089`); dropping those 6 frames changes no model's mAP50 by more than 0.0007; `karkkila_dji_0007`/`0008` are both in val | `leakage.*` |
| 11 | Headline means (0.512 / 0.612 / 0.899) and the hand-typed negative-scene counts (A 9/11/17 of 65, B 1/1/1) match the cells | `docs.*` |
| 12 | All six pre-existing checkpoints (Model A seeds 0-2, Model B v1 seeds 0-2) were re-hashed on the pod after the run: unchanged | `v1.existing_checkpoints_unchanged_after_run` |
| 13 | v1 and historical Level 2 v2 files are byte-identical to `main` (0 changed among 9 protected paths); `MODELS.md` was only appended (+14/-0) | `preservation.*` |
| 14 | Checkpoint selection epochs 86 / 66 / 23 are the same under argmax val mAP50-95 and argmax Ultralytics fitness (0.1 mAP50 + 0.9 mAP50-95) | computed from the three `results.csv` files |
| 15 | Pod-side facts: the pod-injected key gets HTTP 403 on REST stop; GraphQL `podStop` stopped the pods (attempt 3's orchestration record shows the 403; the pods read EXITED) | `results/training/seed0_try3/orch_events.jsonl`, RunPod pod records |
| 16 | Seed 1's first 4 epochs trained on two different pods are identical (44 values, max abs difference 0) | `results/training/aborted_attempts/try3_seed1` vs `seed1_try4` |

## B. INFERRED (reasoned from evidence, not directly measured)
| # | Claim | Why only inferred |
|---|---|---|
| 1 | The three checkpoint files are still present and unchanged on the volume | Last hashed 2026-10-29 (harness at evaluation; seed 0 also by the diagnostic pod). Nothing is known to have touched them, but they were not re-read since; the planned backup will re-verify them |
| 2 | `cache=ram` did not change what the model learned | Cached and uncached pixels are byte-identical on 24/24 probe and 16/16 in-job images, and 4 epochs matched across pods; full-run equivalence and GPU/threading determinism are untested, and Ultralytics warns about non-determinism |
| 3 | The 22 near-duplicate pairs and the same-site consecutive clips are consecutive segments of the same recordings | dHash (9x8 grayscale) similarity plus same site/numbering; no pixel-level confirmation |
| 4 | Seed 0 trained on a busier host than seeds 1-2, and host speed variance (43-150 s/epoch) is CPU contention | Load averages and decode timings from the diagnostic jobs; training results do not depend on it |
| 5 | Model A's weak recall on non-edge-touching Boreal smoke is related to the documented synthetic edge-touch mismatch | A hypothesis: correlation in the table, no ablation |
| 6 | GPU spend about $2.93 | Pod start/stop timestamps x $0.89/h; RunPod billing (now $12.35 total) lags and mixes earlier spend |
| 7 | The clip-clustered intervals are only a rough guide | Percentile bootstrap over 8 clusters is crude; stated as such |

## C. UNKNOWN (not established by any artifact)
- Pixel-level identity of the near-duplicate frames and of consecutive same-site clips; whether `karkkila_dji_0007`/`0008` are one recording (they are kept together in val).
- Provenance, licence and the original authors' split rationale of the Boreal dataset (unchanged from v1 documentation).
- Whether the 5 unlabeled non-Empty frames (all in train) are unlabeled smoke.
- Whether a different clip-disjoint assignment (several near-equal ones exist) would change the conclusions: not run.
- Bit-for-bit reproducibility of a full 100-epoch training run on a fresh pod: not tested.
- CPU inference latency of the checkpoints on the Ubuntu VM; whether RunPod's S3 API supports US-IL-1 (`docs/handoffs/checkpoint_backup.md`).
- Real-domain cell-level truth (does not exist; `level3_observation_contract.md`).

## D. Issues found by this audit (none changes a reported number)
1. **Docs wording:** the results table called the epoch "best epoch (val mAP50-95)" while `best.pt` is chosen by Ultralytics fitness; the epochs coincide for all three seeds, wording now precise (`boreal_v2_results.md` 5.2).
2. `results/level2/matrix.md` (harness output) labels rows only by train domain, so seeds are distinguished by row order; the seed-labelled tables are in `results/report.md` and `seed_aggregation.md`.
3. Provenance of pod inference: the pods ran a bundled copy of the repo from local, unpushed commits (hashes recorded); after the push authorised on 2026-11-02 those commits exist on `science/boreal-v2`.
4. The audit script's checkpoint "volume path" check verifies the recorded path, not the live file (see B1).

## E. Preservation statement
Boreal v1 (`real_boreal_v1.md`, `level2_v1*`, `eval/datasets.yaml`, the six pre-existing checkpoints) and the historical Level 2 v2 baseline (`level2_v2*`, `level2_v2_audit*`) are unmodified (check 13). No result was re-run to improve a metric; all numbers in the reports are those the pods produced on 2026-10-29.
