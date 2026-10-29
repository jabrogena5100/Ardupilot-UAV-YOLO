# Handoff to Claude Code Cloud (reusable, bidirectional)

Read `docs/handoffs/current_state.md` first, then `CLAUDE.md`, `docs/project-status.md`, `models/MODELS.md`. If anything here
disagrees with the repo, the repo wins; fix the doc in the same commit. Do not start any job until the owner has stated the task
for this session: this file is a standing brief, not an instruction to train.

## A. What a cloud session can and cannot see

- Can: the git checkout of this branch; committed experiment artifacts; unit tests (numpy/PyYAML/Pillow level).
- Cannot (unless the owner wires it up): the RunPod volume `/workspace`, `local_data/`, checkpoints (`*.pt`), datasets, or the owner's
  uncommitted work. Do not assume paths from `current_state.md` section 5 exist in the sandbox.
- RunPod access, if wanted, is through the RunPod MCP/plugin configured by the owner. Credentials come from that configuration or the
  environment only. **Never write an API key, token or SSH key into a file, log, commit message or chat.** Never print the environment.
- Never push to `main`. Work on a branch; push only when asked.

## B. Model B training prerequisites (checklist, all must hold before any GPU is paid for)

Model B is `real_boreal_*`: YOLOv8s on Boreal. Existing checkpoints (seeds 0, 1, 2) must never be overwritten or deleted.

1. Written owner approval naming the run: new seed(s) or a Boreal v2 split. Retraining the existing seeds is not requested.
2. Run names that do not exist yet. The scripts abort if the run directory exists; keep that guard. Use `real_boreal_seed<N>` for new
   seeds or a new `real_boreal_v2_*` name. Check `ls /workspace/runs` first.
3. Data present and hash-verified on the volume:
   - `/workspace/splits/boreal_v1`: the per-split file-list SHA-256 values in `manifest.json` must match (the preflight in
     `eval/runpod/boreal_stage2_train_eval.sh` does this; it exits 3 on mismatch).
   - If a Boreal v2 split is used, it is a new directory with its own manifest and its video memberships frozen and published first.
     Build it from `eval/boreal_split_audit.py` (video identity = filename minus site prefix and `_frameN`, case-folded). Do not edit `boreal_v1`.
4. Recipe pinned (reuse `eval/runpod/level2_seeds_train_and_matrix.sh` settings): `yolov8s.pt`, 100 epochs, `patience=100`, imgsz 640,
   batch 32, `deterministic=True`, workers 8, `optimizer=AdamW lr0=0.002 momentum=0.9 warmup_bias_lr=0.0`, `seed=<N>`.
   Software pinned: `ultralytics==8.4.173`, torch 2.8.0+cu128 (record `pip freeze`). Change nothing else without approval.
5. Repo state: run from a git checkout at a recorded commit (the earlier pod used a non-git copy, which left `git.commit: null`).
   Record `git rev-parse HEAD` in the run manifest.
6. Pod: 1x RTX 4090 in US-IL-1 (the volume's data center) worked; A40 and RTX 4000 Ada were out of stock there, and no CPU pods were
   available there. Check capacity and the hourly price first. Dataloader is CPU-bound (4096x2160 Boreal frames): keep the host-speed
   guard (abort if mean epoch 2-4 exceeds the per-run limit; Model B limit was 60 s/epoch). Slow hosts cost money, so re-create the pod.
7. Expected cost at the 4090 rate used so far ($0.89/h): about $0.9 per Model B seed (about 2,200 s train plus setup/eval). Per-run
   timeout in the scripts is 9,000 s.

## C. Budget limits (from `CLAUDE.md`; the owner can only raise them)

- $150 project total; $100 cumulative autonomous; $50 reserve; $20 maximum per experiment; one paid GPU pod at a time.
- Ledger in `docs/project-status.md`: about $9.3 spent, about $90.7 autonomous allowance remaining at the snapshot. Re-read
  `list-billing` / `list-pod-billing` / `list-network-volume-billing` before launching (billing lags) and update the ledger after every paid action.
- State the hourly price and a worst-case cost before creating a pod. Stop and report if the estimate would exceed $20 or the allowance.
- Verify the job finished and artifacts are backed up before deleting a pod. Terminate idle pods; confirm `list-pods` is empty afterwards.
- Storage keeps accruing (about $0.12/day) while the volume exists. Do not delete the volume.

## D. Required tests (run before every commit; report actual output, never claim a pass unseen)

```bash
python3 -m unittest tests.test_boreal_split_audit tests.test_eval_metrics tests.test_run_matrix tests.test_aggregate_seeds
python3 -m unittest discover -s tests -t .      # about 2 min (renders images); run in the background
```
Last run before this snapshot: the 21 tests in the first command passed; the full `discover` was **not** run (UNKNOWN result).
After a training run: re-hash the new and all pre-existing checkpoints against `models/MODELS.md`, confirm the dataset hashes still
match, and confirm no pre-existing run directory changed.

## E. Acceptance criteria for a Model B training run

A run is accepted only if all hold; otherwise report FAILED/INCOMPLETE and do not average it in.
1. Exit code 0, 100 epochs in `results.csv`, `complete: true` in `train_config.json`, checkpoint copied to a new unique filename.
2. Checkpoint SHA-256 recorded in `models/MODELS.md` (new row; no existing row edited except to add a correction note).
3. `args.yaml` matches section B.4 and `train_config.json` holds the full environment and `pip freeze`.
4. Frozen split and test set hashes match before and after; test set never used for checkpoint selection (best.pt by own-domain val).
5. Evaluation only through `eval/run_matrix.py` with `eval/datasets.yaml` (four cells, same code path, `conf=0.001`, `nms_iou=0.7`,
   `max_det=300`, report at 0.25). Predictions and ground truth saved; offline recompute from them agrees with the pod cells.
6. Reported as mean +- SD over seeds with bootstrap CIs separate (video-cluster for Boreal; texture-slot cluster for synthetic test).
   Any Boreal result states the shared-clip finding and, for a v2 split, shows the memberships are disjoint (pairwise intersections all 0).
7. Spend recorded in the ledger; `list-pods` empty; artifacts backed up (checkpoint, logs, `SHA256SUMS`) before pod deletion.
8. Docs updated: `docs/project-status.md` (state, ledger, training log), `models/MODELS.md`, an `docs/experiments/` report. Commit with
   specific file staging, no push unless asked.

## F. Hard prohibitions

No overwriting or deleting checkpoints, runs, datasets, splits or logs. No editing frozen test sets or `boreal_v1`/`synth_smoke_v1`.
No fabricated numbers. No `AGENTS.md` edits. No probability-map implementation or `PriorityPlanner.score()` change before the Level 3
contract quantities are measured and reviewed. No expansion beyond the Scope list in `CLAUDE.md`.

## G. Return path (cloud -> owner/laptop)

When finishing, commit to your own branch and write `docs/handoffs/from_cloud_<YYYY-MM-DD>.md` containing: task asked; commit SHA(s);
tests run with real output summaries; artifacts created and where (volume paths, hashes); spend (billing read, not estimate) and
remaining allowance; pods left running (should be none); checkpoints needing a local backup (the laptop `local_data/backups/` copy
is the owner's step); anything UNKNOWN or unresolved. Then update `current_state.md` only for facts you verified. Do not push to `main`.
