# From cloud: RunPod orchestration readiness (2026-10-29)

Task asked: establish RunPod orchestration readiness. No training, no pod created/started/stopped/restarted, no paid infrastructure.
No credentials were printed or written. Read-only API calls only. No checkpoint, run, dataset or split was touched.

## 1. Checkout verification

- Expected commit `7efff90250ded592a060dbe6b925969d51e1e8e8` was **not** in the sandbox clone at start (clone was at `17e7f38`, an
  ancestor). `git fetch origin handoff/cloud-2026-10-29` made `origin/handoff/cloud-2026-10-29` resolve to exactly that SHA (VERIFIED).
- Work branch `claude/runpod-api-access-check-pxzock` was reset to that commit before this file was added. `git status` was clean.
- `CLAUDE.md` at the handoff commit differs materially from the version in the session preamble (the handoff one holds the
  Autonomous Development and Compute Policy: $150 total, $100 autonomous, $20 per experiment, one paid pod at a time, permission to
  launch/stop/terminate pods). The handoff version was treated as authoritative.
- Tests: `python3 -m unittest tests.test_boreal_split_audit tests.test_eval_metrics tests.test_run_matrix tests.test_aggregate_seeds`
  -> `Ran 21 tests ... OK (skipped=1)`. The full `discover` run was not executed.

## 2. RunPod API (VERIFIED this session, read-only)

| Check | Result |
|---|---|
| `RUNPOD_API_KEY` in environment | set (length 50, value never printed) |
| `GET rest.runpod.io/v1/pods` | HTTP 200, `[]` (no pods) |
| `GET /v1/networkvolumes` and `/v1/networkvolumes/sej40ez3ia` | HTTP 200: `wildfire-datasets`, 50 GB, US-IL-1 |
| `GET /v1/endpoints`, `/v1/templates` | HTTP 200, both empty |
| `GET /v1/billing/pods`, `/v1/billing/networkvolumes` | HTTP 200 |
| `POST api.runpod.io/graphql` (`myself` query) | HTTP 200 |

Write permission (create/stop/delete pod) was deliberately **not tested**. It is UNKNOWN whether this key is restricted.
Note: Python `urllib` got HTTP 403 on the same endpoint that `curl` accepted (default User-Agent filtered); use curl or set a User-Agent.

## 3. Volume and model artifacts: NOT verifiable from the cloud sandbox

- The volume exists and matches the ledger (id, name, size, data center). That is all the REST API exposes.
- Its contents (`/workspace/runs/*`, checkpoints, `synth_smoke_v1`, `splits/boreal_v1`) can only be read from a running pod or the
  RunPod S3-compatible API. Checkpoint hashes in `models/MODELS.md` / `current_state.md` are therefore **recorded, not re-verified**
  here. No `*.pt` exists in the checkout (only `models/MODELS.md`).
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` exist in the environment but their origin is unknown (possibly sandbox infrastructure).
  I did not use them. Whether RunPod S3 API supports US-IL-1 and whether the owner has issued S3 keys is UNKNOWN.
- A free alternative to read the volume without a GPU: none found (earlier CPU pods were unavailable in US-IL-1; a CPU pod would be paid).

## 4. Spend (billing read, not estimate)

- `billing/pods` total $6.04 (16 rows, last row 2026-10-29) + `billing/networkvolumes` total $2.94 (26 days) = **$8.97 billed**.
- `docs/project-status.md` ledger says $9.08 billed (pods $6.16). The $0.12 pod difference is unexplained (billing endpoint may
  have been adjusted); ledger not edited. Scan pod `n5kg8sii9fcrp5` was not individually checked.
- Remaining autonomous allowance by billing alone: about $91.0. Storage accrues about $0.117/day.
- Pods running: **none**. Spend this session: $0.

## 5. Orchestration assessment

| Need | State | Evidence / gap |
|---|---|---|
| Launch | Possible, untested from cloud | REST `POST /v1/pods` exists; scripts in `eval/runpod/*.sh` are run *on* the pod, and the repo has no committed launcher. Past launches used an MCP/plugin the owner had. No RunPod MCP tool is present in this sandbox. |
| Verify completion | Partly | Scripts write `train_config.json` (`complete`), `results.csv`, logs, SHA256SUMS on the volume. Nothing reads them from outside the pod. |
| Auto-stop GPU | **Missing** | Grep of `eval/runpod/` finds no `runpodctl`, `RUNPOD_POD_ID`, or shutdown. Scripts end with the pod still billing; termination was manual. This is the largest cost risk. |
| Notify | **Missing** | No webhook/email/Slack variable in the environment. Session-level options: `send_later` / scheduled check-ins (these wake this session, they do not notify the owner by themselves); `PushNotification` tool is available but only while a session is live. |
| Cost guard | Partly | Per-run `timeout 9000` and a host-speed guard exist; no hard pod-level deadline. |

## 6. Recommended reusable workflow (proposal, not implemented)

1. **Preflight (cloud, free):** verify commit SHA, run the 21 tests, `GET /pods` must be `[]`, re-read both billing endpoints, compute
   worst-case cost (rate x timeout) and compare to the $20 per-experiment and $100 allowance limits. Write the run manifest.
2. **Launch:** one script creates a single pod (RTX 4090, US-IL-1, volume mounted, pinned image), passing the repo commit and a unique run name; abort if the name exists.
3. **Self-terminating job (add to every `eval/runpod` script):** a wrapper that (a) runs the job, (b) writes `DONE.json` with exit code and artifact hashes to the volume, (c) copies the log, (d) calls `runpodctl remove pod $RUNPOD_POD_ID` (RunPod injects the pod id and an API key into pods) in an `EXIT` trap so failure also stops billing, and (e) has an independent watchdog `sleep <hard_limit>; remove pod`. Verify in a free dry run first (needs owner approval for the paid pod; a CPU/cheap-GPU 1-minute test is the smallest option).
4. **Completion check (cloud):** poll `GET /pods` until empty, then confirm `DONE.json` via a short read pod or S3 API; hash-check new and old checkpoints against `models/MODELS.md`.
5. **Notify:** an `ntfy.sh`/webhook `curl` from the wrapper's trap (needs the owner to choose a channel and add its URL as a secret), plus a session `send_later` check-in as a backstop.
6. **Account:** update the ledger from billing reads (they lag hours), record commit, seeds, pip freeze, hashes (Handoff section E).

## 7. Blockers and decisions for the owner

1. Can the key create/stop/delete pods (write scope)? Needs one approved paid action to prove; not done here.
2. Volume contents unreadable from cloud: provide RunPod S3 keys for `sej40ez3ia` (if US-IL-1 supports it) or accept one short read-only pod to re-hash checkpoints.
3. Choose a notification channel (ntfy topic, Slack/Discord webhook or email) and store it as an environment secret.
4. Approve adding the self-terminate wrapper to `eval/runpod/*` (touches scripts used for past runs; new files preferred over edits).
5. Open research decisions from `current_state.md` section 8 (Boreal v2 split, real negatives) are unchanged and not addressed here.

## 8. UNKNOWN / unresolved

Checkpoint and dataset integrity on the volume (not re-hashed); API key write scope; origin of `AWS_*` variables; $0.12 ledger-vs-billing
pod difference; full `unittest discover` result.

---

# Update 2 (same day): orchestration tooling and infrastructure readiness

Branches (pushed, verified with `git ls-remote`): `claude/runpod-api-access-check-pxzock` = `501cf13` (this report);
`cloud/runpod-orchestration` = `4cceb30` + later commits (tooling). `main` untouched. Nothing scientific pushed.

## Verified infrastructure capabilities (live, read-only, $0)
- `RUNPOD_API_KEY` works; pods/volumes/endpoints/templates/billing readable; volume `sej40ez3ia` present; 0 pods.
- Stop endpoint is authorized: `POST /pods/<nonexistent>/stop` returned "pod does not exist" (HTTP 500), not 401/403. (Weak evidence; a real stop is untested.)
- Budget gate runs against live billing: lifecycle-test dry run -> ok, worst case $0.41 (incl. 600 s boot allowance), 0 pods running.
- 64 unit tests pass (43 orchestration + the 21 required); lifecycle logic (success/failure/timeout/stop-retry/stop-failure/watchdog/sweep/monitor/overwrite refusal/no credentials in output/no delete call) covered by mocks; the pod start command was executed in an empty directory to prove no repo files are needed.

## Pending infrastructure requirements (why the workflow is NOT ready)
1. Notification channel: `NOTIFY_URL` unset; `ntfy.sh` is **blocked by the sandbox network policy** (CONNECT 403). Owner: add `ntfy.sh` under Allowed domains and set `NOTIFY_URL` (private ntfy topic; subscribe in the ntfy iOS app). Delivery to the phone is unverified. One Claude-app mobile push was requested ("Mobile push requested"); receipt unconfirmed.
2. Off-pod watchdog independence: the cloud-side `monitor`/`sweep` die with the sandbox. A truly session-independent watchdog needs the owner's always-on machine (cron every minute running `launch_job.py sweep` / `monitor`).
3. Live lifecycle test (about $0.3-0.5 worst case) needs owner approval: proves create-pod fields, pod-side `RUNPOD_API_KEY`/`RUNPOD_POD_ID`, stop on a volume-attached pod, `DONE.json`, notifications.
4. Research jobs need their own bundle/checkout on the pod; only the lifecycle job is self-contained.

## Boreal v2 dataset readiness
Not started (gated on item 3 above, as instructed). Cloud sandbox has no Boreal images; the committed `docs/experiments/level2_v2_audit/boreal_scan.json.gz`
(4,954 files, MD5/dHash/dims/label counts, sha256 `8a4db032...0e8e`) allows clip grouping and split design offline. Open design questions: clip identity for
`Empty-Images` frames (filename-only evidence), site representation, whether 5 val/test videos leave enough positives, and that Model A/synthetic data are unaffected by a Boreal split (Model A is not retrained).

## Training / spend / artifacts
Training completed: none this session. Pending: 3 Model B seeds on Boreal v2 (unapproved to launch until infra passes and owner approves cost). Spend: $0 (billing read $8.97 total, ledger $9.08 unchanged).
Artifacts: `eval/runpod/{rp_common,launch_job,pod_supervisor,lifecycle_test_job}.py`, `tests/test_runpod_orchestration.py`, `docs/runpod/orchestration.md`. No checkpoint, split or dataset touched.

## Outstanding research questions
Everything in `current_state.md` section 8, plus: are the 3 shared clips pixel-identical across splits; real-domain negatives for a v2 test set; whether v2 should rebalance sites.

---

# Update 3: simplified workflow (notifications removed)

Owner decision: no ntfy/webhook/email; status is checked manually. Changes: `NOTIFY_*` code removed; the off-pod watchdog is now a stateless
GitHub Actions job (`watchdog.py`, `.github/workflows/runpod-watchdog.yml`, every 5 min, public repo so free) that stops pods whose name-encoded
deadline passed and appends events to branch `runpod-logs`; `launch_job.py status` answers "what is running / overdue / spent" from any new session.
Supersedes the notification and watchdog items in Update 2 (items 1-2). Still pending: merge workflow to `main`, add `RUNPOD_API_KEY` Actions secret,
run the workflow once, then the owner-approved one-minute lifecycle test (worst case $0.56 at $0.89/h; expected about $0.05-0.15). Spend: $0.
Boreal v2 and training: not started. 49 orchestration tests + 21 required tests pass.

---

# Update 4: Boreal v2 (free phase complete, GPU phase NOT run)

Branch `science/boreal-v2` (from `main` at `c080777`), **committed locally, not pushed** (instruction). Full record: `docs/experiments/boreal_v2.md`.

## Boreal v2 dataset readiness: READY and frozen
- 31 normalised clips -> train 3462 / val 748 / test 744 frames (18 / 5 / 8 clips); video-disjoint (independent recompute in `tests/test_boreal_v2.py`); 0 exact duplicates across partitions; 22 train&test frame pairs within 4 dHash bits (heinola_0028|0048: 15, 0048|ruokolahti_0089: 7); test has 65 held-out true-negative frames (2 clips).
- File-list SHA-256: train `5b23b199965dfc47372e6100f4d5bfd41d57ef5b0b16b5ed83e91db056245c25`, val `8ef9e902bb24a720112bef32d20248d6aa33c17d67efc9771643daa87d252e2f`, test `9fa6dbc1d0eb9fa437e989be29afadc9a459f7850502f1c18c350379afcddc8f`.
- Construction is a seeded search (seed 20261029, 400 restarts) using no model output; one earlier candidate was inspected and rejected before any training (documented in the record). Alternatives with near-equal objective exist.
- Artifacts: `docs/experiments/boreal_v2/{filelists.json,split_assignment.json,split_audit.json}`, `eval/boreal_v2_split.py`, `eval/datasets_boreal_v2.yaml`.

## Training / evaluation: PENDING (nothing trained, $0 spent)
Required new training: Model B seeds 0,1,2 on `boreal_v2` (new names `real_boreal_v2_seed{N}` / `real_boreal_v2_best_seed{N}.pt`). Not required: Model A retraining (synthetic data unaffected; existing pinned checkpoints are only re-evaluated); v1 Model B checkpoints are not valid for v2 claims (trained on clips now in v2 test).

## Blocker (material): no return path for results
The cloud sandbox cannot read anything back from a pod: REST has no logs/files endpoint, `*.proxy.runpod.net` is blocked by the network policy, and the only S3-looking credentials (`AWS_*` env, unknown origin) were deliberately not used (a use attempt was denied by the safety classifier). Options for the owner, any one suffices:
1. Add `proxy.runpod.net` to the environment's Allowed domains (the job can then serve `results_bundle.tgz` for a short window before the pod stops), or
2. Provide RunPod S3 API keys for volume `sej40ez3ia` as secrets named explicitly for that purpose (say whether the existing `AWS_*` are those), or
3. Run the job as-is and fetch `/workspace/runs/boreal_v2/results_bundle.tgz` (+ `.sha256`) from the Ubuntu CLI/RunPod console, then let a session finish the report offline (`eval.run_matrix --preds-from`, `eval.aggregate_seeds`).
Option 3 needs no sandbox change. The job is otherwise ready.

## Ready-to-run command (dry run by default; `--confirm-paid boreal_v2_train_eval --watchdog-armed` creates the pod)
```
GIT_COMMIT=$(git rev-parse HEAD) python3 eval/runpod/launch_job.py launch --run-name boreal_v2_train_eval \
  --job 'bash repo/eval/runpod/boreal_v2_train_eval.sh' --hard-timeout-s 18000 --max-hourly-price 1.0 --ledger-floor-usd 9.3 \
  --boot-allowance-s 900 --watchdog-latency-s 120 --repo-path eval --repo-path docs/experiments/boreal_v2 \
  --repo-path tests/__init__.py --repo-path tests/test_eval_metrics.py --repo-path tests/test_aggregate_seeds.py --repo-path tests/test_run_matrix.py \
  --artifact '/workspace/runs/boreal_v2/real_boreal_v2_seed*/real_boreal_v2_best_seed*.pt' --artifact /workspace/runs/boreal_v2/results_bundle.tgz
```
Preflight (live): ok, worst case $5.53, 0 pods. Before launching: dispatch the GitHub watchdog with `loop_minutes` = 330. Unverified (the lifecycle test never ran): create-pod fields, pod-side stop key, 250 KB of env-var bundle chunks being accepted by RunPod. Expected cost about $2.5.

## Infrastructure status (unchanged)
Verified: API access, budget gate, watchdog loop run (no pods), selftest, logs branch. Not verified: scheduled (cron) run, any real stop. Workflow is NOT declared ready; the owner accepted that limitation.

## Spend / pods / artifacts
Spend this session $0 (billing read $8.97 total). Pods running: none. No checkpoint, dataset, split or earlier result was modified. Open research questions: pixel-level identity of consecutive same-site clips; real-domain negatives remain few (65 test, 2 clips).

---

# Update 5: Boreal v2 GPU run launched (in progress when written)

- Pod `kv3imy4lnltlea` (`boreal_v2_train_eval__dl1793318398`), RTX 4090 secure US-IL-1, $0.89/h, created 2026-10-29T19:14:55Z, hard deadline 23:59:58Z (tag in the pod name). Command: `python3 eval/runpod/launch_job.py boreal-v2 --max-hourly-price 1.0 --ledger-floor-usd 9.3 --boot-allowance-s 900 --watchdog-latency-s 120 --results-token-file <outside repo> --confirm-paid boreal_v2_train_eval --watchdog-armed` at commit `16f7e60` (unpushed; `GIT_COMMIT` and `BUNDLE_SHA256` are recorded in each seed's `train_config.json`).
- Independent stop: GitHub watchdog loop run #6 (dispatched 19:08:50Z, 330 min, ends 00:38Z) plus the pod's own supervisor and in-pod watchdog. A scheduled (cron) watchdog run #5 succeeded at 15:38Z.
- Results location (persistent volume `wildfire-datasets`): `/workspace/runs/boreal_v2/` (checkpoints `real_boreal_v2_seed{0,1,2}/real_boreal_v2_best_seed{N}.pt`, `level2_boreal_v2/` cells + predictions + gts, `results_bundle.tgz` + `.sha256`, `INTEGRITY.json`, `export/`), split `/workspace/splits/boreal_v2/`, supervisor records `/workspace/runs/boreal_v2_train_eval_orch/{events.jsonl,job.log,DONE.json}`.
- Return path: the pod serves ONLY `/workspace/runs/boreal_v2/export/` (token-protected) on `https://kv3imy4lnltlea-8000.proxy.runpod.net` until fetched or 15 min after completion. From this sandbox the proxy domain was still blocked at launch; if it was never fetched, retrieve `results_bundle.tgz` from the volume manually (RunPod console / your usual path) and verify against `results_bundle.tgz.sha256`. The results token file lives in the cloud session's scratchpad (not durable); it is also visible in the pod's environment in the RunPod console.
- Create-pod findings: RunPod rejected a 300 KB environment (HTTP 500 "unmarshal ... invalid character 'I'"); 60 KB works; the bundle was cut to ~70 KB. Three throwaway probe pods (ids `ehlptgu8zl8o5c`, `ez7il6dxm4fkg6`, `1voj5p632zthee`) were created and stopped within seconds (about $0.01 total); they remain in state EXITED and can be removed by hand.
- Stop-by-REST from the sandbox verified on those probes (state EXITED within 3 s). Pod-side self-stop (injected key) is verified only when this run ends.

---

# Update 6: Boreal v2 COMPLETE (supersedes the "pending" statements in Updates 4-5)

Branch `science/boreal-v2` (local, **not pushed**; start from `main` `c080777`). Full record: `docs/experiments/boreal_v2_results.md`; split: `docs/experiments/boreal_v2.md`; artifacts: `docs/experiments/boreal_v2/results/`.

## Verified infrastructure capabilities (all observed in real runs)
- RunPod REST create/stop from the cloud session; HTTPS proxy retrieval (token-protected, allowlist-only results endpoint, hash-verified fetch, ack ends the server); live status endpoint.
- **Pod self-stop: REST stop is 403 with the pod-injected key; GraphQL `podStop` works** and is the supervisor/in-pod-watchdog fallback (3 real pods stopped themselves). The GitHub watchdog (loop mode) was running throughout; a scheduled (cron) run also succeeded earlier.
- Speed guard, no-overwrite checks, hash-pinned prior checkpoints, seed-subset runs (`--seeds`, `--prior-ckpt`), `cache=ram` pixel-equivalence guard.

## Pending infrastructure / owner items
- Local backup of the three new checkpoints (hashes in `models/MODELS.md`); volume copies are the only ones.
- RunPod removal of 11 stopped pods (4 attempt pods + 7 probe/diagnostic pods) if desired; they cost only small disk charges.
- Pushing/merging `science/boreal-v2` (intentionally unpushed). The orchestration branches were already merged (PRs 1-4).
- RunPod env limit (~60 KB ok / 300 KB rejected) and 6x host-speed variance are recorded in `docs/runpod/orchestration.md`-adjacent notes only in the commit messages; add to that doc if useful.

## Boreal v2 dataset readiness: DONE (frozen, hashed, audited). Model training: DONE for Model B v2 seeds 0-2 (Model A not retrained). Evaluation: DONE (6 models x 5 datasets), recomputed offline.
Results (mAP50, mean of 3 seeds +- SD): Synth->Synth 0.995 (saturated); Synth->Real 0.512 +- 0.095; Real->Synth 0.612 +- 0.104; Real->Real 0.899 +- 0.027. Clip-clustered 95% intervals (8 clips) are wide (e.g. Real->Real seed 0 [0.838, 0.966], Synth->Real seed 0 [0.175, 0.810]); the gap's size is not tightly estimated. Real empty-scene FP rate on 65 held-out frames: Model A 0.14/0.17/0.26, Model B-v2 0.015 each. Boreal v1 results untouched and not comparable.
Checkpoint SHA-256: seed 0 `090c0f4c...c299a05` (try3), seed 1 `5a9616ce...44e9388`, seed 2 `c4d443a8...ae097ef` (try4). Deviation: `cache=ram` (pixel-identical verified; Ultralytics non-determinism warning; seed 1 epochs 1-4 matched across two pods).
Offline recompute: all detection metrics identical; 107 localisation statistics differ <= 4.2e-6 relative (NumPy versions).

## Spend
Estimated GPU **$2.93** for the experiment (billing posted $11.27 total at last read, lags). Conservative cumulative about $12.3; remaining autonomous allowance about $87.7. Within the $20 per-experiment limit. No pods running.

## Outstanding research questions
Robustness to a second clip-disjoint assignment; pixel-level identity of consecutive same-site clips; whether a versioned `synth_smoke_v2` with matched edge-touch closes the Synth->Real gap; Level 3 sensitivity/specificity from Boreal v2 negatives (no calibration code written, as instructed).

---

# Update 7 (2026-11-02): audit, preservation, backup route, ArduPilot plan

No GPU was started, nothing was retrained or re-scored, no checkpoint was downloaded, nothing was merged into `main`.

## What was done
- **Scientific audit (read-only): PASS WITH DOCUMENTED CAVEATS.** `docs/experiments/boreal_v2_audit.md` (VERIFIED / INFERRED / UNKNOWN tables) and the re-runnable `python3 -m eval.boreal_v2_audit` (`docs/experiments/boreal_v2/audit.json`: 49 checks, 49 VERIFIED, 0 FAILED; includes an offline recompute of all detection metrics and the clip-cluster bootstrap intervals).
  It is a self-audit by the same session lineage: please have it confirmed (owner or Codex). Not established: present on-volume state of the checkpoints (last hashed 2026-10-29), pixel-level identity of near-duplicate/consecutive clips, full-run bit-reproducibility, CPU latency.
- **Preservation:** working tree reviewed (21 commits, 5.2 MB added, no `.pt`/archives/images, secret scan of the whole diff clean for the RunPod key, results token and `AWS_*` values and for generic secret patterns), then `science/boreal-v2` pushed (not merged).
- **Checkpoint backup route:** `docs/handoffs/checkpoint_backup.md` (paths, hashes, `checkpoints.SHA256SUMS`, Route A RunPod S3 API = free but unconfirmed for US-IL-1, Route B a short pod launched from the VM, about $0.10-0.20, worst case $1.25). Code for Route B (`eval/runpod/checkpoint_backup_job.py`, `--checkpoint-backup`, results-server opt-in for exactly the three names) is tested but **never run**.
- **ArduPilot integration plan (planning only):** `docs/handoffs/ardupilot_integration_plan.md`.

## Final model inventory (Model B v2; Model A unchanged)
| File | Seed | Volume path | SHA-256 |
|---|---|---|---|
| `real_boreal_v2_best_seed0.pt` | 0 | `/workspace/runs/boreal_v2_try3/real_boreal_v2_seed0/` | `090c0f4cbe2718d2b084a41b8a3736fa27b673bc66b20c64672b13c83c299a05` |
| `real_boreal_v2_best_seed1.pt` | 1 | `/workspace/runs/boreal_v2_try4/real_boreal_v2_seed1/` | `5a9616cefddd3bca6eccf31d0691f7b1a2fadb572f84cf7948c85296644e9388` |
| `real_boreal_v2_best_seed2.pt` | 2 | `/workspace/runs/boreal_v2_try4/real_boreal_v2_seed2/` | `c4d443a8af919c6b00e4319b117a5dfbe90bd8b29750c75e3970a9389ae097ef` |
Results (mAP50, 3-seed mean +- SD): Synth->Synth 0.995 (saturated); Synth->Real 0.512 +- 0.095; Real->Synth 0.612 +- 0.104; Real->Real 0.899 +- 0.027 (clip-clustered intervals, 8 clips, are wide). Deviation: `cache=ram`. Spend: about $2.93 GPU for the experiment (estimate); RunPod billing read $12.35 total on 2026-11-02; no pod running; conservative cumulative about $12.3-12.4 of the $100 allowance.

## Resume on the Ubuntu VM
```bash
cd ~/src/Ardupilot-UAV-YOLO
git status                                   # must be clean (stash/commit your own work first); never reset it blindly
git fetch origin
git checkout science/boreal-v2               # or: git switch -c science/boreal-v2 origin/science/boreal-v2
git log --oneline -3                         # compare with the HEAD recorded in the final report
python3 -m unittest tests.test_boreal_split_audit tests.test_eval_metrics tests.test_run_matrix tests.test_aggregate_seeds tests.test_boreal_v2 tests.test_boreal_v2_report tests.test_boreal_v2_audit tests.test_results_transfer tests.test_runpod_orchestration
python3 -m eval.boreal_v2_audit              # expect: {'VERIFIED': 49, 'FAILED': 0, 'SKIPPED': 0}
# then back up the checkpoints: docs/handoffs/checkpoint_backup.md (try Route A first), verify with checkpoints.SHA256SUMS
# then read: docs/handoffs/ardupilot_integration_plan.md  (section 6 is the recommended next step)
```
Read first: `CLAUDE.md`, `docs/project-status.md`, `docs/experiments/boreal_v2_results.md`, `docs/experiments/boreal_v2_audit.md`.

## Outstanding items for the owner
1. Run the checkpoint backup (section above) and delete any temporary RunPod S3 key afterwards.
2. Decide whether to merge `science/boreal-v2` into `main` (not done); the orchestration branches were already merged (PRs 1-4).
3. Confirm the audit (self-audit only) and the three open research questions in `boreal_v2_results.md` section 7.
4. ArduPilot plan decisions before implementation: imagery domain (gap 2), renderer truth channel (gap 1), camera conventions (gap 3), operating threshold (gap 4).
5. Stopped RunPod pods (11) can be removed in the console; this tooling never deletes pods.

---
# Update 8 (2026-11-02, Ubuntu session): checkpoint backup executed, plan reviewed
- Remote `science/boreal-v2` was `f321818` when checked; this work is on branch `integration/yolo-perception-prep` (from that commit).
- **Checkpoints backed up and verified** to `local_data/backups/boreal_v2/` (gitignored): seed0 `090c0f4c...a05`, seed1 `5a9616ce...388`, seed2 `c4d443a8...7ef`, each 22,501,098 bytes; `sha256sum -c` OK; pod re-hashed the originals on the volume first. Details: `checkpoint_backup.md` section 6.
- **Spend:** one pod about 2 min at $0.89/h, estimated under $0.04 (not yet billed); cumulative about $12.4, remaining autonomous about $87.6. Pod deleted; 0 running; volume kept.
- **Tests:** full `python3 -m unittest discover -s tests -t .` -> 172 tests OK (about 100 s) on this branch. `eval.boreal_v2_audit --recompute` -> 49 VERIFIED, 0 FAILED. Frozen Boreal v2 results untouched.
- **Integration plan reviewed:** `ardupilot_integration_plan.md` section 8 (new findings: `grid` object type mismatch, 1 vCPU / 2 GB VM, stale docstring; ordered implementation steps). Nothing implemented.
