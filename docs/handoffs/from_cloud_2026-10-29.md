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
