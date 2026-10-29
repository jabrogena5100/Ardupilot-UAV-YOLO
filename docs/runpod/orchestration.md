# RunPod job orchestration (`eval/runpod/`)

Status (2026-10-29): implemented and tested with mocks, a local HTTP server and read-only live API calls. **No paid pod has been
created. The workflow is not declared ready**: the live lifecycle test has not run and the GitHub watchdog is not yet armed (see Setup).
Notifications were removed from the design; status is checked manually (`launch_job.py status`, or asking a Claude Code Cloud session).

| File | Runs where | Role |
|---|---|---|
| `rp_common.py` | everywhere | REST client (no delete method), budget gate, deadline-tagged pod names, append-only event log, no-overwrite JSON |
| `launch_job.py` | cloud / laptop | `check-access`, `preflight`, `launch`, `lifecycle-test`, `enforce`, `status` |
| `pod_supervisor.py` | on the pod | runs the job, writes `DONE.json` + `events.jsonl`, **stops** the pod, spawns an in-pod watchdog; refuses to run without an API key |
| `lifecycle_test_job.py` | on the pod | disposable job; writes one file under `/workspace/runs/lifecycle_test_*`; trains/evaluates nothing |
| `watchdog.py` | GitHub Actions / anywhere | stateless off-pod watchdog: stops RUNNING pods past the deadline in their name |
| `.github/workflows/runpod-watchdog.yml` | GitHub's servers | runs `watchdog.py` every 5 min, appends events to branch `runpod-logs` |

## Stop layers (pods are stopped, never deleted; volume, checkpoints, logs and `DONE.json` are untouched)
1. **Supervisor on the pod** stops it after success, failure, exception, missing command or timeout (retried, verified by `GET`).
2. **In-pod watchdog process** stops it at the hard deadline even if the supervisor hangs.
3. **Off-pod watchdog on GitHub Actions** (independent of Claude Code Cloud, the Ubuntu VM and the pod): the pod's name is
   `<run>__dl<deadline_epoch>`, deadline = creation + boot allowance (600 s) + hard timeout. Every pass stops overdue pods, retries 3 times
   per pass, and the next pass retries again. A stop that cannot be confirmed logs `STOP_FAILED` and fails the workflow run (visible in
   the Actions tab / GitHub mobile app). Untagged running pods are only reported, unless `--stop-untagged`.
   Latency: cron can lag, so the bound is deadline + about 5-15 min. It also covers a pod whose own script never started (stuck image pull).

### Loop mode (does not depend on cron)
Actions > runpod-watchdog > Run workflow with `loop_minutes` (max 330) keeps one run polling every 60 s, so a single manual dispatch before a
paid run covers the pod's whole window from GitHub's side with about 1-minute stop latency. Cron (every 5 min at minutes 2,7,...) remains the long-term renewal.
When a loop run covers the pod's deadline, pass `--watchdog-latency-s 120` to `launch_job.py`: lifecycle-test worst case becomes about $0.44 instead of $0.56.
Heartbeats are throttled to one per 10 min per pod; stops, failures and API errors are always logged. Events are appended to `runpod-logs` when the run ends.

## Records (append-only)
- On the volume, per run: `/workspace/runs/<run>_orch/{events.jsonl, job.log, DONE.json}`; `<run>_orch_watchdog.jsonl`. `DONE.json` is
  never overwritten; the run directory must not already exist. Includes `STOP_FAILED` / `SHUTDOWN_UNCONFIRMED`.
- Off-pod: `enforce_events.jsonl` on branch `runpod-logs` (each pass with a running pod adds a commit; history is the log).
- Later status from any new session: `python3 eval/runpod/launch_job.py status` (needs `RUNPOD_API_KEY`) shows pods, deadlines, overdue time,
  billed total and the tail of `runpod-logs`. Reading `DONE.json` itself needs a pod or RunPod S3 keys for the volume (not set up).

## Budget gate (before any create)
No pod running; worst case = `price x (boot 600 s + hard timeout + watchdog latency 600 s) + $0.25` <= $20; `max(billed, ledger floor) + worst case` <= $100.
Lifecycle test at $0.89/h: worst case **$0.56** (the earlier $0.41 assumed an on-time watchdog); expected actual cost about $0.05-0.15.
After creation the real `costPerHr` must be <= `--max-hourly-price` or the pod is stopped at once. `launch` is a dry run unless
`--confirm-paid RUN_NAME` equals the run name, and refuses without `--watchdog-armed` (attestation that the GitHub watchdog is live).

## Setup and verification
Merged to `main` in PR #1; the repository secret `RUNPOD_API_KEY` is set (manual run #1 passed). Security review, findings and the
step-by-step verification (manual run with `selftest`, one scheduled run) are in `docs/runpod/watchdog_security_review.md`. The watchdog is **armed only after**
those checks pass. GitHub pauses scheduled workflows after 60 days without repository activity.

## Fresh pod
No files on the volume are needed: the supervisor and job files travel in `$ORCH_BUNDLE_B64` (about 8 KB) inside the start command (tested in an
empty directory). Needs `bash`, `base64`, `tar`, `python3` in the image. Research jobs must bring their own checkout.

## Verified vs not verified
Verified: key works; reads of pods/volumes/billing; volume `sej40ez3ia` present; stop endpoint authorized (stop of a nonexistent pod ->
"does not exist"); budget gate against live billing; dry-run spec; `enforce` against the live API (no pods -> quiet, exit 0); 49 mock tests.
NOT verified: create-pod field names and image tag; pod-side `RUNPOD_POD_ID`/key injection; a real stop; the GitHub workflow running
(needs setup above); `runpod-logs` branch creation. Stopped pods keep a small container-disk charge and may not restart if their GPU is taken.
