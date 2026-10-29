# RunPod job orchestration (`eval/runpod/`)

Status (2026-10-29): implemented and tested **with mocks and a local HTTP server only**. No paid pod has been created by this
code. The workflow is **not** declared ready: the live stop and notification test (below) has not run.

| File | Runs where | Role |
|---|---|---|
| `rp_common.py` | both | REST client (no delete method), budget gate, JSONL event log, no-overwrite JSON writer, notify (ntfy plain text or JSON) |
| `launch_job.py` | cloud / laptop | `check-access`, `notify-test`, `preflight`, `launch`, `lifecycle-test`, `monitor`, `sweep` |
| `pod_supervisor.py` | on the pod | runs the job, writes `DONE.json`, notifies, **stops** the pod, spawns an in-pod watchdog; refuses to start without an API key |
| `lifecycle_test_job.py` | on the pod | disposable job: writes one file under `/workspace/runs/lifecycle_test_*`, trains/evaluates nothing, refuses any other path |
| `tests/test_runpod_orchestration.py` | cloud / laptop | 43 mock tests |

## Fresh pod: no files needed on the volume
`launch_job.py` ships `rp_common.py`, `pod_supervisor.py` (and any `--bundle-file` from `eval/runpod/`) inside the pod start
command as a base64 tar.gz in `$ORCH_BUNDLE_B64` (about 8 KB). The start command unpacks it to a temp dir and runs it. A test runs
that exact command in an empty directory. Needs only `bash`, `base64`, `tar`, `python3` in the image. Research jobs that need the
repo or data must bring their own bundle/checkout; that is not solved here.

## Stop layers (a pod must be stopped, never deleted)
1. Supervisor on the pod: stop after success, failure, exception, missing command or timeout; retried and verified by `GET`.
2. Independent in-pod watchdog process: stops the pod at the hard deadline even if the supervisor hangs.
3. **Off-pod watchdog**: `launch --watch` / `monitor`: stops the pod at `creation + boot allowance (600 s) + hard timeout`, grace 60 s, retries,
   `STOP UNCONFIRMED` urgent notification after 30 min. This bound holds even if the pod's own script never starts.
4. `sweep --name-prefix P`: stops every RUNNING pod whose name starts with P (cron / scheduled safety net).
Limit of layer 3 and 4: they live where they run. In a cloud sandbox they die with the sandbox. For a watchdog independent of any
Claude session, run `monitor`/`sweep` from the owner's always-on machine (cron, every minute) with `RUNPOD_API_KEY` in its environment.

## Budget gate (before any create)
No pod running, worst case = `price x (boot allowance + hard timeout) + $0.25` <= $20, and `max(billed, ledger floor) + worst case` <= $100.
After creation the real `costPerHr` is compared with `--max-hourly-price`; above it the pod is stopped at once.
`launch` is a dry run unless `--confirm-paid RUN_NAME` equals the run name.

## Notifications (iPhone)
`NOTIFY_URL` = a private ntfy topic URL (install the ntfy iOS app, subscribe to a long random topic), `NOTIFY_KIND=ntfy` (default) or `json`.
Sent from the pod (job result, stop failure, watchdog) and from the off-pod watchdog. `launch_job.py notify-test` sends one message.
The cloud sandbox's network policy currently **blocks ntfy.sh** (proxy 403), so the test cannot be run from here until
`ntfy.sh` is added under Allowed domains. The pod is assumed to have open egress (unverified).
Independent second channel: the Claude app's mobile push (Remote Control), usable only while a session is live.

## Workflow
1. `check-access` (read-only; the stop probe targets a nonexistent id) and `notify-test`.
2. `lifecycle-test --max-hourly-price P --ledger-floor-usd L` (dry run), then with `--confirm-paid <run name> --watch` after owner approval.
3. After the pod reads stopped: read `DONE.json` / `events.jsonl` under `/workspace/runs/<run>_orch`, check billing, update the ledger.

## Verified vs not verified
Verified (this session, read-only): API key works; GET pods/volumes/billing; volume `sej40ez3ia` present; stop endpoint is authorized (a stop
of a nonexistent pod returns "does not exist", not 401/403); budget gate against live billing; dry-run spec.
NOT verified: create-pod field names and image tag; that RunPod injects `RUNPOD_POD_ID` and a stop-capable key into the pod; that a
stop works on a pod with a network volume; notification delivery to the phone; the off-pod watchdog against a real pod.
Stopped pods: GPU billing ends, container disk keeps a small charge, and a stopped pod may be unable to restart if its GPU is taken.
Checkpoints live on the network volume, which is unaffected. Remove idle stopped pods by hand after artifacts are verified.
