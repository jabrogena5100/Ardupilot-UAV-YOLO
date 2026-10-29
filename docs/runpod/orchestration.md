# RunPod job orchestration (`eval/runpod/`)

Status: implemented and tested **with mocks only**. No paid pod has been created by this code.

| File | Runs where | Role |
|---|---|---|
| `rp_common.py` | both | REST client (no delete method), budget check, JSONL event log, no-overwrite JSON writer, best-effort notify |
| `launch_job.py` | cloud / laptop | `preflight`, `launch` (dry run unless `--confirm-paid RUN_NAME`), `monitor` (external watchdog) |
| `pod_supervisor.py` | on the pod | runs the job, writes `DONE.json`, notifies, **stops** the pod, spawns an independent in-pod watchdog |
| `tests/test_runpod_orchestration.py` | cloud / laptop | 30 mock tests (fake API, local HTTP server, real subprocess jobs) |

## Guarantees (each covered by a test)
- Pod is stopped (never deleted) after success, failure, job exception, missing command or timeout. Stop is retried and verified via `GET /pods/{id}`; `STOP_FAILED` is logged and notified if it cannot be confirmed.
- Independent watchdog process (own session) stops the pod at the hard deadline even if the supervisor hangs; the job is killed `--finish-margin-s` (default 300 s) earlier so `DONE.json` is written first. `launch_job.py monitor` is a third, off-pod layer.
- Budget gate before creating a pod: one running pod max, worst case (`max price x hard timeout + $0.25`) <= $20, and `max(billed, ledger floor) + worst case` <= $100. Pod price above `--max-hourly-price` after creation => immediate stop.
- Result dir `<mount>/runs/<run>_orch` must not exist; `DONE.json` is never overwritten; artifact files are only read (SHA-256), never written.
- Keys are read from the environment, never logged or put in errors/results; `NOTIFY_URL` is redacted in printed specs.

## Workflow
1. `python3 eval/runpod/launch_job.py preflight --run-name R --job 'bash ...' --hard-timeout-s N --max-hourly-price P --ledger-floor-usd <ledger>` (read-only).
2. Owner approves the named run, then `launch ... --confirm-paid R`.
3. Optional off-pod safety net: `launch_job.py monitor --pod-id ID --deadline-epoch T`.
4. After the pod reads stopped: read `DONE.json`, re-hash checkpoints against `models/MODELS.md`, update the ledger from billing.

## Not verified yet (do in the controlled one-minute test)
- That RunPod injects `RUNPOD_POD_ID` and a key into the pod that is allowed to call `POST /pods/{id}/stop` (if not, pass a scoped key via the launcher env; do not use the full account key).
- REST create-pod field names/values (`gpuTypeIds`, `dataCenterIds`, `dockerStartCmd`, image tag) and that `POST /pods/{id}/stop` works on a pod with a network volume.
- The repo checkout expected at `--repo-dir` (default `/workspace/repo_orch`) on the volume; nothing uploads it.
- Stopped pods: GPU billing ends, but container disk keeps a small charge and a stopped pod may be unable to restart if its GPU is taken. Checkpoints live on the network volume, which is unaffected. Remove idle stopped pods by hand after verifying artifacts.
- Notification delivery to the owner's chosen `NOTIFY_URL` (none configured yet).
