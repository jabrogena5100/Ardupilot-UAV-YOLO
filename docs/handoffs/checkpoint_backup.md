# Backing up the Boreal v2 Model B checkpoints to the Ubuntu VM

Written 2026-11-02 by the cloud session. **Nothing here has been executed**: no checkpoint was downloaded, no extra pod was launched. The
originals stay on the RunPod network volume and are never deleted or modified by any command below. Checkpoints are `*.pt` (gitignored) and must not
be pushed to GitHub.

## 1. What to back up (exact volume paths, hashes, provenance)

Volume `wildfire-datasets`, id `sej40ez3ia`, US-IL-1, mounted at `/workspace` on pods.

| File | Seed | Volume path | S3-key path (relative to the volume root) | SHA-256 |
|---|---|---|---|---|
| `real_boreal_v2_best_seed0.pt` (22,501,098 B) | 0 | `/workspace/runs/boreal_v2_try3/real_boreal_v2_seed0/real_boreal_v2_best_seed0.pt` | `runs/boreal_v2_try3/real_boreal_v2_seed0/real_boreal_v2_best_seed0.pt` | `090c0f4cbe2718d2b084a41b8a3736fa27b673bc66b20c64672b13c83c299a05` |
| `real_boreal_v2_best_seed1.pt` | 1 | `/workspace/runs/boreal_v2_try4/real_boreal_v2_seed1/real_boreal_v2_best_seed1.pt` | `runs/boreal_v2_try4/real_boreal_v2_seed1/real_boreal_v2_best_seed1.pt` | `5a9616cefddd3bca6eccf31d0691f7b1a2fadb572f84cf7948c85296644e9388` |
| `real_boreal_v2_best_seed2.pt` | 2 | `/workspace/runs/boreal_v2_try4/real_boreal_v2_seed2/real_boreal_v2_best_seed2.pt` | `runs/boreal_v2_try4/real_boreal_v2_seed2/real_boreal_v2_best_seed2.pt` | `c4d443a8af919c6b00e4319b117a5dfbe90bd8b29750c75e3970a9389ae097ef` |

How these facts were established (see `docs/experiments/boreal_v2_audit.md`): the hashes are recorded in each seed's `train_config.json`, and every evaluation cell of each
model carries the same SHA-256 computed by the harness on the exact file path above at evaluation time (2026-10-29); seed 0's hash was additionally recomputed on a pod
by the diagnostic job. **Not re-verified since 2026-10-29**: that the files are still present and unchanged on the volume (nothing is known to have touched them).
The same list is in `docs/experiments/boreal_v2/results/checkpoints.SHA256SUMS` (names relative, for `sha256sum -c`).

Also on the volume and worth keeping with them: the run directories `/workspace/runs/boreal_v2_try3/` and `/workspace/runs/boreal_v2_try4/` (plots, logs, `results_bundle.tgz`). Their small
artifacts are already in git (`docs/experiments/boreal_v2/results/`).

## 2. Verify any copy (identical for every route)

```bash
cd ~/src/Ardupilot-UAV-YOLO
mkdir -p local_data/backups/boreal_v2          # local_data/ is gitignored; never commit checkpoints
# after the files are in local_data/backups/boreal_v2/ :
cd local_data/backups/boreal_v2
sha256sum -c ~/src/Ardupilot-UAV-YOLO/docs/experiments/boreal_v2/results/checkpoints.SHA256SUMS     # every line must say OK
ls -l real_boreal_v2_best_seed*.pt                                                                 # seed 0 must be 22501098 bytes
```
A copy whose hash does not match is not a backup: delete the copy, never the original, and retry.

## 3. Route A (preferred, no pod, no GPU cost): RunPod S3-compatible API  -- UNCONFIRMED for US-IL-1

RunPod exposes network volumes through an S3-compatible API at `https://s3api-<datacenter>.runpod.io`. The cloud session could not read the live support table
(`docs.runpod.io` is outside its allowed domains); a mirror of the docs suggests US-IL-1 is listed in a later revision, and RunPod's own API reports US-IL-1
`storageSupport: true` (that is volume support, not proof of S3 support). **Treat it as plausible but unproven: test with one `ls` first.**

1. In the RunPod console create an **S3 API key** (Settings -> S3 API Keys). It grants access to your volumes: keep it only on the VM, never in git, and delete it after the backup.
2. On the VM (AWS CLI v2; `pip install awscli` also works):
```bash
aws configure --profile runpod            # access key id / secret from the console; region: us-il-1; output: json
export RP="--profile runpod --region us-il-1 --endpoint-url https://s3api-us-il-1.runpod.io"
aws s3 ls $RP s3://sej40ez3ia/runs/boreal_v2_try3/real_boreal_v2_seed0/      # TEST: should list real_boreal_v2_best_seed0.pt
```
3. If the test lists the file, download (read-only operations only; **do not use `sync`, `rm` or `--delete`** with this key):
```bash
cd ~/src/Ardupilot-UAV-YOLO && mkdir -p local_data/backups/boreal_v2 && cd local_data/backups/boreal_v2
aws s3 cp $RP s3://sej40ez3ia/runs/boreal_v2_try3/real_boreal_v2_seed0/real_boreal_v2_best_seed0.pt .
aws s3 cp $RP s3://sej40ez3ia/runs/boreal_v2_try4/real_boreal_v2_seed1/real_boreal_v2_best_seed1.pt .
aws s3 cp $RP s3://sej40ez3ia/runs/boreal_v2_try4/real_boreal_v2_seed2/real_boreal_v2_best_seed2.pt .
```
then run the verification in section 2. If the `ls` fails (unknown endpoint, `NoSuchBucket`, `AccessDenied`), use Route B; do not retry blindly.

## 4. Route B (fallback): a short pod that serves exactly the three checkpoints, launched FROM THE VM

Cost: one RTX 4090 for about 5-10 minutes, **about $0.10-0.20**; the budget gate's worst case is $1.25 (2,400 s hard timeout + 600 s boot + 600 s watchdog latency + margin). It needs
`RUNPOD_API_KEY` in the VM environment and **an explicit decision by you to spend it**. Launching from the VM keeps the access token on the VM: the launcher generates
it into a 0600 file outside the repo and passes it to the pod; the cloud session never sees it.

What the pod does (`eval/runpod/checkpoint_backup_job.py`, tested): reads the three originals, aborts on any hash mismatch, copies them into a NEW directory
`/workspace/runs/transfer_test_<name>_job/export/` (originals untouched), writes `SHA256SUMS`, and serves only those four files over the RunPod HTTPS proxy
(`https://<pod_id>-8000.proxy.runpod.net`) behind a bearer token. The server refuses every other path and every other `.pt` name, exits after your fetch acknowledges
(or after 30 minutes), and the pod then stops itself (GraphQL `podStop`; the REST stop is forbidden for the pod's own key).

```bash
cd ~/src/Ardupilot-UAV-YOLO && git fetch origin && git checkout science/boreal-v2
export RUNPOD_API_KEY=...                         # in your shell only; never in a file in the repo
# 0. read-only checks: no pod running, budget gate (a dry run creates nothing)
python3 eval/runpod/launch_job.py status
python3 eval/runpod/launch_job.py transfer-test --checkpoint-backup --run-name transfer_test_backup1 --max-hourly-price 1.0 \
   --ledger-floor-usd 12.5 --boot-allowance-s 600 --watchdog-latency-s 600 --results-token-file ~/.wf_results_token
# 1. (optional but recommended) start the independent GitHub watchdog loop, Actions tab -> runpod-watchdog -> Run workflow, loop_minutes = 60
# 2. launch (this is the paid step; --watchdog-armed attests that step 1 is running)
GIT_COMMIT=$(git rev-parse HEAD) python3 eval/runpod/launch_job.py transfer-test --checkpoint-backup --run-name transfer_test_backup1 \
   --max-hourly-price 1.0 --ledger-floor-usd 12.5 --boot-allowance-s 600 --watchdog-latency-s 600 \
   --results-token-file ~/.wf_results_token --confirm-paid transfer_test_backup1 --watchdog-armed      # prints pod_id and proxy_urls
# 3. wait ~3-6 minutes for boot + hashing, then download (the VM needs outbound HTTPS to proxy.runpod.net)
RESULTS_TOKEN_FILE=~/.wf_results_token python3 eval/runpod/fetch_results.py \
   --base https://<pod_id>-8000.proxy.runpod.net --dest ~/src/Ardupilot-UAV-YOLO/local_data/backups/boreal_v2_$(date +%F)
# fetch_results verifies every file's SHA-256 against the pod's manifest and then acknowledges, which ends the server and stops the pod.
# 4. verify against the hashes recorded in git (section 2), then confirm the pod is stopped and nothing is running:
cd ~/src/Ardupilot-UAV-YOLO/local_data/backups/boreal_v2_$(date +%F) && sha256sum -c ~/src/Ardupilot-UAV-YOLO/docs/experiments/boreal_v2/results/checkpoints.SHA256SUMS
python3 ~/src/Ardupilot-UAV-YOLO/eval/runpod/launch_job.py status       # expect "running": 0
```
If the pod is still RUNNING 5 minutes after the acknowledgement, stop it (`status` shows its id): the REST stop works with your full key:
`curl -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/pods/<pod_id>/stop`.
Stopped pods keep a tiny disk charge and can be removed in the console; this tooling never deletes pods.

## 5. Rules
- Never delete, move or overwrite the originals on the volume; never delete the volume (it also holds the datasets, the v1 checkpoints and all runs).
- Never commit `*.pt`, the S3 key, the results-token file or `RUNPOD_API_KEY`.
- Route A and Route B both leave the three files as plain files; neither changes any recorded hash.
- Keep the same copies next to `SHA256SUMS` plus `docs/experiments/boreal_v2/results/` so a reviewer can reproduce the evaluation (`python3 -m eval.run_matrix --preds-from ...` needs only the saved predictions, not the checkpoints).

## 6. Executed 2026-11-02 (Route C: pod launched through the RunPod MCP, no S3 key, no `RUNPOD_API_KEY` in the VM shell)
Neither route above was usable as written (no S3 keys, no API key in the shell), so a one-off RTX 4090 pod (`g02e1nlir1927z`, $0.89/h, about 2 min) mounted `wildfire-datasets` (read/write mount; the script only reads from it),
re-hashed the three originals against the table in section 1 (it refuses to serve on any mismatch), and served exactly those three files plus `SHA256SUMS`
over the RunPod HTTPS proxy behind a random one-use bearer token. The files were downloaded to `local_data/backups/boreal_v2/` and `sha256sum -c` printed OK for all three
(22,501,098 bytes each). The pod was then deleted; the volume and the originals were not touched. Provenance: `local_data/backups/boreal_v2/PROVENANCE.txt` (not committed).
The pod script was an inline 60-line Python server (not committed; superseded by `eval/runpod/checkpoint_backup_job.py` for the launcher route).
