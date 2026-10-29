# RunPod watchdog: security review and verification guide (2026-10-29)

Scope: `.github/workflows/runpod-watchdog.yml`, `eval/runpod/watchdog.py`, `eval/runpod/rp_common.py` as merged in PR #1 (`c1daa45`;
infra files identical to reviewed commit `0c0fed3`) plus the hardening branch `cloud/watchdog-hardening`. Reviewer: Claude (cloud session). Not an
independent audit; the owner or Codex should still read the workflow once.

## Evidence already collected (live)
- Workflow run #1 (manual `workflow_dispatch` on `main`, actor = owner) succeeded on a GitHub-hosted runner. The log shows `RUNPOD_API_KEY: ***`
  (secret present, masked), the enforce step reached RunPod and exited 0 (an API failure would exit 2), and "no events this pass" (no pods exist).
- A scheduled (cron) execution has **not** been observed yet at the time of writing.

## Findings
| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | Medium | Original workflow: `actions/checkout` persisted the job's write token in `.git/config`, so the step holding `RUNPOD_API_KEY` (running repo code) could also read a token with `contents: write`. | **Fixed**: `persist-credentials: false`; the token is passed (as `GH_TOKEN` env) only to the log-append step; the secret only to the enforce step. A test asserts both. |
| 2 | Low | Token was interpolated into clone URLs in a `run:` script. | **Fixed**: passed via env and `http.extraheader`, masked with `::add-mask::`; no `${{ }}` expressions inside any `run:` body (tested). |
| 3 | Info | `contents: write` is job-wide; the token could in principle push to `main`. The workflow only pushes `runpod-logs`. | Accepted; recommended: branch protection on `main` (blocks pushes by the Actions token unless allowed). |
| 4 | Owner | Anyone with write access can dispatch the workflow on a branch of their own and, with a repository-level secret, read it. Single-owner repo today. | Not changed (would disarm the verified setup). If collaborators are added: create Environment `runpod` restricted to `main`, move the secret there, add `environment: runpod` to the job. |
| 5 | Owner | "Cannot delete pods or volumes" is a property of **our code** (verified), not of the key. A RunPod key with delete scope could delete if leaked. | Use a restricted RunPod key; RunPod's scope options were not verified from here. |
| 6 | Design | Only pods whose name ends `__dl<10-digit epoch>` are ever stopped, and only after the deadline passed. Untagged running pods are reported (`untagged_running_pod`), never stopped (flag `--stop-untagged` exists, unused). A manually created pod is therefore **not** protected. | Intentional; decide whether to ever enable `--stop-untagged`. |
| 7 | Design | Any pod carrying an expired tag is stopped regardless of name prefix. Stopping is the safe direction; a tag cannot make the watchdog start, resize or delete anything. | Accepted. |
| 8 | Ops | Cron may lag (typically minutes); GitHub pauses scheduled workflows after 60 days of repository inactivity; failed runs notify by GitHub's default mechanisms only. | Documented; worst-case budget includes 600 s latency. |
| 9 | Info | Repo is public: branch `runpod-logs` is world-readable (pod ids, names, timestamps, error strings; no keys, no spend). | Accepted; do not encode sensitive text in run names. |
| 10 | Info | `actions/checkout@v4` is pinned to a tag, not a commit SHA (first-party action; Node 20 deprecation warning seen). The only non-checkout code is stdlib Python from this repo. | Optional: pin to SHA. |
| 11 | Info | PR #1 also carried earlier handoff commits (e.g. `CLAUDE.md` rewrite, level2 docs) to `main`; not part of the orchestration diff. | Owner awareness. |

## What the watchdog can and cannot do (code-level, tested)
- API calls it can make: `GET /pods`, `GET /pods/{id}`, `POST /pods/{id}/stop`. `RunPodAPI` has no DELETE/terminate method and no volume
  mutation (`GET /networkvolumes` only); tests check the verb/path set and that `enforce` touches only list/get/stop.
- It never creates or starts pods. It never reads or writes the network volume. Checkpoints, logs and `DONE.json` are unaffected.
- GitHub side: writes only `enforce_events.jsonl` on `runpod-logs` (append commits, no force-push; tested). Logging permission needed: `contents: write`.
- Independence: runs on GitHub-hosted runners with the repo secret; needs neither Claude Code Cloud, the Ubuntu VM nor the pod.
  (Claude Code Cloud only reads its output.)

## Verification procedure (owner, about 10 minutes, no paid resources)
1. Merge the hardening PR (`cloud/watchdog-hardening` -> `main`). Confirm the `RUNPOD_API_KEY` repository secret exists (run #1 already proves it).
2. Actions > runpod-watchdog > **Run workflow** on `main` with `selftest` = true. Expect: green run; Summary shows a `selftest` event with
   `api_read: ok`, `pods_total: 0`; branch `runpod-logs` now exists with `enforce_events.jsonl` containing that line (proves log-write permission).
   Confirm no force-push or main change: the only new commit is on `runpod-logs`.
3. Wait for a **scheduled** run (event = `schedule`; within roughly 5-20 min of the merge): green, "none (no running pods)". The Actions list shows
   the event type; paste the run URL to the next session or ask it to check with `actions_list`.
4. Negative control (optional, free): temporarily rename the secret; the next run must fail with "watchdog is NOT enforcing". Restore it.
5. Only after 2 and 3 pass is the watchdog "armed"; then the lifecycle test may be requested with `--watchdog-armed`.

## Status
Armed? **Not yet**: hardening not merged, selftest not run, no scheduled run observed. Lifecycle test not started; $0 spent.
