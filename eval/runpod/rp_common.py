"""Shared pieces for RunPod job orchestration (stdlib only; imported by launch_job.py and pod_supervisor.py).

Credentials are read from the environment by name and are never logged, written to result files or put in exceptions.
"""
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

API_BASE = "https://rest.runpod.io/v1"
UA = "wildfire-orchestrator/1"  # the default urllib User-Agent is rejected with 403

# Limits from CLAUDE.md "Autonomous Development and Compute Policy".
LIMITS = {"per_experiment_usd": 20.0, "autonomous_usd": 100.0, "max_running_pods": 1}
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,63}$")


class RunPodError(RuntimeError):
    pass


class RunPodAPI:
    """Minimal REST client. `key` defaults to $RUNPOD_API_KEY. Has no delete method on purpose."""

    def __init__(self, key=None, base=API_BASE, timeout=30):
        self._key = key or os.environ.get("RUNPOD_API_KEY")
        if not self._key:
            raise RunPodError("RUNPOD_API_KEY is not set")
        self.base, self.timeout = base, timeout

    def _call(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method, headers={
            "Authorization": "Bearer " + self._key, "Content-Type": "application/json", "User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:  # never include request headers
            raise RunPodError(f"{method} {path} -> HTTP {e.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RunPodError(f"{method} {path} -> {type(e).__name__}") from None
        return json.loads(raw) if raw.strip() else {}

    def list_pods(self):
        return self._call("GET", "/pods")

    def get_pod(self, pod_id):
        return self._call("GET", f"/pods/{pod_id}")

    def create_pod(self, spec):
        return self._call("POST", "/pods", spec)

    def stop_pod(self, pod_id):
        return self._call("POST", f"/pods/{pod_id}/stop")

    def billed_total(self):
        pods = self._call("GET", "/billing/pods")
        vols = self._call("GET", "/billing/networkvolumes")
        return sum(r["amount"] for r in pods) + sum(r["amount"] for r in vols)


def is_running(pod):
    return str(pod.get("desiredStatus", "")).upper() == "RUNNING"


def is_stopped(pod):
    return str(pod.get("desiredStatus", "")).upper() in ("EXITED", "STOPPED")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class EventLog:
    """Append-only JSONL log (one object per line) so a crash never corrupts earlier records."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def emit(self, event, **fields):
        rec = {"t": utc_now(), "event": event, **fields}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return rec


def write_json_new(path, obj):
    """Write JSON atomically and refuse to overwrite (results are never replaced)."""
    if os.path.exists(path):
        raise FileExistsError(path)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, path)  # fails if path appeared meanwhile
    os.remove(tmp)


def notify(url, payload, post=None, timeout=10):
    """Best-effort JSON POST (ntfy topic, Slack/Discord-style webhook or any HTTP endpoint). Never raises; returns bool.

    `url` is a secret-bearing value: it is not echoed anywhere.
    """
    if not url:
        return False
    try:
        if post is not None:
            post(url, payload)
            return True
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers={
            "Content-Type": "application/json", "User-Agent": UA,
            "Title": payload.get("title", "runpod job")})
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def check_budget(*, price_per_hr, hard_timeout_s, billed_usd, ledger_floor_usd=0.0, other_running=0, setup_margin_usd=0.25,
                 limits=LIMITS):
    """Return (ok, reasons, worst_case_usd). Worst case = hourly price * hard timeout + margin (GPU billed to the stop call)."""
    worst = price_per_hr * hard_timeout_s / 3600.0 + setup_margin_usd
    spent = max(billed_usd, ledger_floor_usd)  # billing lags by hours, so the ledger floor is the conservative bound
    reasons = []
    if other_running >= limits["max_running_pods"]:
        reasons.append(f"{other_running} pod(s) already running (limit {limits['max_running_pods']})")
    if worst > limits["per_experiment_usd"]:
        reasons.append(f"worst case ${worst:.2f} exceeds per-experiment limit ${limits['per_experiment_usd']:.2f}")
    if spent + worst > limits["autonomous_usd"]:
        reasons.append(f"spent ${spent:.2f} + worst case ${worst:.2f} exceeds autonomous allowance ${limits['autonomous_usd']:.2f}")
    return (not reasons), reasons, worst
