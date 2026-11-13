#!/usr/bin/env python3
"""
eval/level3_queue.py — run the planned Level 3 configs through N parallel slots (eval/level3_pairs.py `run --slot k`).

    python3 eval/level3_queue.py --plan local_data/level3/final --python ~/venvs/yolo-cpu/bin/python --slots 3 \
        --stop-file local_data/level3/final/STOP [--repeat baseline:2001 A0:2001] [--dry-run]

* Jobs come from <plan>/matrix.json ordered by scenario then arm (partial results stay balanced). A job is skipped when
  <plan>/results/<tag>/run_meta.json is clean (fog_exit_code 0 and empty leftover_after_cleanup); nothing existing is overwritten.
* --repeat ARM:SCENARIO re-runs a finished config into <plan>/results_repeat/<tag> (a separate directory, jitter measurement).
* Touch the stop file to halt: running jobs finish, no new job starts. Delete it to resume.
* A job also waits while available memory is below --mem-per-run-mb (+ headroom), so slots never push the VM into swap.
* Waiting is done on the Popen handles this script created; there is no pgrep/pkill by command line.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = os.path.join(ROOT, "eval", "level3_pairs.py")
ARM_ORDER = ["baseline", "A0", "A1", "A2", "B2s0", "B2s1", "B2s2"]


def is_clean(out_dir: str) -> bool:
    try:
        m = json.load(open(os.path.join(out_dir, "run_meta.json")))
    except (OSError, ValueError):
        return False
    return m.get("fog_exit_code") == 0 and m.get("leftover_after_cleanup") == []


def mem_available_mb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024.0
    return 0.0


def build_jobs(plan: str, repeats=(), only=None) -> list:
    matrix = json.load(open(os.path.join(plan, "matrix.json")))
    rows = sorted(matrix["runs"], key=lambda r: (r["scenario"], ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 99))
    jobs = []
    for r in rows:
        tag = os.path.splitext(os.path.basename(r["config"]))[0]
        out = os.path.join(plan, "results", tag)
        if only and tag not in only:
            continue
        if is_clean(out):
            continue
        jobs.append({"tag": tag, "config": os.path.abspath(r["config"]), "out": os.path.abspath(out), "kind": "main"})
    for rep in repeats:
        arm, sc = rep.split(":")
        tag = f"l3-{arm}-{sc}"
        cfg = os.path.join(plan, f"{tag}.yaml")
        out = os.path.join(plan, "results_repeat", tag)
        if is_clean(out):
            continue
        jobs.append({"tag": tag, "config": os.path.abspath(cfg), "out": os.path.abspath(out), "kind": "repeat"})
    return jobs


class Queue:
    def __init__(self, jobs, python, slots, stop_file, mem_mb, headroom_mb, log_path, stagger_s):
        self.jobs, self.python, self.slots = list(jobs), python, slots
        self.stop_file, self.mem_mb, self.headroom_mb, self.stagger_s = stop_file, mem_mb, headroom_mb, stagger_s
        self.lock = threading.Lock()
        self.log_path = log_path
        self.logdir = os.path.join(os.path.dirname(log_path), "queue_logs")
        os.makedirs(self.logdir, exist_ok=True)
        self.results = []
        self.last_start = 0.0

    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%d_%H:%M:%S')} {msg}"
        with self.lock:
            print(line, flush=True)
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    def next_job(self):
        with self.lock:
            return self.jobs.pop(0) if self.jobs else None

    def worker(self, slot):
        while True:
            if os.path.exists(self.stop_file):
                self.log(f"slot {slot}: stop file present, not starting another run")
                return
            with self.lock:
                if not self.jobs:
                    return
            # stagger SITL boots and wait for memory before claiming a job
            while True:
                if os.path.exists(self.stop_file):
                    return
                with self.lock:
                    ok = (time.time() - self.last_start >= self.stagger_s) and mem_available_mb() >= self.mem_mb + self.headroom_mb
                    if ok:
                        self.last_start = time.time()
                        job = self.jobs.pop(0) if self.jobs else None
                        break
                time.sleep(5)
            if job is None:
                return
            self.run_job(slot, job)

    def run_job(self, slot, job):
        self.log(f"slot {slot}: START {job['kind']} {job['tag']} (MemAvailable {mem_available_mb():.0f} MB)")
        t0 = time.time()
        cmd = [self.python, PAIRS, "run", "--config", job["config"], "--python", self.python, "--out", job["out"], "--slot", str(slot)]
        with open(os.path.join(self.logdir, f"{job['kind']}-{job['tag']}.log"), "a") as f:
            rc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).wait()
        clean = is_clean(job["out"])
        self.results.append({"tag": job["tag"], "kind": job["kind"], "slot": slot, "rc": rc, "clean": clean})
        self.log(f"slot {slot}: END {job['kind']} {job['tag']} rc={rc} clean={clean} wall={time.time() - t0:.0f}s")

    def run(self):
        ths = [threading.Thread(target=self.worker, args=(k,), daemon=True) for k in range(self.slots)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        bad = [r for r in self.results if not r["clean"]]
        self.log(f"queue finished: {len(self.results)} run(s), {len(bad)} not clean {[r['tag'] for r in bad]}; {len(self.jobs)} left unstarted")
        return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--repeat", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=None, help="restrict main jobs to these tags")
    ap.add_argument("--repeat-first", action="store_true", help="put the repeats at the front of the queue (contention measurement)")
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--mem-per-run-mb", type=float, default=750)
    ap.add_argument("--headroom-mb", type=float, default=150)
    ap.add_argument("--stagger-s", type=float, default=45)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    plan = os.path.abspath(a.plan)
    jobs = build_jobs(plan, a.repeat, a.only)
    if a.repeat_first:
        jobs.sort(key=lambda j: j["kind"] != "repeat")
    if a.max_jobs:
        jobs = jobs[:a.max_jobs]
    stop = a.stop_file or os.path.join(plan, "STOP")
    print(f"{len(jobs)} job(s), {a.slots} slot(s), stop file {stop}")
    for j in jobs:
        print(f"  {j['kind']:6s} {j['tag']}  -> {os.path.relpath(j['out'], plan)}")
    if a.dry_run:
        return 0
    q = Queue(jobs, a.python, a.slots, stop, a.mem_per_run_mb, a.headroom_mb, os.path.join(plan, "queue.log"), a.stagger_s)
    return 0 if q.run() else 1


if __name__ == "__main__":
    sys.exit(main())
