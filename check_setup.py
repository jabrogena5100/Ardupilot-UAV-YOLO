#!/usr/bin/env python3
"""
check_setup.py - tells you which parts of this project work on THIS computer.  Read-only; changes nothing.

    python3 check_setup.py

Four tiers, each needs more than the last:
  1 REPLAY      view the recorded final experiments in a browser (Python only)
  2 RECOMPUTE   re-derive tables/figures from saved run files (needs the data bundle)
  3 FRESH RUN   run a new SITL experiment (ArduPilot SITL, tmux, pymavlink, ...); arms A/B also need weights + ultralytics
  4 TRAINING    out of scope for a laptop (GPU, datasets)
Exit status is 0 if tier 1 works, else 1.
"""
import importlib.util
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OK, WARN, NO = "[ ok ]", "[warn]", "[FAIL]"
tier1_ok = True


def has(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def line(flag, msg, hint=""):
    print(f"  {flag} {msg}" + (f"\n         -> {hint}" if hint and flag != OK else ""))


def p(*a):
    return os.path.join(ROOT, *a)


print(f"Python {sys.version.split()[0]} at {sys.executable}")
v = sys.version_info[:2]
line(OK if v >= (3, 9) else NO, "Python >= 3.9 (tested with 3.10)", "install Python 3.10+")

print("\nTier 1 - REPLAY (browser demo)")
snap = os.path.exists(p("docs", "demo", "level3_final_replay.html"))
raw = os.path.isdir(p("local_data", "level3", "final", "results"))
line(OK if snap else NO, "committed snapshot docs/demo/level3_final_replay.html", "git status; the file should come with the clone")
line(OK if raw else WARN, "raw final runs local_data/level3/final/results (optional; gitignored)",
     "not in git; only needed for the live-from-files viewer and tier 2")
tier1_ok = snap or raw
line(OK if has("yaml") else WARN, "PyYAML (only for the full viewer, not the snapshot)", "pip install PyYAML")
line(OK if os.path.exists(p("docs", "experiments", "level3_final", "per_run.csv")) else NO, "recorded metrics table per_run.csv")

print("\nTier 2 - RECOMPUTE from saved runs")
n = len(os.listdir(p("local_data", "level3", "final", "results"))) if raw else 0
line(OK if n >= 56 else WARN, f"{n} run directories (need 56)", "request the data bundle (see docs/PROFESSOR_GUIDE.md section 2)")
line(OK if has("numpy") else WARN, "numpy", "pip install numpy")

print("\nTier 3 - FRESH SITL RUN")
for m, pk in (("yaml", "PyYAML"), ("numpy", "numpy"), ("cv2", "opencv-python-headless"), ("PIL", "pillow"),
              ("pymavlink", "pymavlink"), ("MAVProxy", "MAVProxy")):
    line(OK if has(m) else WARN, f"python module {m}", f"pip install {pk}")
line(OK if shutil.which("tmux") else WARN, "tmux", "sudo apt install tmux")
sv = shutil.which("sim_vehicle.py")
line(OK if sv else WARN, "sim_vehicle.py on PATH (ArduPilot checkout: Tools/autotest)", "see docs/PROFESSOR_GUIDE.md section 5")
if sv:
    real = os.path.realpath(sv)
    ac = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(real))), "build", "sitl", "bin", "arducopter")
    line(OK if os.path.exists(ac) else WARN, "SITL copter built (build/sitl/bin/arducopter)", "cd <ardupilot> && ./waf configure --board sitl && ./waf copter")
try:
    mem = int([l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1]) / 1048576
    line(OK if mem >= 1.8 else WARN, f"RAM {mem:.1f} GB (the final runs used a 2 GB VM, one run at a time)", "give the VM at least 2 GB")
except OSError:
    line(WARN, "could not read /proc/meminfo (not Linux?)")
line(OK if os.path.exists(p("eval", "level3_thresholds.json")) else NO, "frozen thresholds/fire/scenario files in eval/")

print("\n  Arms with a detector (A, B) also need:")
line(OK if has("ultralytics") and has("torch") else WARN, "ultralytics + torch (CPU build is enough)", "pip install -r requirements-yolo.txt")
try:
    sys.path.insert(0, ROOT)
    from eval import level3_pairs as LP
    for arm, v in LP.ARMS.items():
        if v is None:
            continue
        ok = os.path.exists(p(v[0]))
        line(OK if ok else WARN, f"weights for {arm}: {v[0]}", "not in git; see docs/PROFESSOR_GUIDE.md section 2.3")
except Exception as e:
    line(WARN, f"could not read the arm table ({e})")

print("\nTier 4 - TRAINING: needs a GPU (the project used RunPod) and the Boreal and synthetic datasets; not checked here.")
print("\nSummary: tier 1 (replay) " + ("READY - run: python3 demo.py" if tier1_ok else "NOT ready"))
sys.exit(0 if tier1_ok else 1)
