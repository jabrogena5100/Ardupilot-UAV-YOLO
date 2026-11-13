"""
dashboard/replay_data.py — read-only loader for saved Level 3 pilot runs.

Everything here reads files that already exist on disk. It never starts SITL,
never imports ultralytics/torch, and never writes into a results directory.
The only thing it writes is the optional trajectory cache built by
`dashboard/replay_build.py` (under local_data/level3/_replay_cache/).

Rule for gaps: if a file is missing or a value cannot be derived from a saved
artifact, the field is None and `missing` explains why. Nothing is imputed,
interpolated or defaulted.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L3 = os.path.join(ROOT, "local_data", "level3")
CACHE = os.path.join(L3, "_replay_cache")

# pilot directory -> (label, note). Order is display order.
PILOTS = {
    "final": ("FINAL held-out evaluation (56 runs, single UAV, 600 s)",
              "The completed Level 3 paired evaluation reported in docs/experiments/level3_final.md: 8 scenarios x 7 arms, oracle off, one UAV."),
    "pilot2": ("pilot 2 (corrected)", "Current pilot: geofence sign fix, ignition v2, Model B v2 thresholds."),
    "pilot": ("pilot 1 (superseded)",
              "Superseded: run before the geofence-sign fix and ignition v2; coverage baseline never reached the fire "
              "(docs/experiments/level3_methodology.md sec. 10, 12)."),
}
ARM_ORDER = {"baseline": 0, "A0": 1, "A1": 2, "A2": 3, "B2s0": 4, "B2s1": 5, "B2s2": 6}
ARM_LABEL = {"baseline": "Baseline (no detector)",
             "A0": "Model A (synthetic-trained, seed 0)", "A1": "Model A (synthetic-trained, seed 1)", "A2": "Model A (synthetic-trained, seed 2)",
             "B2s0": "Model B (real-trained, seed 0)", "B2s1": "Model B (real-trained, seed 1)", "B2s2": "Model B (real-trained, seed 2)"}
FAMILY = {"baseline": "baseline", "A0": "A", "A1": "A", "A2": "A", "B2s0": "B", "B2s1": "B", "B2s2": "B"}
T_END = 600.0
INV_RE = re.compile(r"mode=INVESTIGATE\u2192(\d+),(\d+)")
TAG_RE = re.compile(r"^l3-(?P<arm>[A-Za-z0-9]+?)-(?P<scn>\d+)$")


# ---------------- small readers ----------------

def _json(path: str) -> Optional[Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _csv(path: str) -> Optional[List[Dict[str, str]]]:
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return None


def _kv(path: str) -> Optional[Dict[str, str]]:
    try:
        with open(path) as f:
            out = {}
            for line in f:
                if "=" in line:
                    k, v = line.rstrip("\n").split("=", 1)
                    out[k] = v
            return out
    except OSError:
        return None


def _yaml(path: str) -> Optional[Dict[str, Any]]:
    try:
        import yaml  # PyYAML is already a project requirement
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _f(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _first(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


# ---------------- discovery ----------------

def _classify(d: str) -> Dict[str, Any]:
    """Status of one run directory, from the files in it."""
    end = _first(glob.glob(os.path.join(d, "experiment_end_*.txt")))
    manifest = _json(os.path.join(d, "manifest.json")) or {}
    run_meta = _json(os.path.join(d, "run_meta.json"))
    endkv = _kv(end) if end else None
    has_tlog = os.path.exists(os.path.join(d, "sitl_work", "mav.tlog"))
    if not endkv:
        return {"status": "interrupted",
                "why": "experiment_end file missing or empty: the run did not finish writing its result (e.g. host crash)",
                "end": None, "has_tlog": has_tlog, "manifest": manifest}
    reason = endkv.get("reason")
    if reason != "timeout":
        return {"status": "interrupted", "why": f"ended early (reason={reason}), not a full-length run",
                "end": endkv, "has_tlog": has_tlog, "manifest": manifest}
    if manifest.get("fog_exit_code") not in (0, None):
        return {"status": "interrupted", "why": f"fog_exit_code={manifest.get('fog_exit_code')}",
                "end": endkv, "has_tlog": has_tlog, "manifest": manifest}
    return {"status": "completed", "why": None, "end": endkv, "has_tlog": has_tlog, "manifest": manifest}


def _end_bytes(d: str) -> bytes:
    p = _first(glob.glob(os.path.join(d, "experiment_end_*.txt")))
    try:
        return open(p, "rb").read() if p else b""
    except OSError:
        return b""


def discover() -> Dict[str, Any]:
    """List runs, planned-but-never-run cells, and discarded attempts."""
    runs: List[Dict[str, Any]] = []
    res_sig: Dict[Any, bytes] = {}

    for pilot in PILOTS:
        pdir = os.path.join(L3, pilot)
        # results/<tag> is the full copy (has SITL logs). Prefer it.
        for d in sorted(glob.glob(os.path.join(pdir, "results", "*"))):
            if not os.path.isdir(d):
                continue
            res_sig[(pilot, os.path.basename(d))] = _end_bytes(d)
            runs.append(_describe(pilot, d, os.path.basename(d)))
        # runs/<stamp>-<tag>: dirs not already represented by results/
        for d in sorted(glob.glob(os.path.join(pdir, "runs", "*"))):
            if not os.path.isdir(d):
                continue
            tag = re.sub(r"^\d{8}-\d{6}-", "", os.path.basename(d))
            # results/<tag> is a copy of the matching runs/ dir: same experiment_end bytes -> same run, skip the copy
            if (pilot, tag) in res_sig and res_sig[(pilot, tag)] == _end_bytes(d) and _end_bytes(d):
                continue
            runs.append(_describe(pilot, d, os.path.basename(d)))

    # post-fix trajectory diagnostic (a one-off baseline flight)
    d = os.path.join(L3, "pilot2", "traj_baseline_postfix")
    if os.path.isdir(d):
        runs.append(_describe("pilot2", d, "traj_baseline_postfix", diagnostic=True))

    # planned-but-not-run: matrix.json cells with no run directory
    planned: List[Dict[str, Any]] = []
    for pilot in PILOTS:
        mx = _json(os.path.join(L3, pilot, "matrix.json")) or {}
        have = {(r["pilot"], r["arm"], r["scenario"]) for r in runs if not r.get("diagnostic")}
        for c in mx.get("runs", []):
            key = (pilot, c.get("arm"), str(c.get("scenario")))
            if key not in have:
                planned.append({"pilot": pilot, "arm": c.get("arm"), "scenario": str(c.get("scenario")),
                                "label": ARM_LABEL.get(c.get("arm"), c.get("arm")),
                                "status": "not_run",
                                "why": "in matrix.json but no run directory or result exists"})
    discarded = []
    dd = os.path.join(L3, "pilot_attempt1_sync_discarded")
    if os.path.isdir(dd):
        discarded.append({"path": "local_data/level3/pilot_attempt1_sync_discarded",
                          "why": "attempt 1 exited at the start gate; discarded by the methodology doc, holds no result CSVs"})

    runs.sort(key=lambda r: (list(PILOTS).index(r["pilot"]), r["scenario"] or "", ARM_ORDER.get(r["arm"], 9), r["id"]))
    planned.sort(key=lambda r: (r["scenario"], ARM_ORDER.get(r["arm"], 9)))
    return {"runs": runs, "planned": planned, "discarded": discarded,
            "pilots": {k: {"label": v[0], "note": v[1]} for k, v in PILOTS.items()},
            "arm_label": ARM_LABEL}


def _describe(pilot: str, d: str, name: str, diagnostic: bool = False) -> Dict[str, Any]:
    c = _classify(d)
    tag = name
    mt = None
    # runs/<stamp>-<tag> -> tag
    m = re.match(r"^\d{8}-\d{6}-(.+)$", name)
    if m:
        tag = m.group(1)
    mt = TAG_RE.match(tag)
    end = c["end"] or {}
    mf = c["manifest"]
    traj = os.path.exists(os.path.join(CACHE, f"{pilot}__{name}.traj.json"))
    return {
        "id": f"{pilot}/{name}", "pilot": pilot, "name": name, "tag": tag,
        "arm": mt.group("arm") if mt else None, "scenario": mt.group("scn") if mt else None,
        "label": ARM_LABEL.get(mt.group("arm")) if mt else ("Baseline diagnostic flight (not part of the matrix)" if diagnostic else tag),
        "status": c["status"], "why": c["why"], "diagnostic": diagnostic,
        "start_utc": mf.get("start_utc"), "wall_s": mf.get("wall_s"),
        "det_true": _f(end.get("det_true")), "det_false": _f(end.get("det_false")),
        "visited_frac": _f(end.get("visited_frac")), "burnt_frac": _f(end.get("burnt_frac")),
        "has_telemetry": c["has_tlog"], "telemetry_cached": traj,
        "dir": os.path.relpath(d, ROOT),
    }


# ---------------- full run load ----------------

def _resolve(run_id: str) -> Optional[str]:
    pilot, _, name = run_id.partition("/")
    if pilot not in PILOTS or not name or ".." in name or "/" in name:
        return None
    for sub in ("results", "runs", ""):
        d = os.path.join(L3, pilot, sub, name)
        if os.path.isdir(d) and (sub or name == "traj_baseline_postfix"):
            return d
    return None


def load_run(run_id: str) -> Optional[Dict[str, Any]]:
    d = _resolve(run_id)
    if d is None:
        return None
    pilot, _, name = run_id.partition("/")
    desc = _describe(pilot, d, name, diagnostic=(name == "traj_baseline_postfix"))
    tag = desc["tag"]
    missing: Dict[str, str] = {}

    args = _json(os.path.join(d, "args.json")) or {}
    cfg = _yaml(_first(glob.glob(os.path.join(d, "l3-*.yaml"))) or "") or {}
    manifest = _json(os.path.join(d, "manifest.json")) or {}
    run_meta = _json(os.path.join(d, "run_meta.json"))
    gate = _json(os.path.join(d, "start_gate.json"))
    end = _kv(_first(glob.glob(os.path.join(d, "experiment_end_*.txt"))) or "")

    N = int(args.get("grid_cells") or (cfg.get("grid") or {}).get("cells") or 10)
    side_m = float(args.get("grid_miles") or (cfg.get("grid") or {}).get("miles") or 1.0) * 1609.344

    percep = cfg.get("perception") or {}
    detector_on = bool(percep.get("enabled")) if cfg else None
    meta = {
        "tag": tag, "arm": desc["arm"], "scenario": desc["scenario"], "label": desc["label"],
        "status": desc["status"], "why": desc["why"], "pilot": pilot,
        "N": N, "side_m": side_m, "cell_m": side_m / N,
        "origin": [args.get("origin_lat"), args.get("origin_lon")],
        "max_seconds": args.get("max_seconds"),
        "detector_enabled": detector_on,
        "model_tag": (run_meta or {}).get("model_tag") or percep.get("model_tag"),
        "conf_threshold": (run_meta or {}).get("conf_threshold") or percep.get("conf_threshold"),
        "weights_sha256": (((run_meta or {}).get("models") or [{}])[0]).get("sha256"),
        "git_commit": manifest.get("git_commit"), "git_dirty": manifest.get("git_dirty"),
        "start_utc": manifest.get("start_utc"), "wall_s": manifest.get("wall_s"),
        "oracle_hazards": args.get("oracle_hazards"),
        "start_gate_wait_s": (gate or {}).get("wait_s"),
        "ignition": (cfg.get("fire_model") or {}).get("initial_ignitions"),
        "scenario_seed": (percep.get("camera_service") or {}).get("scenario_seed"),
    }
    if not cfg:
        missing["config"] = "run yaml missing or unreadable"
    if run_meta is None:
        missing["run_meta"] = "run_meta.json not present (weights/thresholds provenance unknown)"

    # ---- fire: per-cell ignition times (exact) ----
    fire_rows = _csv(_first(glob.glob(os.path.join(d, "fire_cells_*.csv"))) or "")
    fire = None
    if fire_rows is None:
        missing["fire"] = "fire_cells csv missing"
    else:
        fire = [{"i": int(r["i"]), "j": int(r["j"]), "state_end": int(r["fire_state_end"]),
                 "ignite_t": _f(r["ignite_t"]), "first_visit_t": _f(r["first_visit_t"])} for r in fire_rows]
    burn_s = None
    fm = cfg.get("fire_model") or {}
    if fm.get("burn_steps") is not None and fm.get("dt_seconds") is not None:
        burn_s = float(fm["burn_steps"]) * float(fm["dt_seconds"])
    else:
        missing["burn_duration"] = "burn_steps/dt_seconds not in run config; burnt state timing not derivable"

    fire_t = _series(_first(glob.glob(os.path.join(d, "fire_t_*.csv"))),
                     ["t", "burning_cnt", "burnt_cnt"], missing, "fire_t")
    cov_t = _series(_first(glob.glob(os.path.join(d, "coverage_t_*.csv"))),
                    ["t", "visited_frac", "traverse_frac"], missing, "coverage_t")

    # cross-check: do the per-cell ignition times reproduce the logged burning/burnt counts?
    fire_check = None
    if fire and burn_s is not None and fire_t and fire_t["t"]:
        worst_burning = worst_burnt = 0
        for t, bg, bt in zip(fire_t["t"], fire_t["burning_cnt"], fire_t["burnt_cnt"]):
            if t is None or bg is None or bt is None:
                continue
            g = sum(1 for c in fire if c["ignite_t"] is not None and c["ignite_t"] <= t < c["ignite_t"] + burn_s)
            b = sum(1 for c in fire if c["ignite_t"] is not None and t >= c["ignite_t"] + burn_s)
            worst_burning = max(worst_burning, abs(g - bg))
            worst_burnt = max(worst_burnt, abs(b - bt))
        fire_check = {"max_abs_diff_burning_cells": worst_burning, "max_abs_diff_burnt_cells": worst_burnt,
                      "samples": len(fire_t["t"])}

    # ---- coverage: per-cell first-visit time (exact) ----
    cell_rows = _csv(_first(glob.glob(os.path.join(d, "fog_cells_*.csv"))) or "")
    cells = None
    if cell_rows is None:
        missing["coverage_cells"] = "fog_cells csv missing"
    else:
        # fog writes visited_first_t = 0.0 for cells that were never visited; that is "no visit", not "visited at t=0"
        cells = [{"i": int(r["i"]), "j": int(r["j"]), "visited": int(r["visited"]),
                  "first_t": _f(r["visited_first_t"]) if int(r["visited"]) else None, "visits": _f(r["visits"])} for r in cell_rows]

    # ---- detections ----
    det_path = _first(glob.glob(os.path.join(d, "detections_*.csv")))
    dets = None
    if det_path is None:
        missing["detections"] = "detections csv missing"
    else:
        dets = []
        for r in _csv(det_path) or []:
            dets.append({"t": _f(r["t"]), "i": int(r["i"]), "j": int(r["j"]), "conf": _f(r["conf"]),
                         "truth_state": r.get("truth_state"), "correct": int(r["correct"]) if r.get("correct") not in (None, "") else None})
    if detector_on is False:
        missing["detections"] = ("detector disabled in this arm (baseline): zero detections here means "
                                 "'no detector', not 'detector saw nothing'")

    # ---- trajectory (cached from the saved MAVLink tlog) ----
    traj = _json(os.path.join(CACHE, f"{pilot}__{name}.traj.json"))
    if traj is None:
        missing["trajectory"] = ("no trajectory cache (not built, or build failed: run dashboard/replay_build.py and read its output)" if desc["has_telemetry"]
                                 else "no sitl_work/mav.tlog saved for this run: UAV path cannot be shown")

    inv = investigation(d, fire, missing)
    rec = recorded_metrics(pilot, tag)
    if rec is None and pilot == "final":
        missing["recorded_metrics"] = "no row for this run in analysis/per_run.csv"
    meta["family"] = FAMILY.get(desc["arm"])
    return {"desc": desc, "meta": meta, "end": end, "fire": fire, "burn_s": burn_s, "fire_t": fire_t,
            "fire_check": fire_check, "coverage_t": cov_t, "cells": cells, "detections": dets, "trajectory": traj,
            "investigation": inv, "recorded": rec, "missing": missing}


def investigation(d: str, fire: Optional[List[Dict[str, Any]]], missing: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Cells the UAV was sent to investigate, from the agent HUD log (mode=INVESTIGATE->i,j).

    The HUD lines carry NO timestamps (docs/experiments/level3_final.md sec. 7), so targets are
    returned in first-appearance order with a line count, and are NOT placed on the time axis.
    `near_fire` uses the same rule as eval/level3_analyze.py: within Chebyshev distance 1 of a cell
    that burned at any time in the run (judged in hindsight; the UAV did not know this).
    """
    log = os.path.join(d, "agent_A.log")
    if not os.path.exists(log):
        missing["investigation"] = "agent_A.log not saved for this run: investigation targets unknown"
        return None
    burned = {(c["i"], c["j"]) for c in (fire or [])}
    order: List[List[int]] = []
    count: Dict[tuple, int] = {}
    with open(log, errors="replace") as f:
        for line in f:
            m = INV_RE.search(line)
            if not m:
                continue
            c = (int(m.group(1)), int(m.group(2)))
            if c not in count:
                order.append(list(c))
            count[c] = count.get(c, 0) + 1
    targets = []
    for k, c in enumerate(order):
        near = any(max(abs(c[0] - b[0]), abs(c[1] - b[1])) <= 1 for b in burned) if fire is not None else None
        targets.append({"i": c[0], "j": c[1], "order": k + 1, "hud_lines": count[tuple(c)], "near_fire": near})
    return {"targets": targets, "timed": False,
            "note": "Agent HUD lines are not timestamped, so investigation targets are shown for the whole run, not on the timeline. "
                    "HUD lines are roughly one per second of INVESTIGATE mode (approximate)."}


_PER_RUN: Dict[str, Any] = {}


def recorded_metrics(pilot: str, tag: str) -> Optional[Dict[str, Any]]:
    """Row from the recorded analysis table (final only). Blank cell -> None; nothing is imputed."""
    if pilot != "final":
        return None
    # the committed, audit-corrected table; local_data/level3/final/analysis/per_run.csv is an older, stale copy
    path = os.path.join(ROOT, "docs", "experiments", "level3_final", "per_run.csv")
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return None
    if path not in _PER_RUN or _PER_RUN[path][0] != mt:
        rows = _csv(path)
        if rows is None:
            return None
        _PER_RUN[path] = (mt, {r["tag"]: r for r in rows})
    r = _PER_RUN[path][1].get(tag)
    if r is None:
        return None
    first = _f(r.get("first_true_det_s")); ign = _f(r.get("ign_first_visit_s"))
    no_det = FAMILY.get(r.get("arm")) == "baseline"   # no detector: "first true detection" does not exist, it is not censored
    return {
        "first_true_det_s": first, "first_true_det_capped_s": None if no_det else (first if first is not None else T_END),
        "first_true_det_censored": (first is None) and not no_det, "first_true_det_na": no_det,
        "ign_first_visit_s": ign, "ign_first_visit_capped_s": ign if ign is not None else T_END,
        "ign_first_visit_censored": ign is None,
        "visited_frac": _f(r.get("visited_frac")), "det_true": _f(r.get("det_true")), "det_false": _f(r.get("det_false")),
        "fire_cells": _f(r.get("fire_cells")), "fire_cells_visited": _f(r.get("fire_cells_visited")),
        "inv_true_s": _f(r.get("inv_true_s")), "inv_false_s": _f(r.get("inv_false_s")),
        "n_investigate_targets": _f(r.get("n_investigate_targets")),
        "source": "docs/experiments/level3_final/per_run.csv (eval/level3_analyze.py output, audit-corrected)",
    }


def _series(path: Optional[str], cols: List[str], missing: Dict[str, str], key: str, step_s: float = 2.0):
    """Column-oriented series, thinned to ~one row per `step_s` of run time (taking real rows, not averages)."""
    rows = _csv(path or "")
    if rows is None:
        missing[key] = f"{key} csv missing"
        return None
    out: Dict[str, List[Optional[float]]] = {c: [] for c in cols}
    nxt = -math.inf
    for r in rows:
        t = _f(r.get("t"))
        if t is None or t < nxt:
            continue
        nxt = t + step_s
        for c in cols:
            out[c].append(_f(r.get(c)))
    return out
