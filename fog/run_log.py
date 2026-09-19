#!/usr/bin/env python3
"""
fog/run_log.py — everything fog writes to disk.

Named run_log rather than logging so that `import logging` inside this package
unambiguously means the standard library.

Layout (unchanged from the original, plus two new files):

    runs/<YYYYmmdd-HHMMSS>-<tag>/
        coverage_t_<tag>.csv      time series: visited / traverse fractions
        fire_t_<tag>.csv          time series: burning / burnt counts
        fog_cells_<tag>.csv       per-cell: who touched it first, when
        detections_<tag>.csv      NEW: every detection, scored vs truth
        experiment_end_<tag>.txt  reason, duration, final stats
        args.json                 effective settings, for reproducibility
        manifest.json             NEW: model weights + config used
        <original config file>    copied verbatim

The manifest matters more than it looks. Layer 2 produces four numbers from
two model files, and Layer 3 runs the sim against one of them. Six months from
now "the run where synthetic did badly" is only meaningful if the run itself
records which .pt file it loaded.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence

COVERAGE_HEADER = ("t", "visited_frac", "traverse_frac", "tag", "mission", "strategy", "agents_n")
FIRE_HEADER = ("t", "burning_cnt", "burnt_cnt", "burning_frac", "burnt_frac",
               "proximity_cnt", "tag", "mission", "strategy", "agents_n")
CELLS_HEADER = ("i", "j", "visited", "visited_first_sys", "visited_first_strat", "visited_first_t",
                "visits", "last_t", "traverse_done", "tag", "mission", "strategy", "agents_n")
DETECTION_HEADER = ("t", "sys", "i", "j", "cls", "conf", "model",
                    "truth_state", "correct", "tag", "strategy")


def safe_tag(s: str) -> str:
    if not s:
        return "exp"
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in s)


class RunLogger:
    def __init__(self, tag: str, cfg_path: str = "", cfg_dir: str = ""):
        self.tag = safe_tag(tag)
        self.stamp = time.strftime("%Y%m%d-%H%M%S")
        base = cfg_dir or (os.path.dirname(os.path.abspath(cfg_path)) if cfg_path else os.getcwd())
        self.outdir = os.path.join(base, "runs", f"{self.stamp}-{self.tag}")
        os.makedirs(self.outdir, exist_ok=True)

        self.cov_path = self._p(f"coverage_t_{self.tag}.csv")
        self.fire_path = self._p(f"fire_t_{self.tag}.csv")
        self.cells_path = self._p(f"fog_cells_{self.tag}.csv")
        self.det_path = self._p(f"detections_{self.tag}.csv")
        self.end_path = self._p(f"experiment_end_{self.tag}.txt")

        self.cov_rows: List[Sequence[Any]] = [COVERAGE_HEADER]
        self.fire_rows: List[Sequence[Any]] = [FIRE_HEADER]
        self.det_rows: List[Sequence[Any]] = [DETECTION_HEADER]

        # config context, filled by attach_config()
        self.mission = ""
        self.strategy = ""
        self.agents_n = 0

        if cfg_path and os.path.exists(cfg_path):
            try:
                shutil.copyfile(cfg_path, self._p(os.path.basename(cfg_path)))
            except Exception as e:
                print(f"[fog] WARN: could not copy config into run folder: {e}")

    def _p(self, name: str) -> str:
        return os.path.join(self.outdir, name)

    # ---------------- setup ----------------

    def attach_config(self, cfg: Dict[str, Any]) -> None:
        """Pull the few config fields that get stamped into every CSV row."""
        agents = cfg.get("agents")
        if isinstance(agents, list) and agents:
            self.agents_n = len(agents)
            first = agents[0]
            if isinstance(first, dict):
                self.mission = str(first.get("mission", "") or "")
                self.strategy = str(first.get("strategy_tag", "") or "")

    def write_args(self, effective: Dict[str, Any]) -> None:
        self._dump_json(self._p("args.json"), effective, "args.json")

    def write_manifest(self, cfg: Dict[str, Any]) -> None:
        """Record which model weights this run depended on, with size and mtime
        so a silently-retrained .pt is detectable after the fact."""
        percep = cfg.get("perception", {}) if isinstance(cfg, dict) else {}
        entries = []
        for key in ("weights", "model_path", "weights_path"):
            path = percep.get(key)
            if not path:
                continue
            rec: Dict[str, Any] = {"key": key, "path": path, "exists": os.path.exists(path)}
            if rec["exists"]:
                st = os.stat(path)
                rec["size_bytes"] = st.st_size
                rec["mtime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
            entries.append(rec)
        manifest = {
            "tag": self.tag,
            "stamp": self.stamp,
            "models": entries,
            "model_tag": percep.get("model_tag", ""),
            "conf_threshold": percep.get("conf_threshold"),
            "perception_enabled": bool(percep.get("enabled", False)),
        }
        self._dump_json(self._p("manifest.json"), manifest, "manifest.json")

    # ---------------- per-tick ----------------

    def add_coverage(self, t: float, visited_frac: float, traverse_frac: float) -> None:
        self.cov_rows.append((f"{t:.2f}", f"{visited_frac:.6f}", f"{traverse_frac:.6f}",
                              self.tag, self.mission, self.strategy, str(self.agents_n)))

    def add_fire(self, t: float, burn: int, burnt: int,
                 burn_frac: float, burnt_frac: float, proximity: int) -> None:
        self.fire_rows.append((f"{t:.2f}", str(burn), str(burnt),
                               f"{burn_frac:.6f}", f"{burnt_frac:.6f}", str(proximity),
                               self.tag, self.mission, self.strategy, str(self.agents_n)))

    def add_detection(self, rec: Dict[str, Any]) -> None:
        self.det_rows.append((
            f"{rec['t']:.2f}", rec["sys"], rec["i"], rec["j"], rec["cls"],
            f"{rec['conf']:.4f}", rec["model"], rec["truth_state"], rec["correct"],
            self.tag, self.strategy,
        ))

    # ---------------- teardown ----------------

    def finish(self, fog: Any, reason: str, t_end: float, t0: float,
               visited_frac: float, traverse_frac: float) -> None:
        self._write_csv(self.cov_path, self.cov_rows)
        self._write_csv(self.fire_path, self.fire_rows)
        self._write_csv(self.det_path, self.det_rows)

        cells = [CELLS_HEADER]
        for i in range(fog.N):
            for j in range(fog.N):
                c = fog.grid[i][j]
                first_t = 0.0 if c.visited_first_t == float("inf") else round(c.visited_first_t - t0, 2)
                cells.append((i, j, int(c.visited), c.visited_first_sys, c.visited_first_strat,
                              first_t, c.visits, round(c.last_t - t0, 2) if c.last_t else 0.0,
                              int(c.done), self.tag, self.mission, self.strategy, str(self.agents_n)))
        self._write_csv(self.cells_path, cells)

        scores = fog.detection_scores()
        try:
            with open(self.end_path, "w") as f:
                f.write(f"reason={reason}\n")
                f.write(f"time_s={t_end:.2f}\n")
                f.write(f"visited_frac={visited_frac:.6f}\n")
                f.write(f"traverse_frac={traverse_frac:.6f}\n")
                f.write(f"tag={self.tag}\n")
                if self.mission:
                    f.write(f"mission={self.mission}\n")
                if self.strategy:
                    f.write(f"strategy={self.strategy}\n")
                f.write(f"agents_n={self.agents_n}\n")
                f.write(f"burnt_cnt={len(fog.burnt_cells())}\n")
                f.write(f"burnt_frac={len(fog.burnt_cells()) / (fog.N * fog.N):.6f}\n")
                for k, v in scores.items():
                    f.write(f"{k}={v}\n")
        except Exception as e:
            print(f"[fog] ERROR writing {self.end_path}: {e}")

        print(f"[fog] wrote results to {self.outdir}")

    # ---------------- internals ----------------

    @staticmethod
    def _write_csv(path: str, rows: List[Sequence[Any]]) -> None:
        try:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerows(rows)
        except Exception as e:
            print(f"[fog] ERROR writing {path}: {e}")

    @staticmethod
    def _dump_json(path: str, obj: Any, label: str) -> None:
        try:
            with open(path, "w") as f:
                json.dump(obj, f, indent=2, default=str)
        except Exception as e:
            print(f"[fog] WARN: failed writing {label}: {e}")