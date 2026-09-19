#!/usr/bin/env python3
"""
fog_tracker.py — ground station: judge, fire oracle, and recorder.

This file is now wiring only. The pieces live in:
    fog/state.py    world model (coverage, fire truth, detection scoring)
    fog/run_log.py  CSVs, manifest, run folder
    fog/config.py   config loading
    comms/gossip.py multicast bus
    comms/schema.py message shapes
    dashboard/      the live view, as its own process

What changed from the previous version, and why
-----------------------------------------------
1. THE ORACLE IS OFF BY DEFAULT.
   The old build broadcast every fire state change as HAZARD_UPDATE to every
   agent, unconditionally. Agents folded that straight into their avoid-set,
   which means they already knew exactly where the fire was. A perception
   layer bolted on top of that would change nothing, and Level 3 would be a
   demo of a system that does not need the thing it is demonstrating.

   Now it is gated on fire_model.oracle_hazards (default false). Set it true
   deliberately, to measure the perfect-information upper bound that your
   YOLO-driven runs get compared against. Fog prints a loud banner when it is
   on so the mode can never be mistaken in a screenshot.

2. Fog ingests DETECTION and scores it against truth at ingest time, writing
   detections_<tag>.csv. That file is the Level 3 result.

3. Fog broadcasts FOG_STATE instead of serving HTML. The dashboard is a
   separate read-only subscriber, so it cannot stall or crash the recorder.

4. The old --viz-port PNG/HTML server and render_png() are gone. Run
   `python3 dashboard/server.py --port 8080` instead.

Usage
-----
    python3 fog_tracker.py \\
        --origin-lat 21.2970 --origin-lon -157.8170 \\
        --grid-miles 1.0 --grid-cells 10 --traverse-frac 0.80 \\
        --gossip-group 239.255.0.1 --gossip-port 5005 \\
        --tick-rate-hz 2 --end-when visited --max-seconds 1800 \\
        --config exp_swarm_fire.yaml
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from typing import Any, Dict, Optional

# Allow `python3 fog_tracker.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comms import schema
from comms.gossip import GossipBus
from fog.config import load_config, resolve_tag
from fog.run_log import RunLogger
from fog.state import Fog
from wildfire_sim import FireSim

BANNER = "=" * 66


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fog tracker: coverage judge and fire oracle")
    ap.add_argument("--origin-lat", type=float, required=True)
    ap.add_argument("--origin-lon", type=float, required=True)
    ap.add_argument("--grid-miles", type=float, default=1.0)
    ap.add_argument("--grid-cells", type=int, default=10)
    ap.add_argument("--traverse-frac", type=float, default=0.80)
    ap.add_argument("--gossip-group", default="239.255.0.1")
    ap.add_argument("--gossip-port", type=int, default=5005)
    ap.add_argument("--tick-rate-hz", type=float, default=2.0)
    ap.add_argument(
        "--end-when",
        choices=["visited", "traverse_all", "time", "all_done"],
        default="visited",
        help="visited = first-touch coverage; traverse_all = every cell spanned "
             "edge-to-edge; all_done is a backward-compatible alias for "
             "traverse_all; time = stop at --max-seconds.",
    )
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--experiment-tag", default="")
    ap.add_argument("--config", default="", help="Path to the experiment .yaml/.json")
    ap.add_argument("--no-broadcast-state", action="store_true",
                    help="Do not emit FOG_STATE (the dashboard needs it).")
    ap.add_argument(
        "--viz-port", type=int, default=0,
        help="Deprecated and ignored. Run dashboard/server.py instead.",
    )
    return ap.parse_args()


def setup_fire(cfg: Dict[str, Any], fog: Fog) -> Optional[FireSim]:
    fire_cfg = cfg.get("fire_model", {}) if isinstance(cfg, dict) else {}
    if not fire_cfg.get("enabled", False):
        return None
    try:
        sim = FireSim.from_config(cfg, fog.N)
        fog.fire_sim = sim
        sim.ignite_many(fire_cfg.get("initial_ignitions") or [])
        fog.sync_fire_from_sim(sim)
        print(f"[fog] FireSim on: {fog.N}x{fog.N}, dt={sim.dt_seconds}s, "
              f"p_ignite={sim.p_ignite}, {sim.neighborhood}")
        return sim
    except Exception as e:
        print(f"[fog] WARN: FireSim init failed ({e}); running without fire.")
        return None


def main() -> None:
    args = parse_args()
    if args.end_when == "all_done":
        args.end_when = "traverse_all"
    if args.viz_port:
        print("[fog] NOTE: --viz-port is no longer served here. "
              "Run: python3 dashboard/server.py --port %d" % args.viz_port)

    cfg_path = os.path.abspath(args.config) if args.config else ""
    cfg = load_config(cfg_path)
    tag = resolve_tag(args.experiment_tag, cfg, cfg_path)

    # ---- world ----
    percep_cfg = cfg.get("perception", {}) if isinstance(cfg, dict) else {}
    fog = Fog(
        args.origin_lat, args.origin_lon,
        miles=args.grid_miles, cells=args.grid_cells,
        traverse_frac=args.traverse_frac,
        detection_truth_radius=int(percep_cfg.get("truth_radius_cells", 1)),
    )

    # ---- fire, and the oracle decision ----
    fire_cfg = cfg.get("fire_model", {}) if isinstance(cfg, dict) else {}
    fire_sim = setup_fire(cfg, fog)
    oracle = bool(fire_cfg.get("oracle_hazards", False))
    if oracle:
        print(BANNER)
        print("[fog] ORACLE HAZARDS ON — agents are being told exactly where the")
        print("      fire is. Perception is bypassed. This is the upper-bound")
        print("      baseline, not a perception result. Set")
        print("      fire_model.oracle_hazards: false for real Level 3 runs.")
        print(BANNER)

    # ---- recorder ----
    logger = RunLogger(tag, cfg_path=cfg_path)
    logger.attach_config(cfg)
    logger.write_args({
        "origin_lat": args.origin_lat, "origin_lon": args.origin_lon,
        "grid_miles": args.grid_miles, "grid_cells": args.grid_cells,
        "traverse_frac": args.traverse_frac,
        "gossip_group": args.gossip_group, "gossip_port": args.gossip_port,
        "tick_rate_hz": args.tick_rate_hz,
        "end_when": args.end_when, "max_seconds": args.max_seconds,
        "experiment_tag_effective": logger.tag, "stamp": logger.stamp,
        "outdir": logger.outdir, "config_path": cfg_path,
        "oracle_hazards": oracle,
        "detection_truth_radius": fog.detection_truth_radius,
    })
    logger.write_manifest(cfg)

    bus = GossipBus(args.gossip_group, args.gossip_port, name="fog")

    t0 = time.time()
    next_tick = 0.0
    next_fire_step = 0.0
    tick_period = 1.0 / max(0.1, args.tick_rate_hz)
    viz_meta = {
        "tag": logger.tag,
        "end_when": args.end_when,
        "grid_cells": fog.N,
        "oracle": oracle,
        "fire_enabled": fire_sim is not None,
    }

    stopping = {"flag": False}

    def _sigint(*_: Any) -> None:
        stopping["flag"] = True

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    def broadcast_end(reason: str) -> None:
        bus.send(schema.make_experiment_end(reason=reason, exp=logger.tag))
        print(f"[fog] >>> EXPERIMENT_END reason='{reason}'")

    def shutdown(reason: str, t_now: float, vfrac: float, tfrac: float) -> None:
        for _ in range(3):  # UDP: say it more than once
            broadcast_end(reason)
            time.sleep(0.05)
        logger.finish(fog, reason, t_now, t0, vfrac, tfrac)
        bus.close()

    print(f"[fog] started | grid {fog.N}x{fog.N} over {args.grid_miles:.2f} mi "
          f"({fog.cell_m:.0f} m cells) | end_when={args.end_when} | tag='{logger.tag}'")
    print(f"[fog] out={logger.outdir}")

    vfrac = tfrac = 0.0

    while True:
        # ---------- 1. ingest ----------
        for m in bus.recv_all():
            if not schema.is_valid(m):
                continue
            typ = m["type"]
            t_msg = float(m.get("t", time.time()))

            if typ == schema.VISIT:
                fog.ingest_visit(m.get("sys"), t_msg, int(m["i"]), int(m["j"]), m.get("strat"))

            elif typ == schema.POSE:
                lat, lon = m.get("lat"), m.get("lon")
                if lat is None or lon is None:
                    continue
                fog.ingest_pose(
                    m.get("sys"), t_msg, float(lat), float(lon),
                    m.get("cell"), m.get("strat"),
                    extra={
                        "alt": m.get("alt"), "hdg": m.get("hdg"), "batt": m.get("batt"),
                        "mode": m.get("mode"), "mission": m.get("mission"),
                        "claim": m.get("claim"), "name": m.get("name"),
                    },
                )

            elif typ == schema.DETECTION:
                rec = fog.ingest_detection(m, t_rel=time.time() - t0)
                logger.add_detection(rec)
                verdict = "HIT " if rec["correct"] else "MISS"
                print(f"[fog] DET {verdict} sys{rec['sys']} ({rec['i']},{rec['j']}) "
                      f"{rec['cls']} {rec['conf']:.2f}")

            elif typ == schema.BELIEF:
                fog.ingest_belief(m.get("sys"), m.get("p"), m.get("N", 0))

        # ---------- 2. advance fire (fog owns truth) ----------
        if fire_sim is not None and time.time() >= next_fire_step:
            changed = fire_sim.step()
            fog.sync_fire_from_sim(fire_sim)
            next_fire_step = time.time() + max(0.001, fire_sim.dt_seconds)
            if oracle:
                for i, j, state, heat in changed:
                    bus.send(schema.make_hazard(i=i, j=j, state=state, heat=heat))

        # ---------- 3. tick: record, publish, check end ----------
        now = time.time() - t0
        if now >= next_tick:
            next_tick = now + tick_period

            vfrac, tfrac, vcnt, tcnt = fog.snapshot(now)
            burn, burnt, bfrac, btfrac, prox = fog.fire_snapshot(now)
            logger.add_coverage(now, vfrac, tfrac)
            logger.add_fire(now, burn, burnt, bfrac, btfrac, prox)

            if not args.no_broadcast_state:
                bus.send(schema.make_fog_state(payload=fog.compact_state(now, viz_meta)))

            scores = fog.detection_scores()
            print(f"[fog] t={now:6.1f}s visited {vcnt}/{fog.N**2} ({100*vfrac:.1f}%) | "
                  f"traverse {tcnt}/{fog.N**2} ({100*tfrac:.1f}%) | "
                  f"fire {burn} burning, {burnt} burnt | "
                  f"det {scores['det_true']}T/{scores['det_false']}F")

            # ---- end conditions ----
            total = fog.N * fog.N
            reason = ""
            if stopping["flag"]:
                reason = "interrupted"
            elif args.end_when == "visited" and vcnt >= total:
                reason = "full_coverage_visited"
            elif args.end_when == "traverse_all" and tcnt >= total:
                reason = "full_coverage_traverse"
            elif args.end_when == "time" and args.max_seconds > 0.0 and now >= args.max_seconds:
                reason = "timeout"
            elif args.end_when != "time" and vcnt >= total:
                # safety guard: never hang once every cell is visited
                reason = "full_coverage_visited(count_guard)"
            elif args.max_seconds > 0.0 and now >= args.max_seconds:
                # hard cap regardless of mode, so a stuck drone cannot run forever
                reason = "timeout(safety_cap)"

            if reason:
                shutdown(reason, now, vfrac, tfrac)
                # Hard exit: a lingering socket thread should not keep a
                # finished experiment alive in the launcher's wait loop.
                os._exit(0)

        time.sleep(0.002)


if __name__ == "__main__":
    main()