#!/usr/bin/env python3
"""
dashboard/server.py — the live view, as a separate read-only process.

    python3 dashboard/server.py --port 8080
    # then open http://localhost:8080

Design
------
This process only ever listens. Its bus is rx_only, so it physically cannot
inject a VISIT or HAZARD packet into a run that is being recorded. If you
close the browser, kill this process, or wedge it with fifty open tabs, fog
keeps flying the experiment and keeps writing CSVs. That is the entire reason
it is not a thread inside fog_tracker.py any more.

It listens for:
    FOG_STATE  — the authoritative world snapshot, ~2 Hz
    POSE       — per-drone telemetry at full rate, for smooth movement
    DETECTION  — the AI perception feed
    EXPERIMENT_END — freezes the view and shows why it stopped

Start order does not matter. Before the first FOG_STATE arrives the page shows
"waiting for fog"; POSE packets still populate the drone panel, so you can see
your swarm is alive even if fog is not running.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # project root, for comms/

from comms import schema           # noqa: E402
from comms.gossip import GossipBus  # noqa: E402

STATIC_DIR = os.path.join(HERE, "static")
MAX_DETECTIONS = 200
STALE_AFTER_S = 10.0


class Mirror:
    """Thread-safe local copy of what the bus has said so far."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.fog: Dict[str, Any] = {}
        self.fog_seen_at: float = 0.0
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.detections: List[Dict[str, Any]] = []
        self.ended: Optional[str] = None
        self.packets = 0

    def ingest(self, m: Dict[str, Any]) -> None:
        typ = m.get("type")
        with self._lock:
            self.packets += 1

            if typ == schema.FOG_STATE:
                self.fog = m
                self.fog_seen_at = time.time()

            elif typ == schema.POSE:
                sid = str(m.get("sys", 0))
                a = self.agents.get(sid, {})
                a.update({
                    "sys": m.get("sys"), "name": m.get("name"),
                    "lat": m.get("lat"), "lon": m.get("lon"), "alt": m.get("alt"),
                    "vx": m.get("vx"), "vy": m.get("vy"), "vz": m.get("vz"),
                    "hdg": m.get("hdg"), "batt": m.get("batt"), "mode": m.get("mode"),
                    "mission": m.get("mission"), "strat": m.get("strat"),
                    "cell": m.get("cell"), "claim": m.get("claim"),
                    "last_t": m.get("t", time.time()),
                })
                self.agents[sid] = a

            elif typ == schema.DETECTION:
                self.detections.append({
                    "t": m.get("t"), "sys": m.get("sys"),
                    "i": m.get("i"), "j": m.get("j"),
                    "cls": m.get("cls"), "conf": m.get("conf"),
                    "model": m.get("model"), "bbox": m.get("bbox"),
                })
                if len(self.detections) > MAX_DETECTIONS:
                    del self.detections[:-MAX_DETECTIONS]

            elif typ == schema.EXPERIMENT_END:
                self.ended = str(m.get("reason", "ended"))

    def view(self) -> Dict[str, Any]:
        """Merge fog's authoritative snapshot with live telemetry.

        Where both have an opinion about a drone, fog wins on grid facts
        (which cell, whether a detection was correct) and POSE wins on
        telemetry (altitude, heading, battery), because POSE arrives ~10x more
        often and fog's copy of it is a tick stale."""
        now = time.time()
        with self._lock:
            fog = dict(self.fog)
            agents: Dict[str, Any] = {}
            fog_agents = fog.get("agents", {}) or {}
            for sid in set(list(fog_agents.keys()) + list(self.agents.keys())):
                merged = dict(fog_agents.get(sid, {}))
                merged.update({k: v for k, v in self.agents.get(sid, {}).items() if v is not None})
                last_t = float(merged.get("last_t") or 0.0)
                merged["age_s"] = round(now - last_t, 1) if last_t else None
                merged["stale"] = bool(last_t and (now - last_t) > STALE_AFTER_S)
                agents[sid] = merged

            return {
                "ok": True,
                "fog_connected": bool(self.fog) and (now - self.fog_seen_at) < STALE_AFTER_S,
                "fog_age_s": round(now - self.fog_seen_at, 1) if self.fog_seen_at else None,
                "world": fog,
                "agents": agents,
                "detections": self.detections[-40:],
                "ended": self.ended,
                "packets": self.packets,
                "server_time": now,
            }


def rx_loop(mirror: Mirror, group: str, port: int, stop: threading.Event) -> None:
    bus = GossipBus(group, port, rx_only=True, name="dashboard")
    print(f"[dash] listening on {group}:{port} (read-only)")
    while not stop.is_set():
        msgs = bus.recv_all()
        if not msgs:
            time.sleep(0.02)
            continue
        for m in msgs:
            if schema.is_valid(m):
                mirror.ingest(m)
    bus.close()


def make_handler(mirror: Mirror):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser navigated away mid-response

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            path = self.path.split("?", 1)[0]

            if path == "/api/state":
                self._send(200, "application/json",
                           json.dumps(mirror.view(), default=str).encode())
                return

            if path in ("/", "/index.html"):
                path = "/index.html"

            # static files, with a traversal guard
            rel = path.lstrip("/")
            full = os.path.normpath(os.path.join(STATIC_DIR, rel))
            if not full.startswith(STATIC_DIR) or not os.path.isfile(full):
                self._send(404, "text/plain", b"not found")
                return
            ctype = {
                ".html": "text/html; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".svg": "image/svg+xml",
            }.get(os.path.splitext(full)[1].lower(), "application/octet-stream")
            with open(full, "rb") as f:
                self._send(200, ctype, f.read())

        def log_message(self, *_a: Any) -> None:
            return  # keep the console for fog's output

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only wildfire monitoring dashboard")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--gossip-group", default="239.255.0.1")
    ap.add_argument("--gossip-port", type=int, default=5005)
    args = ap.parse_args()

    if not os.path.isdir(STATIC_DIR):
        print(f"[dash] ERROR: {STATIC_DIR} is missing (need index.html, app.js, style.css)")
        sys.exit(1)

    mirror = Mirror()
    stop = threading.Event()
    t = threading.Thread(target=rx_loop, args=(mirror, args.gossip_group, args.gossip_port, stop),
                         daemon=True)
    t.start()

    srv = ThreadingHTTPServer((args.host, args.port), make_handler(mirror))
    print(f"[dash] http://localhost:{args.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[dash] stopping")
    finally:
        stop.set()
        srv.server_close()


if __name__ == "__main__":
    main()