#!/usr/bin/env python3
"""
dashboard/replay_server.py — read-only browser replay of SAVED Level 3 pilot runs.

    python3 dashboard/replay_server.py --port 8081          # http://<vm-ip>:8081
    ~/venv-dronekit/bin/python dashboard/replay_build.py    # once, to cache UAV paths from the tlogs

This is separate from dashboard/server.py (the live gossip-bus view). It does no
simulation, loads no model, opens no UDP socket and never writes into a results
directory: each request reads a handful of small CSV/JSON files. Memory is a few MB,
so it is fine alongside (or instead of) the live dashboard on a 1 vCPU / 2 GB VM.
Static assets reuse dashboard/static/style.css.

    GET /api/runs           run list: completed / interrupted / not-run / discarded
    GET /api/run?id=<id>    everything for one run (id from /api/runs)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from dashboard import replay_data as rd  # noqa: E402

STATIC_DIR = os.path.join(HERE, "static")
CTYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8"}


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
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, "application/json", json.dumps(obj, default=str, allow_nan=False).encode())

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        if u.path == "/api/runs":
            self._json(rd.discover())
            return
        if u.path == "/api/run":
            rid = (parse_qs(u.query).get("id") or [""])[0]
            data = rd.load_run(rid)
            if data is None:
                self._json({"error": f"unknown run id {rid!r}"}, 404)
            else:
                self._json(data)
            return
        rel = "replay.html" if u.path in ("/", "/index.html") else u.path.lstrip("/")
        full = os.path.realpath(os.path.join(STATIC_DIR, rel))
        if not full.startswith(STATIC_DIR + os.sep) or not os.path.isfile(full):
            self._send(404, "text/plain", b"not found")
            return
        with open(full, "rb") as f:
            self._send(200, CTYPES.get(os.path.splitext(full)[1].lower(), "application/octet-stream"), f.read())

    def log_message(self, *_a) -> None:
        return


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only replay dashboard for saved Level 3 runs")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 so the host machine can reach the VM; use 127.0.0.1 with an SSH tunnel")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[replay] serving saved runs from {rd.L3}")
    print(f"[replay] http://{a.host if a.host != '0.0.0.0' else 'localhost'}:{a.port}   (read-only; Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[replay] stopping")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
