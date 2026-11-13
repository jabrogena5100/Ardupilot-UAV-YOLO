#!/usr/bin/env python3
"""
demo.py - one command to open the Level 3 replay demonstration in a browser.

    python3 demo.py                 # http://localhost:8081
    python3 demo.py --port 9000     # another port
    python3 demo.py --host 0.0.0.0  # reachable from outside a VM (VirtualBox port forwarding)
    python3 demo.py --no-browser    # do not try to open a browser window

This is a REPLAY of already-recorded experiments. It starts no ArduPilot SITL, loads no neural network and
writes nothing to any results directory.

Two data sources, chosen automatically (override with --source):
  full      dashboard/replay_server.py reads the raw run files under local_data/level3/final/ (gitignored; the team's
            machine or a data bundle has them). Everything is served live from those files.
  snapshot  docs/demo/level3_final_replay.html, a single self-contained file committed to git (same 56 runs, same
            viewer, data inlined). Works on a fresh clone with only the Python standard library.
"""
import argparse
import functools
import os
import socket
import subprocess
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(ROOT, "docs", "demo")
SNAP = "level3_final_replay.html"
RAW = os.path.join(ROOT, "local_data", "level3", "final", "results")
CACHE = os.path.join(ROOT, "local_data", "level3", "_replay_cache")


def have_full() -> bool:
    return os.path.isdir(RAW) and os.path.isdir(CACHE) and len(os.listdir(RAW)) >= 56


def port_free(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/" + SNAP
        super().do_GET()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach the page from outside a VM")
    ap.add_argument("--source", choices=["auto", "full", "snapshot"], default="auto")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    src = a.source
    if src == "auto":
        src = "full" if have_full() else "snapshot"
    if src == "full" and not have_full():
        print(f"[demo] --source full needs the raw runs in {RAW} and the path cache in {CACHE}; "
              "they are not in git. Use --source snapshot.", file=sys.stderr)
        return 2
    if src == "snapshot" and not os.path.exists(os.path.join(SNAP_DIR, SNAP)):
        print(f"[demo] {SNAP_DIR}/{SNAP} is missing; is this a complete checkout (git status)?", file=sys.stderr)
        return 2
    if not port_free(a.host, a.port):
        print(f"[demo] port {a.port} is already in use. Try: python3 demo.py --port {a.port + 1}", file=sys.stderr)
        return 2

    shown = "localhost" if a.host in ("127.0.0.1", "0.0.0.0") else a.host
    url = f"http://{shown}:{a.port}"
    print(f"[demo] REPLAY of recorded Level 3 runs (not a live simulation); source: {src}", flush=True)
    print(f"[demo] open {url}   (Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    if src == "full":
        try:
            return subprocess.call([sys.executable, os.path.join(ROOT, "dashboard", "replay_server.py"),
                                    "--host", a.host, "--port", str(a.port)])
        except KeyboardInterrupt:  # Ctrl+C also reaches the child, which prints its own "stopping"
            return 0
    srv = ThreadingHTTPServer((a.host, a.port), functools.partial(Quiet, directory=SNAP_DIR))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[demo] stopped")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
