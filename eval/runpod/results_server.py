#!/usr/bin/env python3
"""Narrow, authenticated, read-only results endpoint that runs ON the pod (stdlib only).

    RESULTS_TOKEN=... python3 results_server.py --root /workspace/runs/<tag>/export --port 8000 --serve-seconds 1500

Reached through RunPod's HTTPS proxy (https://<pod_id>-8000.proxy.runpod.net). What it will serve, and nothing else:
  * only files listed in <root>/export_manifest.json (name, bytes, sha256), flat names matching [A-Za-z0-9_.-]{1,100}
  * never a directory listing, never a path outside <root>, never a symlink, never *.pt/*.pth/*.ckpt/*.bin/*.safetensors
  * every request needs the bearer token (Authorization header or ?token=), compared in constant time; wrong tokens are delayed and
    after 20 failures the server stops answering
Routes: GET /manifest, GET /status (fixed progress file status.json, <= 64 KiB), GET /f/<name>, POST /ack (marks results fetched; the server then exits so the pod can stop sooner), GET /healthz (token required).
The server exits when --serve-seconds elapsed or after /ack. It writes nothing except <root>/../results_server.log lines (no tokens).
"""
import argparse
import hashlib
import hmac
import http.server
import json
import os
import re
import sys
import threading
import time

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
BLOCKED_EXT = (".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".onnx")
MAX_FAILS = 20


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_manifest(root):
    """Validated allowlist {name: {bytes, sha256}}; raises ValueError on anything that should not be served."""
    with open(os.path.join(root, "export_manifest.json")) as f:
        m = json.load(f)
    real_root = os.path.realpath(root)
    out = {}
    for name, ent in m["files"].items():
        if not NAME_RE.match(name) or name.lower().endswith(BLOCKED_EXT) or name in ("export_manifest.json", "status.json"):
            raise ValueError("not servable: " + name)
        p = os.path.join(root, name)
        if os.path.islink(p) or not os.path.isfile(p) or os.path.dirname(os.path.realpath(p)) != real_root:
            raise ValueError("not a plain file inside root: " + name)
        if os.path.getsize(p) != ent["bytes"]:
            raise ValueError("size differs from manifest: " + name)
        out[name] = ent
    return out


def write_manifest(root, names):
    """Build export_manifest.json for plain files already copied into root (called by the job before serving)."""
    files = {}
    for n in sorted(names):
        p = os.path.join(root, n)
        files[n] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p)}
    tmp = os.path.join(root, "export_manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump({"files": files}, f, indent=1, sort_keys=True)
    os.replace(tmp, os.path.join(root, "export_manifest.json"))   # atomic: a reader never sees a half-written allowlist
    return files


STATUS_MAX = 65536


def current_allow(root):
    try:
        return load_manifest(root)
    except (ValueError, OSError, KeyError):
        return {}            # fail closed: an invalid manifest serves nothing


def make_handler(root, token, allow, state):
    tok = token.encode()

    class H(http.server.BaseHTTPRequestHandler):
        server_version = "results/1"

        def log_message(self, *a):
            pass

        def _auth(self):
            if state["fails"] >= MAX_FAILS:
                return False
            got = self.headers.get("Authorization", "")
            got = got[7:] if got.startswith("Bearer ") else ""
            if not got and "?" in self.path:
                for kv in self.path.split("?", 1)[1].split("&"):
                    if kv.startswith("token="):
                        got = kv[6:]
            if hmac.compare_digest(got.encode(), tok):
                return True
            state["fails"] += 1
            time.sleep(0.5)
            return False

        def _send(self, code, body=b"", ctype="application/octet-stream", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._auth():
                return self._send(401, b"unauthorized")
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                return self._send(200, b"ok")
            allow = current_allow(root)
            if path == "/status":      # one fixed, size-capped progress file written by the job; not part of the manifest
                sp = os.path.join(root, "status.json")
                if os.path.islink(sp) or not os.path.isfile(sp):
                    return self._send(404, b"no status yet")
                with open(sp, "rb") as f:
                    return self._send(200, f.read(STATUS_MAX), "application/json")
            if path == "/manifest":
                return self._send(200, json.dumps({"files": allow}).encode(), "application/json")
            if path.startswith("/f/"):
                name = path[3:]
                if name not in allow:          # exact allowlist match: no traversal, no globbing, no directories
                    return self._send(404, b"not found")
                p = os.path.join(root, name)
                if os.path.islink(p) or os.path.dirname(os.path.realpath(p)) != os.path.realpath(root):
                    return self._send(404, b"not found")
                size = os.path.getsize(p)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.send_header("X-Sha256", allow[name]["sha256"])
                self.end_headers()
                with open(p, "rb") as f:
                    while True:
                        b = f.read(1 << 20)
                        if not b:
                            break
                        self.wfile.write(b)
                return
            return self._send(404, b"not found")

        def do_POST(self):
            if not self._auth():
                return self._send(401, b"unauthorized")
            if self.path.split("?", 1)[0] == "/ack":
                state["acked"] = True
                return self._send(200, b"ack")
            return self._send(404, b"not found")

        def do_PUT(self):
            self._send(405, b"read-only")
        do_DELETE = do_PATCH = do_PUT

    return H


def serve(root, port, token, serve_seconds, host="0.0.0.0", poll=1.0):
    if not token or len(token) < 24:
        raise SystemExit("RESULTS_TOKEN missing or too short (>= 24 chars required)")
    allow = current_allow(root)
    state = {"acked": False, "fails": 0}
    srv = http.server.ThreadingHTTPServer((host, port), make_handler(root, token, allow, state))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    t_end = time.time() + serve_seconds
    while time.time() < t_end and not state["acked"]:
        time.sleep(poll)
    srv.shutdown()
    srv.server_close()
    return "acked" if state["acked"] else "window_elapsed"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--serve-seconds", type=int, default=1500)
    a = ap.parse_args(argv)
    res = serve(os.path.realpath(a.root), a.port, os.environ.get("RESULTS_TOKEN", ""), a.serve_seconds)
    print("results_server finished:", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
