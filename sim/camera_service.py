#!/usr/bin/env python3
"""
sim/camera_service.py — fog-side virtual camera: serves rendered frames, never truth.

Runs inside the fog process (a daemon thread started by fog_tracker.py when `perception.camera_service.enabled`),
because fog owns the fire. It is the ONLY place where fire truth is turned into pixels:

    truth_fn() -> [SmokeSource]  --(FrameSynth.render at the requesting UAV's pose)-->  JPEG bytes

`POST /frame` takes the requesting UAV's own pose and returns a JPEG. There is no endpoint that returns sources,
cells or fire state. (A frame is a pure function of the pose, the sources and the scenario seed.)
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, List, Optional, Sequence, Tuple

import cv2

from sim.camera import GridSpec
from sim.frame_synth import FrameSynth, PlumeModel, SmokeSource, sources_from_fire_cells


class FireCellsTruth:
    """Adapter from fog's fire grid to smoke sources. Duck-typed on `fog` so this module does not import fog:
    needs `burning_cells() -> [(i, j)]` and `fire_heat[i][j]`."""

    def __init__(self, fog: Any, grid: GridSpec, plume: PlumeModel, scenario_seed: int):
        self.fog, self.grid, self.plume, self.seed = fog, grid, plume, int(scenario_seed)

    def __call__(self) -> List[SmokeSource]:
        cells = [(i, j, float(self.fog.fire_heat[i][j])) for i, j in self.fog.burning_cells()]
        return sources_from_fire_cells(cells, self.grid, self.plume, self.seed)


class FrameService:
    def __init__(self, synth: FrameSynth, truth_fn: Callable[[], Sequence[SmokeSource]],
                 host: str = "127.0.0.1", port: int = 8765, jpeg_quality: int = 92):
        self.synth, self.truth_fn, self.jpeg_quality = synth, truth_fn, int(jpeg_quality)
        self._lock = threading.Lock()                    # one render at a time (CPU bound; synth is not reentrant)
        self.frames_served = 0
        svc = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):                   # quiet
                pass

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, b'{"ok": true}', "application/json")
                else:
                    self._send(404, b"", "text/plain")

            def do_POST(self):
                if self.path != "/frame":
                    return self._send(404, b"", "text/plain")
                try:
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                    lat, lon, alt = float(req["lat"]), float(req["lon"]), float(req["alt"])
                    hdg = req.get("heading_deg")
                    key = (req.get("sysid"), req.get("frame_id"))
                except Exception as e:
                    return self._send(400, str(e).encode(), "text/plain")
                with svc._lock:
                    img = svc.synth.render(lat, lon, alt, None if hdg is None else float(hdg), list(svc.truth_fn()), key)
                    svc.frames_served += 1
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, svc.jpeg_quality])
                self._send(200, buf.tobytes(), "image/jpeg")

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer((host, int(port)), H)
        self.port = self.httpd.server_address[1]
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "FrameService":
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True, name="frame-service")
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
