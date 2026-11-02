#!/usr/bin/env python3
"""
perception/frames.py — the agent-side view of the virtual camera: a pose goes out, pixels come back.

`FrameSource` is the only thing the detector knows about where images come from. The implementation below talks
HTTP to the camera service that the fog process runs (sim/camera_service.py). What crosses the wire is the UAV's
OWN pose (which the agent already has from MAVLink) and a JPEG back. There is deliberately no field, header or
endpoint that carries fire truth, and this module imports nothing from fog/ or wildfire_sim.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, Optional, Protocol

import numpy as np

from perception.base import Pose


class FrameSource(Protocol):
    def get_frame(self, pose: Pose) -> Optional[np.ndarray]:
        """uint8 H x W x 3 BGR frame for a camera at `pose`, or None if no frame is available."""
        ...


def pose_payload(pose: Pose, frame_id: str) -> Dict[str, Any]:
    """The complete request body: the vehicle's own state, nothing else."""
    return {"sysid": int(pose.sysid), "lat": pose.lat, "lon": pose.lon, "alt": pose.alt,
            "heading_deg": pose.heading_deg, "t": float(pose.t), "frame_id": frame_id}


class HttpFrameSource:
    def __init__(self, url: str = "http://127.0.0.1:8765", timeout_s: float = 60.0):
        self.url = url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._n = 0

    def get_frame(self, pose: Pose) -> Optional[np.ndarray]:
        import cv2
        self._n += 1
        body = json.dumps(pose_payload(pose, f"s{pose.sysid}-{self._n}")).encode()
        req = urllib.request.Request(self.url + "/frame", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            data = r.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return img
