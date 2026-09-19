#!/usr/bin/env python3
"""
perception/base.py — the seam between the flight code and the neural network.

The agent must not know that YOLO exists. It asks a Perception object "what do
you see from here?" and gets back a list of Detection records already in grid
coordinates. That keeps three things swappable without touching swarm_agent.py:

    NullPerception     nothing is ever detected (default; this is what runs
                       today, and it reproduces the pre-Layer-3 behaviour
                       exactly)
    YoloPerception     camera -> sim/frame_synth -> detector -> sim/camera
                       projection, which you will write in perception/detector.py
    ReplayPerception   detections read from a CSV, so you can debug the
                       planner and the dashboard without a GPU in the room

Anything implementing `observe(pose) -> List[Detection]` will do; this is a
structural protocol, not a base class to inherit from.

Why projection lives behind this interface
------------------------------------------
YOLO returns pixel boxes. The planner needs cells. Somebody has to own the
transform from a bounding box plus UAV pose, altitude, and camera FOV to a
ground footprint — and that somebody is sim/camera.py, called from inside your
Perception implementation. If you let bounding boxes leak out to the agent,
the projection code ends up smeared across the control loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class Detection:
    """One detection, already projected onto the grid."""
    i: int
    j: int
    cls: str                       # "smoke" | "fire"
    conf: float                    # 0..1
    bbox: Optional[Sequence[float]] = None   # [x1,y1,x2,y2] in image pixels
    frame_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pose:
    """What a Perception implementation gets to look at. Deliberately a plain
    value object rather than the agent itself, so nothing downstream can reach
    back and command the vehicle."""
    lat: Optional[float]
    lon: Optional[float]
    alt: Optional[float]
    heading_deg: Optional[float]
    cell: Optional[Tuple[int, int]]
    t: float
    sysid: int = 0


class NullPerception:
    """Sees nothing, costs nothing. The default."""

    model_tag = "none"
    enabled = False

    def observe(self, pose: Pose) -> List[Detection]:
        return []

    def close(self) -> None:
        pass


def from_config(cfg: Optional[Dict[str, Any]], grid: Any = None) -> Any:
    """Build a Perception from the config's `perception:` section.

    Expected shape:

        perception:
          enabled: true
          weights: models/synthetic_best.pt
          model_tag: synthetic_v1
          conf_threshold: 0.45
          rate_hz: 1.0
          camera: {fov_deg: 78, tilt_deg: 30}

    Returns NullPerception when disabled, when the section is missing, or when
    the real implementation cannot be imported. That last case is deliberate:
    a missing torch install should degrade a demo to "coverage only", not crash
    four drones out of the sky mid-flight. The warning is printed once, loudly.
    """
    cfg = cfg or {}
    if not cfg.get("enabled", False):
        return NullPerception()

    try:
        from .detector import YoloPerception  # type: ignore
    except Exception as e:
        print(f"[perception] WARN: perception.enabled is true but the detector "
              f"could not be loaded ({e}). Falling back to NullPerception — "
              f"this run will have NO detections.")
        return NullPerception()

    try:
        return YoloPerception(cfg, grid=grid)
    except Exception as e:
        print(f"[perception] WARN: detector failed to initialise ({e}). "
              f"Falling back to NullPerception.")
        return NullPerception()