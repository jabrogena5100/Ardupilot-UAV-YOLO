#!/usr/bin/env python3
"""
perception/detector.py — YOLO smoke detection from a virtual camera, projected onto the experiment grid.

    pose --FrameSource--> BGR frame --predictor--> pixel boxes --Camera.bbox_to_cell--> Detection(i, j, ...)

Three independently replaceable pieces, so each is testable alone:
  * FrameSource (perception/frames.py)   where pixels come from; here an HTTP client of the fog-side camera service.
  * predictor(frame) -> [(xyxy, conf)]   the model; Ultralytics by default, any callable in tests.
  * Camera / GridSpec (sim/camera.py)    the one pinhole projection shared with the dataset generator.

Fire truth never reaches this module: it sees only pixels for its own pose.

Config (the `perception:` section; see exp_swarm_perception.yaml):
    enabled: true
    weights: local_data/backups/boreal_v2/real_boreal_v2_best_seed0.pt   # default; never committed
    weights_sha256: <hex>        # required unless the file name is a registry v2 checkpoint (KNOWN_SHA256) or allow_unverified
    model_tag: real_boreal_v2_seed0
    conf_threshold: 0.25         # per-model value chosen on calibration scenarios (eval/level3_calibrate_detector.py); 0.25 until then
    camera: {hfov_deg: 78, pitch_deg: 60, width: 1280, height: 720, yaw_to_track: true}   # oblique, fixed; pitch = degrees below the HORIZONTAL (90 = nadir)
    frame_source: {url: http://127.0.0.1:8765, timeout_s: 60}
    cell_anchor: center          # box point projected to the ground: center (default when nadir) | bottom_center
    strict: false                # true: a failed init raises instead of falling back to NullPerception

Preprocessing is left to Ultralytics (letterbox to imgsz, normalise); the frame is uint8 BGR as it expects for numpy input.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from perception.base import Detection, Pose
from sim.camera import Camera, CameraIntrinsics, GridSpec

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join("local_data", "backups", "boreal_v2", "real_boreal_v2_best_seed0.pt")

# Registry (models/MODELS.md) hashes by file name: Model B (Boreal v2) and Model A (synth_smoke_v1) checkpoints.
KNOWN_SHA256 = {
    "synth_smoke_best_v1.pt": "1da10c4814a905d92deec2d9c36185f872eabea447d47c389b9b065468009564",
    "synth_smoke_best_v1_seed1.pt": "31c4528e37c54f779cdbfbf02d8e325877108696f3608391c0490b22ec34a454",
    "synth_smoke_best_v1_seed2.pt": "3d5a31689ab212434b206949090c948a4de0b4b653abf8d0099619529146556a",
    "real_boreal_v2_best_seed0.pt": "090c0f4cbe2718d2b084a41b8a3736fa27b673bc66b20c64672b13c83c299a05",
    "real_boreal_v2_best_seed1.pt": "5a9616cefddd3bca6eccf31d0691f7b1a2fadb572f84cf7948c85296644e9388",
    "real_boreal_v2_best_seed2.pt": "c4d443a8af919c6b00e4319b117a5dfbe90bd8b29750c75e3970a9389ae097ef",
}

Predictor = Callable[[Any], Sequence[Tuple[Sequence[float], float]]]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_weights(path: str, expected: Optional[str], allow_unverified: bool = False) -> str:
    """Return the file's SHA-256 after checking it against `expected` (or the registry by file name). Raises on a
    missing file or a mismatch; with no expected hash it raises unless allow_unverified is set."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"weights not found: {path}")
    want = (expected or KNOWN_SHA256.get(os.path.basename(path)) or "").lower()
    got = sha256_file(path)
    if want:
        if got != want:
            raise ValueError(f"weights SHA-256 mismatch for {path}: got {got}, expected {want}")
    elif not allow_unverified:
        raise ValueError(f"no expected SHA-256 for {path}: set perception.weights_sha256 or allow_unverified")
    return got


def make_ultralytics_predictor(weights: str, conf: float, imgsz: int = 640, iou: float = 0.7, max_det: int = 300,
                               threads: int = 1) -> Predictor:
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(max(1, int(threads)))
    model = YOLO(weights)

    def predict(frame):
        r = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, augment=False, device="cpu",
                          verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy().tolist()
        confs = r.boxes.conf.cpu().numpy().tolist()
        return list(zip(xyxy, confs))
    return predict


def box_cell(box: Sequence[float], camera: Camera, grid: GridSpec, anchor: str) -> Optional[Tuple[int, int]]:
    """Grid cell of a pixel box. anchor="bottom_center": the plume base of an oblique view (Camera.bbox_to_cell).
    anchor="center": the ground point under the box centre, for a nadir view, where an elevated plume appears as a
    blob around (not below) its base and the box bottom has no ground meaning. Both go through sim/camera.py."""
    if anchor == "bottom_center":
        return camera.bbox_to_cell(box, grid)
    x1, y1, x2, y2 = box
    n, e, valid = camera.unproject_ground(np.array(0.5 * (x1 + x2)), np.array(0.5 * (y1 + y2)))
    return grid.cell_of(float(n), float(e)) if bool(valid) else None


def boxes_to_detections(boxes: Sequence[Tuple[Sequence[float], float]], camera: Camera, grid: GridSpec,
                       frame_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None,
                       anchor: str = "bottom_center") -> List[Detection]:
    """Pixel boxes -> grid-cell Detections. A box whose ground anchor falls outside the grid, or above the horizon,
    is dropped (it cannot be assigned a cell). Plain floats/ints only: no tensors or arrays leave this function."""
    out: List[Detection] = []
    for xyxy, conf in boxes:
        box = [float(v) for v in xyxy]
        cell = box_cell(box, camera, grid, anchor)
        if cell is None:
            continue
        out.append(Detection(i=int(cell[0]), j=int(cell[1]), cls="smoke", conf=float(conf), bbox=box,
                             frame_id=frame_id, meta=dict(meta or {})))
    return out


class YoloPerception:
    enabled = True

    def __init__(self, cfg: Dict[str, Any], grid: Any = None, frame_source: Any = None,
                 predictor: Optional[Predictor] = None, verify: bool = True):
        self.cfg = cfg
        self.model_tag = str(cfg.get("model_tag", "yolo"))
        self.conf = float(cfg.get("conf_threshold", 0.25))
        gc = getattr(grid, "cfg", None)
        if gc is None:
            raise ValueError("YoloPerception needs the agent's GridMission (grid.cfg) to derive GridSpec and origin")
        self.lat0, self.lon0 = float(gc.lat0), float(gc.lon0)
        self.grid_spec = GridSpec.from_miles(float(gc.miles), int(gc.cells))     # i-east / j-north, as fog/state.py
        cam = cfg.get("camera") or {}
        self.hfov_deg = float(cam.get("hfov_deg", 78.0))
        self.pitch_deg = float(cam.get("pitch_deg", 60.0))                     # degrees BELOW the horizontal (90 = nadir)
        self.anchor = str(cfg.get("cell_anchor", "center" if self.pitch_deg >= 80.0 else "bottom_center"))
        self.sha = None
        weights = cfg.get("weights") or DEFAULT_WEIGHTS
        if not os.path.isabs(weights):
            weights = os.path.join(REPO_ROOT, weights)
        self.weights = weights
        if predictor is None:
            self.sha = verify_weights(weights, cfg.get("weights_sha256"), bool(cfg.get("allow_unverified", False)))
            predictor = make_ultralytics_predictor(weights, self.conf, int(cfg.get("imgsz", 640)),
                                                   float(cfg.get("nms_iou", 0.7)), int(cfg.get("max_det", 300)),
                                                   int(cfg.get("torch_threads", 1)))
        self.predictor = predictor
        if frame_source is None:
            from perception.frames import HttpFrameSource
            fs = cfg.get("frame_source") or {}
            frame_source = HttpFrameSource(fs.get("url", "http://127.0.0.1:8765"), float(fs.get("timeout_s", 60.0)))
        self.frames = frame_source
        self.latencies_ms: List[float] = []
        self.n_frames = self.n_boxes = self.n_dropped = 0
        print(f"[perception] YOLO loaded tag={self.model_tag} sha256={(self.sha or 'injected')[:12]} "
              f"conf>={self.conf} camera hfov={self.hfov_deg} pitch={self.pitch_deg} (nadir=90) weights={weights}")

    def observe(self, pose: Pose) -> List[Detection]:
        frame = self.frames.get_frame(pose)
        if frame is None:
            return []
        t0 = time.perf_counter()
        boxes = [b for b in self.predictor(frame) if float(b[1]) >= self.conf]
        dt = (time.perf_counter() - t0) * 1000.0
        self.latencies_ms.append(dt)
        h, w = frame.shape[:2]
        cam = Camera.from_pose(pose, CameraIntrinsics(w, h, self.hfov_deg), self.lat0, self.lon0, self.pitch_deg)
        fid = f"{self.model_tag}:{pose.sysid}:{self.n_frames}"
        dets = boxes_to_detections(boxes, cam, self.grid_spec, frame_id=fid, meta={"infer_ms": round(dt, 1)}, anchor=self.anchor)
        self.n_frames += 1
        self.n_boxes += len(boxes)
        self.n_dropped += len(boxes) - len(dets)
        return dets

    def close(self) -> None:
        if self.latencies_ms:
            lat = sorted(self.latencies_ms)
            print(f"[perception] {self.model_tag}: {self.n_frames} frames, {self.n_boxes} boxes "
                  f"({self.n_dropped} unmappable), infer ms median={lat[len(lat) // 2]:.0f} max={lat[-1]:.0f}")
