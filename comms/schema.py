#!/usr/bin/env python3
"""
comms/schema.py — the wire format for everything on the multicast bus.

Why this file exists
--------------------
Before this, every process built raw dicts inline. Four message shapes were
already in use and Layer 3 adds three more. One typo in a key name ("sysid"
vs "sys") fails silently over UDP: the packet sends, nobody reads it, and you
spend an evening wondering why the dashboard is empty.

Every message carries:
    type : one of the constants below
    v    : protocol version (int)
    t    : wall-clock send time (float, time.time())

Producers should use the make_* helpers. Consumers should switch on
msg["type"] and use .get() for anything optional.

Grid convention (unchanged from the original code)
--------------------------------------------------
    i = east index, j = north index, both in [0, N)
    (0,0) is the south-west corner of the grid.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

PROTOCOL_VERSION = 1

# ---------------- message type constants ----------------
POSE = "POSE"                    # agent -> all : telemetry, every loop tick
VISIT = "VISIT"                  # agent -> all : "I have been in cell (i,j)"
MAP = "MAP"                      # agent -> all : visited/done bitmaps for merging
HAZARD_UPDATE = "HAZARD_UPDATE"  # fog   -> all : ground-truth fire (ORACLE, off by default)
DETECTION = "DETECTION"          # agent -> all : a neural-network detection  [Layer 3]
BELIEF = "BELIEF"                # agent -> all : this agent's probability map [Layer 3]
FOG_STATE = "FOG_STATE"          # fog   -> all : compact snapshot for the dashboard
EXPERIMENT_END = "EXPERIMENT_END"  # fog -> all : stop flying

ALL_TYPES = {
    POSE, VISIT, MAP, HAZARD_UPDATE,
    DETECTION, BELIEF, FOG_STATE, EXPERIMENT_END,
}

# Fire cell states, mirrored from wildfire_sim.py so consumers don't import it.
UNBURNED, BURNING, BURNT = 0, 1, 2


def _base(msg_type: str, t: Optional[float] = None) -> Dict[str, Any]:
    return {"type": msg_type, "v": PROTOCOL_VERSION, "t": float(t if t is not None else time.time())}


# ---------------- producers ----------------

def make_pose(
    *,
    sysid: int,
    name: str,
    lat: Optional[float],
    lon: Optional[float],
    alt: Optional[float] = None,
    vx: Optional[float] = None,
    vy: Optional[float] = None,
    vz: Optional[float] = None,
    heading_deg: Optional[float] = None,
    battery_pct: Optional[int] = None,
    mode: Optional[str] = None,
    mission: str = "",
    strat: str = "",
    exp: str = "",
    cell: Optional[Sequence[int]] = None,
    claim: Optional[Sequence[int]] = None,
    t: Optional[float] = None,
) -> Dict[str, Any]:
    """Agent telemetry.

    heading_deg / battery_pct / mode are new. CustomVehicle already tracks all
    three (VehicleState.heading_deg, .battery_remaining_pct, .mode), so wiring
    them into swarm_agent.state_packet() is three lines. The dashboard degrades
    gracefully to em-dashes when they are absent.
    """
    m = _base(POSE, t)
    m.update({
        "sys": int(sysid), "name": name,
        "lat": lat, "lon": lon, "alt": alt,
        "vx": vx, "vy": vy, "vz": vz,
        "hdg": heading_deg, "batt": battery_pct, "mode": mode,
        "mission": mission, "strat": strat, "exp": exp,
        "cell": list(cell) if cell else None,
        "claim": list(claim) if claim else None,
    })
    return m


def make_visit(*, sysid: int, i: int, j: int, strat: str = "", t: Optional[float] = None) -> Dict[str, Any]:
    m = _base(VISIT, t)
    m.update({"sys": int(sysid), "i": int(i), "j": int(j), "strat": strat})
    return m


def make_map(*, sysid: int, n_cells: int, vis_hex: str, done_hex: str,
             strat: str = "", cell: Optional[Sequence[int]] = None,
             t: Optional[float] = None) -> Dict[str, Any]:
    m = _base(MAP, t)
    m.update({"sys": int(sysid), "N": int(n_cells), "vis": vis_hex, "done": done_hex, "strat": strat})
    if cell:
        m["cell"] = list(cell)
    return m


def make_hazard(*, i: int, j: int, state: int, heat: float = 0.0,
                hazard: str = "fire", t: Optional[float] = None) -> Dict[str, Any]:
    """Ground-truth fire state. Only fog sends this, and only when
    fire_model.oracle_hazards is explicitly true. See fog_tracker.py."""
    m = _base(HAZARD_UPDATE, t)
    m.update({"hazard": hazard, "i": int(i), "j": int(j),
              "state": int(state), "heat": round(float(heat), 2)})
    return m


def make_detection(*, sysid: int, i: int, j: int, cls: str, conf: float,
                   model_tag: str = "", bbox: Optional[Sequence[float]] = None,
                   lat: Optional[float] = None, lon: Optional[float] = None,
                   alt: Optional[float] = None, frame_id: Optional[str] = None,
                   t: Optional[float] = None) -> Dict[str, Any]:
    """One neural-network detection, already projected from image space onto
    the grid by sim/camera.py.

    cls   : "smoke" | "fire"
    conf  : 0..1 from YOLO
    bbox  : optional [x1,y1,x2,y2] in image pixels, kept for the dashboard's
            camera view. Leave it out if you are tight on datagram budget.
    frame_id : optional key into whatever frame store the dashboard reads.
    """
    m = _base(DETECTION, t)
    m.update({"sys": int(sysid), "i": int(i), "j": int(j),
              "cls": str(cls), "conf": round(float(conf), 4),
              "model": model_tag})
    if bbox is not None:
        m["bbox"] = [round(float(b), 1) for b in bbox]
    if lat is not None:
        m["lat"] = lat
    if lon is not None:
        m["lon"] = lon
    if alt is not None:
        m["alt"] = alt
    if frame_id:
        m["frame"] = frame_id
    return m


def make_belief(*, sysid: int, n_cells: int, probs: List[List[float]],
                strat: str = "", t: Optional[float] = None) -> Dict[str, Any]:
    """This agent's current probability map, quantised to 2 decimals.

    Datagram budget: an N x N grid of "0.xx" costs roughly 5*N*N bytes. At
    N=10 that is ~500 B and fine. At N=40 it is ~8 KB, which will fragment and
    may exceed the 8192-byte receive buffer in comms/gossip.py. Above ~N=24,
    send only the cells above a probability floor as a sparse list instead.
    """
    m = _base(BELIEF, t)
    m.update({"sys": int(sysid), "N": int(n_cells), "strat": strat,
              "p": [[round(float(x), 2) for x in row] for row in probs]})
    return m


def make_fog_state(*, payload: Dict[str, Any], t: Optional[float] = None) -> Dict[str, Any]:
    """Compact whole-world snapshot, broadcast by fog at its tick rate.

    This is what makes the dashboard a pure read-only subscriber: it never
    calls into fog, so a wedged browser tab cannot slow down or crash the
    process that owns your CSVs.
    """
    m = _base(FOG_STATE, t)
    m.update(payload)
    return m


def make_experiment_end(*, reason: str, exp: str = "", t: Optional[float] = None) -> Dict[str, Any]:
    m = _base(EXPERIMENT_END, t)
    m.update({"reason": reason, "exp": exp})
    return m


# ---------------- consumer-side validation ----------------

_REQUIRED = {
    POSE: ("sys",),
    VISIT: ("sys", "i", "j"),
    MAP: ("sys", "N", "vis"),
    HAZARD_UPDATE: ("i", "j", "state"),
    DETECTION: ("sys", "i", "j", "cls", "conf"),
    BELIEF: ("sys", "N", "p"),
    FOG_STATE: ("N",),
    EXPERIMENT_END: ("reason",),
}


def is_valid(msg: Any) -> bool:
    """Cheap structural check. Returns False instead of raising, because a
    malformed packet on a shared multicast group is a routine event, not an
    error worth killing the run over."""
    if not isinstance(msg, dict):
        return False
    typ = msg.get("type")
    if typ not in ALL_TYPES:
        return False
    for key in _REQUIRED.get(typ, ()):  # type: ignore[arg-type]
        if key not in msg:
            return False
    return True


def bits_to_hex(bits: int) -> str:
    """Shared with mapping/bitmap.py so agent and fog agree on MAP encoding."""
    return format(bits, "x")


def hex_to_bits(hx: str) -> Optional[int]:
    try:
        return int(hx, 16)
    except Exception:
        return None