#!/usr/bin/env python3
"""
mission/geo.py — the handful of geometry functions everything shares.

Extracted from swarm_agent.py so that grid_mission, the agent, and anything
in planning/ or sim/ all measure distance the same way. Three copies of a
projection is three chances to disagree about where cell (4,5) is.

Frames
------
NE metres, origin at the experiment origin: +n is north, +e is east.
Grid indices: i = east, j = north, (0,0) = south-west corner.

Note on the projection: this uses cos(lat0), the origin's latitude, rather
than cos of the mean of the two latitudes. fog/state.py uses the mean-latitude
form. Over a 1-mile box at Hawaii's latitude the two differ by well under a
centimetre, so they are interchangeable here — but this file keeps the exact
formula the flight code has always used, because changing the projection under
a working controller is not the kind of thing you do during a refactor.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

R_EARTH = 6378137.0
M_PER_MILE = 1609.344


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def latlon_to_ne(lat_deg: float, lon_deg: float,
                 lat0_deg: float, lon0_deg: float) -> Tuple[float, float]:
    """Degrees -> (north_m, east_m) relative to the origin."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    dn = (lat - lat0) * R_EARTH
    de = (lon - lon0) * R_EARTH * math.cos(lat0)
    return dn, de


def ne_to_latlon(n_m: float, e_m: float,
                 lat0_deg: float, lon0_deg: float) -> Tuple[float, float]:
    """Exact inverse of latlon_to_ne (same cos(lat0) form)."""
    lat0 = math.radians(lat0_deg)
    lat = lat0_deg + math.degrees(n_m / R_EARTH)
    lon = lon0_deg + math.degrees(e_m / (R_EARTH * math.cos(lat0)))
    return lat, lon


def cell_index(n: float, e: float, half_side_m: float, cell_m: float, N: int) -> Optional[Tuple[int, int]]:
    """Metres-from-origin -> (i, j), or None if outside the grid."""
    x = e + half_side_m
    y = n + half_side_m
    if x < 0.0 or y < 0.0 or x >= 2 * half_side_m or y >= 2 * half_side_m:
        return None
    i = int(x // cell_m)
    j = int(y // cell_m)
    if 0 <= i < N and 0 <= j < N:
        return (i, j)
    return None


def track_yaw(vx_n: float, vy_e: float, last: Optional[float], min_speed: float = 2.0) -> Optional[float]:
    """Yaw (rad, compass bearing: 0 = north, pi/2 = east) that points the vehicle along its NED velocity; below min_speed keep `last`."""
    return math.atan2(vy_e, vx_n) if math.hypot(vx_n, vy_e) >= min_speed else last


def cheby_dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Chebyshev (chessboard-king) distance in grid cells. This is the metric
    comm_grid_radius is expressed in: radius 2 means a 5x5 neighbourhood."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))