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
from typing import Tuple

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


def cheby_dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Chebyshev (chessboard-king) distance in grid cells. This is the metric
    comm_grid_radius is expressed in: radius 2 means a 5x5 neighbourhood."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))