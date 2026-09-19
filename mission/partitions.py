#!/usr/bin/env python3
"""
mission/partitions.py — who is allowed to fly where.

Pulled out of Agent.allowed_cells(). A partition is the *static* division of
labour, set once from the config; the dynamic division (claims, neighbour
visits) happens at run time in the agent. Keeping them apart matters for your
Level 2/3 write-up, because "coop vs solo" is a comms question while
"quadrants vs none" is a partition question, and a reviewer will ask you to
separate the two effects.

Schemes
-------
none       every drone may fly anywhere; coordination is purely via gossip
halves     even ids take the west half, odd ids the east half
stripes_i  interleaved columns: drone k takes every n-th column
stripes_j  interleaved rows
quadrants  the grid quartered; id 0..3 selects one

stripes_* interleave rather than block, which spreads each drone's work over
the whole area. That costs travel time but degrades gracefully: lose one drone
and you lose a thin comb of cells everywhere, not one solid quarter of the
map. Worth a sentence in the report when you compare schemes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

SCHEMES = ("none", "halves", "stripes_i", "stripes_j", "quadrants")


def allowed_cells(scheme: str, n_cells: int, part_id: int = 0,
                  part_n: int = 1) -> Optional[List[Tuple[int, int]]]:
    """Return the cells this drone owns, or None for "all of them".

    None and "every cell" are deliberately different return values: None tells
    GridMission.nearest_frontier to skip the membership test entirely, which
    matters because that test runs inside the ring search on every retarget.
    """
    scheme = (scheme or "none").lower()
    if scheme == "none":
        return None
    if scheme not in SCHEMES:
        print(f"[partitions] WARN: unknown scheme '{scheme}'; treating as 'none'")
        return None

    N = int(n_cells)
    mid = N // 2
    pid = max(0, int(part_id))
    pn = max(1, int(part_n))

    out: List[Tuple[int, int]] = []
    for i in range(N):
        for j in range(N):
            if scheme == "halves":
                keep = (i < mid) if (pid % 2 == 0) else (i >= mid)
            elif scheme == "stripes_i":
                keep = (i % pn) == (pid % pn)
            elif scheme == "stripes_j":
                keep = (j % pn) == (pid % pn)
            elif scheme == "quadrants":
                q = (0 if i < mid else 1) + (0 if j < mid else 2)
                keep = q == (pid % 4)
            else:
                keep = True
            if keep:
                out.append((i, j))

    if not out:
        # An empty partition means this drone has nothing to do and will idle
        # for the whole run — almost always a config error (e.g. partition_n
        # larger than the grid). Fail loud, fly anyway.
        print(f"[partitions] WARN: scheme='{scheme}' id={pid} n={pn} selected 0 "
              f"cells on a {N}x{N} grid; falling back to the full grid.")
        return None
    return out