#!/usr/bin/env python3
"""
mapping/bitmap.py — the shared coverage map, as two integers.

Each drone keeps two N*N bitmaps:
    visited — a drone has been in this cell
    done    — a drone has traversed this cell edge-to-edge

They ride the bus inside MAP packets as hex strings, and merging two drones'
maps is a bitwise OR. At N=10 a whole map is 25 hex characters, so a drone can
hand its entire world model to a neighbour in one small datagram — which is
the reason the swarm works at all without a central server.

BIT ORDER — READ THIS BEFORE CHANGING ANYTHING
----------------------------------------------
    bit index = i * N + j        (i = east, j = north)

The old inline version in swarm_agent.py used `j * N + i`, the transpose of
what fog and the dashboard use. It was self-consistent, because only agents
ever read MAP packets, so nothing broke. But the moment fog or an eval script
wants to read an agent's map, a transposed grid is a bug that looks like a
plausible result: coverage that is mirrored about the diagonal still has the
right *count*, so the percentages all check out and only the picture is wrong.

This file makes i*N+j the single convention, matching fog/state.py, the
dashboard's BigInt reader, and comms/schema.py. Every agent must be on the
same build for MAP merging to be meaningful — that is true of any wire-format
change, and is why the version field exists in schema.py.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple


def bit_index(i: int, j: int, n_cells: int) -> int:
    return i * n_cells + j


def popcount(bits: int) -> int:
    return bits.bit_count() if hasattr(bits, "bit_count") else bin(bits).count("1")


class CoverageBitmap:
    """Two bitmaps plus the merge logic. No sockets, no grid geometry."""

    def __init__(self, n_cells: int):
        self.N = int(n_cells)
        self.total = self.N * self.N
        self.mask = (1 << self.total) - 1
        self.vis_bits = 0
        self.done_bits = 0
        # Cells known to be finished, by anyone. The agent unions this into its
        # avoid-set so it stops retargeting work a neighbour already did.
        self.merged_avoid: Set[Tuple[int, int]] = set()

    # ---------------- local updates ----------------

    def set_visited(self, i: int, j: int) -> None:
        if 0 <= i < self.N and 0 <= j < self.N:
            self.vis_bits |= 1 << bit_index(i, j, self.N)

    def set_done(self, i: int, j: int) -> None:
        if 0 <= i < self.N and 0 <= j < self.N:
            self.done_bits |= 1 << bit_index(i, j, self.N)
            self.merged_avoid.add((i, j))

    def is_done(self, i: int, j: int) -> bool:
        if not (0 <= i < self.N and 0 <= j < self.N):
            return False
        return bool((self.done_bits >> bit_index(i, j, self.N)) & 1)

    # ---------------- wire format ----------------

    def vis_hex(self) -> str:
        return self._to_hex(self.vis_bits)

    def done_hex(self) -> str:
        return self._to_hex(self.done_bits)

    def _to_hex(self, bits: int) -> str:
        width = (self.total + 3) // 4
        return f"{bits:0{width}x}"

    @staticmethod
    def _from_hex(hx: str) -> Optional[int]:
        try:
            return int(hx, 16)
        except (TypeError, ValueError):
            return None

    # ---------------- merging ----------------

    def merge_visited(self, hx: str) -> bool:
        """OR in a neighbour's visited map. True if anything was new."""
        other = self._from_hex(hx)
        if other is None:
            return False
        merged = self.vis_bits | (other & self.mask)
        if merged == self.vis_bits:
            return False
        self.vis_bits = merged
        return True

    def merge_done(self, hx: str) -> List[Tuple[int, int]]:
        """OR in a neighbour's done map. Returns the cells that are newly done
        from this drone's point of view, so the caller can mark them off in its
        own GridMission without rescanning the whole grid."""
        other = self._from_hex(hx)
        if other is None:
            return []
        newly = (other & self.mask) & ~self.done_bits
        if not newly:
            return []
        self.done_bits |= newly
        out: List[Tuple[int, int]] = []
        idx = 0
        while newly:
            if newly & 1:
                i, j = divmod(idx, self.N)   # inverse of i*N + j
                out.append((i, j))
                self.merged_avoid.add((i, j))
            newly >>= 1
            idx += 1
        return out

    # ---------------- endgame ----------------

    def union_bits(self) -> int:
        return (self.vis_bits | self.done_bits) & self.mask

    def single_remaining(self) -> Optional[Tuple[int, int, int]]:
        """If exactly one cell in the union is still unclaimed, return
        (i, j, bit_index); otherwise None.

        This exists because of a specific endgame failure: with 99 of 100 cells
        covered, every drone's nearest-frontier search keeps returning the same
        last cell, they all avoid it as "claimed by someone else", and the run
        stalls a few metres from completion until the timeout fires. Detecting
        the one-cell case lets the agent force the issue."""
        bits = self.union_bits()
        if popcount(bits) != self.total - 1:
            return None
        for idx in range(self.total):
            if not ((bits >> idx) & 1):
                i, j = divmod(idx, self.N)
                return (i, j, idx)
        return None

    def coverage_frac(self) -> float:
        return popcount(self.union_bits()) / self.total if self.total else 0.0