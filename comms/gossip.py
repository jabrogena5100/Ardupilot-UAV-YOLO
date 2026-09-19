#!/usr/bin/env python3
"""
comms/gossip.py — one UDP multicast bus, shared by every process.

This is the class that used to live twice: as `Gossip` in swarm_agent.py and
as `GossipBus` in fog_tracker.py. Same socket setup, slightly different
method names, which is exactly the kind of drift that bites when you add a
message type to one copy and forget the other.

Usage
-----
    from comms.gossip import GossipBus

    bus = GossipBus("239.255.0.1", 5005)          # read + write
    bus = GossipBus("239.255.0.1", 5005, rx_only=True)   # dashboard

    for msg in bus.recv_all():
        ...
    bus.send(schema.make_visit(sysid=1, i=3, j=4))

Note on rx_only: it is not a security boundary, just a guard rail. The
dashboard must never be able to inject VISIT or HAZARD packets into a run
that is being recorded, so its bus refuses to transmit.
"""

from __future__ import annotations

import json
import socket
import struct
import sys
from typing import Any, Dict, List, Optional

# Datagrams above this size fragment at the IP layer. Fragmentation works on
# loopback and on a quiet LAN, but a single lost fragment drops the whole
# packet, so warn rather than silently degrade.
MTU_WARN_BYTES = 1200
RECV_BUFFER = 65535


class GossipBus:
    def __init__(
        self,
        group: str = "239.255.0.1",
        port: int = 5005,
        iface: str = "0.0.0.0",
        rx_only: bool = False,
        name: str = "bus",
    ):
        self.group, self.port = group, int(port)
        self.rx_only = bool(rx_only)
        self.name = name
        self._warned_big = False
        self._dropped = 0

        # ---- receive socket (joins the multicast group) ----
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # SO_REUSEPORT lets several of our processes bind the same port on
        # macOS/BSD. Linux is fine with SO_REUSEADDR alone; ignore if absent.
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        try:
            self.rx.bind(("", self.port))
        except OSError:
            self.rx.bind((iface, self.port))
        mreq = struct.pack("=4sl", socket.inet_aton(self.group), socket.INADDR_ANY)
        self.rx.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.rx.setblocking(False)

        # ---- send socket ----
        self.tx: Optional[socket.socket] = None
        if not self.rx_only:
            self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            self.tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, struct.pack("@i", 1))
            # Loop back to other processes on this machine (the default on
            # Linux, but set it explicitly so SITL-on-macOS behaves too).
            self.tx.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, struct.pack("@i", 1))

    # ---------------- receive ----------------
    def recv_all(self, limit: int = 256) -> List[Dict[str, Any]]:
        """Drain up to `limit` datagrams. Never blocks. Bad JSON is counted and
        discarded — see .dropped."""
        out: List[Dict[str, Any]] = []
        for _ in range(limit):
            try:
                data, _addr = self.rx.recvfrom(RECV_BUFFER)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                self._dropped += 1
                continue
            if isinstance(msg, dict):
                out.append(msg)
            else:
                self._dropped += 1
        return out

    @property
    def dropped(self) -> int:
        return self._dropped

    # ---------------- send ----------------
    def send(self, payload: Dict[str, Any]) -> bool:
        if self.tx is None:
            raise RuntimeError(f"[{self.name}] bus is rx_only; refusing to transmit {payload.get('type')}")
        try:
            blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        except Exception as e:
            print(f"[{self.name}] WARN: unserialisable payload: {e}", file=sys.stderr)
            return False
        if len(blob) > MTU_WARN_BYTES and not self._warned_big:
            self._warned_big = True
            print(
                f"[{self.name}] WARN: {payload.get('type')} is {len(blob)} B "
                f"(> {MTU_WARN_BYTES} B). It will fragment; consider a sparse "
                f"encoding. This warning prints once.",
                file=sys.stderr,
            )
        try:
            self.tx.sendto(blob, (self.group, self.port))
            return True
        except Exception:
            return False

    # ---------------- lifecycle ----------------
    def close(self) -> None:
        for sock in (self.rx, self.tx):
            try:
                if sock is not None:
                    sock.close()
            except Exception:
                pass

    def __enter__(self) -> "GossipBus":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()