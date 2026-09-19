#!/usr/bin/env python3
"""
custom_vehicle.py

A robust, dependency-minimal micro-API around pymavlink for ArduPilot/PX4.

What this module gives you
--------------------------
• Connection management:
  - Simple `CustomVehicle(conn_str)` constructor around `mavutil.mavlink_connection`.
  - Initial heartbeat wait with timeout and an optional reconnect loop.
  - Background RX thread (one per CustomVehicle) that updates a thread-safe state cache.

• Telemetry & state:
  - Thread-safe `VehicleState` dataclass with:
    * mode, armed
    * global position (lat, lon), relative & absolute altitude
    * NED velocities, attitude, heading
    * battery %, voltage, current
    * GPS fix type & satellites
    * EKF “ok” flag when available
  - Access via `get_state()` → `Dict[str, Any]` for easy logging / JSON.

• Message rate control:
  - `set_message_rate(msg_id, hz)` using `MAV_CMD_SET_MESSAGE_INTERVAL`.
  - Legacy fallback `request_data_stream()` when the modern command fails.
  - `set_default_message_rates()` requests a reasonable subset for swarm/debug:
    GLOBAL_POSITION_INT, VFR_HUD, ATTITUDE, SYS_STATUS, BATTERY_STATUS,
    GPS_RAW_INT, EKF_STATUS_REPORT, HEARTBEAT.

• Flight primitives:
  - `set_mode("GUIDED")`, `arm(True/False)`, `takeoff(alt_m)`, `land()`, `rtl()`.
  - Simple “GUIDED-style” goto helpers:
    * `goto_global(lat, lon, rel_alt_m)`
    * `goto_local_ned(x, y, z, yaw=None, yaw_rate=None)` (LOCAL_NED)

• Velocity / yaw control:
  - `send_ned_velocity(vx, vy, vz, body_frame=False, yaw=None, yaw_rate=None)`
    * LOCAL_NED or BODY_NED.
    * Position ignored, velocity used, acceleration ignored.
  - `send_body_velocity(...)` convenience wrapper (BODY_NED).
  - `condition_yaw(yaw_deg, yaw_rate_dps=0.0, relative=False)`.

• Safety helpers:
  - `healthy(...)` with checks on:
    * telemetry freshness,
    * GPS fix type,
    * battery percentage,
    * EKF status if available.
  - `wait_for_healthy(...)` to block until the above conditions become good.
  - Generic waiters:
    * `wait_for_mode(mode_name, timeout)`
    * `wait_for_armed(armed=True, timeout)`
    * `wait_for(predicate, timeout, poll_hz)`
  - `brake()` (zero velocity a few times) and `emergency_stop()` (brake + best-effort LAND).

• Simple setpoint streaming:
  - `start_velocity_stream(vx, vy, vz, body_frame=True, ..., rate_hz=10.0)`:
    spawns a small thread that re-sends velocity at the given rate.
    Returns a `stop()` function you call to terminate the stream.

Typical usage
-------------
In a script or agent:

    from custom_vehicle import CustomVehicle

    v = CustomVehicle("udp:127.0.0.1:14550", autorequest_rates=True)
    ok, issues = v.healthy()
    print("Healthy?", ok, "issues:", issues)

    v.set_mode("GUIDED"); v.wait_for_mode("GUIDED", 10)
    v.arm(True);          v.wait_for_armed(True, 10)
    v.takeoff(20)
    v.wait_for(lambda s: (s.get("alt_rel_m") or 0) > 18, timeout=30)

    # Slide forward in BODY_NED for 3 seconds
    for _ in range(30):
        v.send_body_velocity(3.0, 0.0, 0.0)
        time.sleep(0.1)

    v.rtl()
    v.close()

Note on ports & QGC
-------------------
If you want QGroundControl to see the same vehicle as your script:

• Start SITL with `--out=udp:127.0.0.1:14550` and connect QGC to UDP 14550.
• Use `CustomVehicle("udp:127.0.0.1:14550")` in scripts that are meant to show up
  in the same QGC view.

For multi-vehicle SITL, each instance usually gets its own `--out=udp:127.0.0.1:<port>`
and each agent uses the corresponding port as `conn_str`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from pymavlink import mavutil


# ========== Utilities ==========

def _now() -> float:
    """Return current wall-clock time in seconds (float)."""
    return time.time()


class VehicleError(RuntimeError):
    """Base error type for CustomVehicle-related issues (connection, modes, etc.)."""
    pass


class TimeoutError(VehicleError):
    """Raised when a blocking wait (heartbeat, mode, etc.) exceeds its timeout."""
    pass


# ========== State Model ==========

@dataclass
class VehicleState:
    """
    Thread-safe snapshot of vehicle state, maintained by the RX thread.

    NOTE:
    • All fields are optional and may start as None until messages arrive.
    • Units:
      - lat, lon: degrees
      - alt_rel_m, alt_abs_m: meters
      - vx, vy, vz: m/s in NED frame (N+, E+, D+)
      - roll_rad, pitch_rad, yaw_rad: radians
      - heading_deg: degrees (0..360, typically)
      - battery_voltage: Volts
      - battery_current: Amps
      - battery_remaining_pct: percentage (0–100) or -1 when unknown
      - gps_fix_type: ArduPilot GPS fix enum (0..6)
      - gps_satellites: integer
    """

    # Identity & mode
    armed: Optional[bool] = None
    mode: Optional[str] = None

    # Position (global & relative)
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_rel_m: Optional[float] = None
    alt_abs_m: Optional[float] = None

    # Velocity (m/s, NED)
    vx: Optional[float] = None
    vy: Optional[float] = None
    vz: Optional[float] = None

    # Orientation
    roll_rad: Optional[float] = None
    pitch_rad: Optional[float] = None
    yaw_rad: Optional[float] = None
    heading_deg: Optional[float] = None

    # Speeds
    airspeed: Optional[float] = None
    groundspeed: Optional[float] = None

    # Battery & health
    battery_voltage: Optional[float] = None
    battery_current: Optional[float] = None
    battery_remaining_pct: Optional[int] = None  # 0..100 or -1

    # GPS
    gps_fix_type: Optional[int] = None
    gps_satellites: Optional[int] = None

    # Timing
    last_msg_time: Optional[float] = None

    # EKF (best-effort; may be None on some stacks)
    ekf_ok: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow dict view of the current state (good for logging/JSON)."""
        return asdict(self)


# ========== Main Class ==========

class CustomVehicle:
    """
    Thin, ergonomic wrapper around `pymavlink.mavutil.mavlink_connection`.

    Design goals
    ------------
    • Keep the public API small and predictable (good for unit tests & swarm agents).
    • Avoid pulling in heavy dependencies (just pymavlink + stdlib).
    • Separate RX (telemetry) from TX (commands):
      - RX thread continuously updates `VehicleState`.
      - All commands (`set_mode`, `arm`, `send_body_velocity`, etc.) are plain methods.

    Threading model
    ---------------
    • One RX thread per CustomVehicle instance (`CV_RX`).
    • State cache is protected with a single lock (`_state_lock`).
    • Listener callbacks (`on(...)` / `on_any(...)`) run on the RX thread, so:
      - keep them short;
      - avoid blocking operations (sleep, network, file I/O) inside listeners.

    Typical pattern in agents
    -------------------------
    1. Construct:
         v = CustomVehicle(conn_str, autorequest_rates=True)
    2. Preflight:
         v.set_mode("GUIDED"); v.wait_for_mode("GUIDED", 10)
         v.arm(True);          v.wait_for_armed(True, 10)
         v.takeoff(alt_m)
         v.wait_for(lambda s: (s.get("alt_rel_m") or 0) > alt_m * 0.9, timeout=30)
    3. Guidance:
         v.send_body_velocity(...)
         # or start a streaming setpoint:
         stop = v.start_velocity_stream(...)
         ...
         stop()
    4. Cleanup:
         v.close()
    """

    # ---------- Construction ----------

    def __init__(
        self,
        conn_str: str,
        *,
        autorequest_rates: bool = True,
        debug: bool = False,
        heartbeat_timeout: float = 10.0,
        reconnect_retries: int = 0,
    ) -> None:
        """
        Create and connect a CustomVehicle.

        Parameters
        ----------
        conn_str : str
            MAVLink connection string, e.g.:
            • "udp:127.0.0.1:14550"
            • "udp:0.0.0.0:14550"   (listen)
            • "tcp:127.0.0.1:5760"
            • "com14"               (Windows serial)
        autorequest_rates : bool, default True
            If True, immediately request a set of high-value message streams
            (GLOBAL_POSITION_INT, VFR_HUD, ATTITUDE, etc.) via MAV_CMD_SET_MESSAGE_INTERVAL
            and legacy data streams if needed.
        debug : bool, default False
            If True, the RX loop prints every incoming message name and its to_dict().
        heartbeat_timeout : float, default 10.0
            Seconds to wait for the initial HEARTBEAT. Raises TimeoutError on failure.
        reconnect_retries : int, default 0
            If >0, attempt to reconnect this many times before giving up.

        Notes
        -----
        • Connection is established and heartbeat is awaited in the constructor.
        • The RX thread is started at the end of __init__, once we have:
            - a live connection
            - valid target_system / target_component
        """
        self._conn_str = conn_str
        self._debug = debug
        self._listeners: Dict[str, List[Callable[[Any], None]]] = {}
        self._catch_all: List[Callable[[str, Any], None]] = []

        # State cache (thread-safe)
        self._state = VehicleState()
        self._state_lock = threading.Lock()

        # Thread management
        self._stop_evt = threading.Event()
        self._rx_thread = threading.Thread(target=self._rx_loop, name="CV_RX", daemon=True)

        # Connection & heartbeat
        self._m = self._connect_with_retries(conn_str, reconnect_retries)
        self._wait_heartbeat(timeout=heartbeat_timeout)
        self.target_system = self._m.target_system
        self.target_component = self._m.target_component

        # Optionally set default message rates
        if autorequest_rates:
            self.set_default_message_rates()

        # Start reader thread last so handlers see correct rates
        self._rx_thread.start()

    # ---------- Connection helpers ----------

    def _connect_with_retries(self, conn_str: str, retries: int):
        """Internal helper: connect to MAVLink endpoint with simple retry loop."""
        attempt = 0
        last_err: Optional[Exception] = None
        while attempt <= retries:
            try:
                return mavutil.mavlink_connection(conn_str)
            except Exception as e:  # pragma: no cover (network/transport-specific)
                last_err = e
                attempt += 1
                time.sleep(0.5)
        raise VehicleError(f"Failed to connect to {conn_str}: {last_err}")

    def _wait_heartbeat(self, timeout: float) -> None:
        """Block until a HEARTBEAT is seen or timeout expires."""
        hb = self._m.wait_heartbeat(timeout=timeout)
        if hb is None:
            raise TimeoutError("Timed out waiting for heartbeat")

    # ---------- Message rate control ----------

    def set_message_rate(self, msg_id: int, hz: float) -> None:
        """
        Request a specific MAVLink message at `hz` via MAV_CMD_SET_MESSAGE_INTERVAL.

        Parameters
        ----------
        msg_id : int
            MAVLink message ID, usually from `mavutil.mavlink.MAVLINK_MSG_ID_*`.
        hz : float
            Desired rate in Hertz. Use 0 to stop the stream.
        """
        interval_us = int(1e6 / hz) if hz > 0 else 0
        self._m.mav.command_long_send(
            self.target_system,
            self.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            msg_id,
            interval_us,
            0, 0, 0, 0, 0,
        )

    def request_data_stream(self, stream_id: int, hz: float) -> None:
        """
        Legacy request_data_stream for stacks that don't support SET_MESSAGE_INTERVAL.

        Parameters
        ----------
        stream_id : int
            One of mavutil.mavlink.MAV_DATA_STREAM_* constants.
        hz : float
            Desired rate in Hertz.
        """
        self._m.mav.request_data_stream_send(
            self.target_system,
            self.target_component,
            stream_id,
            int(hz),
            1,
        )

    def set_default_message_rates(self) -> None:
        """
        Request a standard set of useful telemetry messages at reasonable rates.

        Intended for SITL/swarm debugging. Safe to call multiple times.
        """
        want = {
            "GLOBAL_POSITION_INT": 5,
            "VFR_HUD": 5,
            "ATTITUDE": 10,
            "SYS_STATUS": 1,
            "BATTERY_STATUS": 1,
            "GPS_RAW_INT": 1,
            "EKF_STATUS_REPORT": 1,
            "HEARTBEAT": 1,
        }
        for name, hz in want.items():
            msg_id = getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}", None)
            if msg_id is None:
                continue
            try:
                self.set_message_rate(msg_id, hz)
            except Exception:
                # Fallback to legacy streams if the modern command fails
                if name == "VFR_HUD":
                    self.request_data_stream(mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, hz)
                elif name in ("GLOBAL_POSITION_INT", "ATTITUDE"):
                    self.request_data_stream(mavutil.mavlink.MAV_DATA_STREAM_POSITION, hz)
                elif name in ("SYS_STATUS", "BATTERY_STATUS"):
                    self.request_data_stream(mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS, hz)

    # ---------- Listeners ----------

    def on(self, *msg_names: str) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
        """
        Decorator: register a callback for one or more MAVLink message names.

        Example
        -------
            v = CustomVehicle(...)

            @v.on("GLOBAL_POSITION_INT")
            def got_pos(msg):
                print(msg.lat, msg.lon)

        All listeners run on the RX thread.
        """
        def deco(fn: Callable[[Any], None]) -> Callable[[Any], None]:
            for name in msg_names:
                self._listeners.setdefault(name, []).append(fn)
            return fn
        return deco

    def on_any(self, fn: Callable[[str, Any], None]) -> Callable[[str, Any], None]:
        """
        Register a catch-all listener receiving (message_name, msg) for every packet.

        Example
        -------
            def log_any(name, msg):
                print(name, msg.to_dict())
            v.on_any(log_any)
        """
        self._catch_all.append(fn)
        return fn

    # ---------- State access ----------

    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot of `VehicleState` as a plain dict (thread-safe)."""
        with self._state_lock:
            return self._state.to_dict()

    # ---------- Parameters ----------

    def param_get(self, name: str, timeout: float = 3.0) -> Optional[float]:
        """
        Request a param by name and return its value, or None on timeout.

        Notes
        -----
        • Blocks until the matching PARAM_VALUE is seen or timeout passes.
        • Does not update any internal param cache; just returns the numeric value.
        """
        # Send request
        self._m.mav.param_request_read_send(self.target_system, self.target_component, name.encode(), -1)
        deadline = _now() + timeout
        while _now() < deadline:
            msg = self._m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if msg and msg.param_id.decode(errors="ignore").strip("\x00") == name:
                return float(msg.param_value)
        return None

    def param_set(
        self,
        name: str,
        value: float,
        param_type: int = None,
        ack_timeout: float = 2.0
    ) -> bool:
        """
        Set a parameter and best-effort verify via PARAM_VALUE echo.

        Returns
        -------
        bool
            True if the PARAM_VALUE echo matches the requested value (within 1e-6);
            False otherwise.
        """
        if param_type is None:
            param_type = mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        self._m.mav.param_set_send(
            self.target_system,
            self.target_component,
            name.encode(),
            float(value),
            param_type,
        )
        got = self.param_get(name, timeout=ack_timeout)
        return (got is not None) and (abs(got - float(value)) < 1e-6)

    # ---------- Flight control primitives ----------

    def set_mode(self, mode_name: str) -> None:
        """
        Set vehicle mode by name (e.g., "GUIDED", "LOITER", "AUTO").

        Raises
        ------
        VehicleError
            If the mode is unknown for this autopilot.
        """
        mapping = self._m.mode_mapping()
        if not mapping or mode_name not in mapping:
            raise VehicleError(f"Unknown mode {mode_name}; available: {sorted(mapping.keys()) if mapping else 'N/A'}")
        self._m.set_mode(mapping[mode_name])

    def wait_for_mode(self, mode_name: str, timeout: float = 5.0) -> bool:
        """Block until `get_state()["mode"] == mode_name` or timeout passes."""
        return self.wait_for(lambda st: st.get("mode") == mode_name, timeout=timeout)

    def arm(self, state: bool = True) -> None:
        """
        Arm or disarm the vehicle.

        Parameters
        ----------
        state : bool
            True to arm, False to disarm.
        """
        self._m.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1 if state else 0, 0, 0, 0, 0, 0, 0,
        )

    def wait_for_armed(self, armed: bool = True, timeout: float = 5.0) -> bool:
        """
        Block until `get_state()["armed"] == armed` or timeout passes.

        Notes
        -----
        Handles both `True/False` and None gracefully.
        """
        return self.wait_for(lambda st: st.get("armed") is armed or st.get("armed") == armed, timeout=timeout)

    def takeoff(self, alt_m: float) -> None:
        """
        Command a NAV_TAKEOFF to the given relative altitude (meters).

        NOTE: This assumes the autopilot supports GUIDED NAV_TAKEOFF semantics
        (e.g., ArduCopter in GUIDED mode). Callers should:
        • ensure mode is GUIDED,
        • ensure the vehicle is armed,
        • then call takeoff().
        """
        self._m.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, float(alt_m),
        )

    def land(self) -> None:
        """Command a NAV_LAND at the current position."""
        self._m.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0, 0, 0, 0, 0, 0, 0, 0,
        )

    def rtl(self) -> None:
        """Command a NAV_RETURN_TO_LAUNCH (RTL)."""
        self._m.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            0, 0, 0, 0, 0, 0, 0, 0,
        )

    def goto_global(self, lat: float, lon: float, rel_alt_m: float) -> None:
        """
        GUIDED-style global waypoint goto using MISSION_ITEM.

        Parameters
        ----------
        lat : float
            Target latitude in degrees.
        lon : float
            Target longitude in degrees.
        rel_alt_m : float
            Target relative altitude in meters (MAV_FRAME_GLOBAL_RELATIVE_ALT).
        """
        self._m.mav.mission_item_send(
            self.target_system, self.target_component,
            0,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            2,  # current=2 (GUIDED)
            0,
            0, 0, 0, 0,
            float(lat), float(lon), float(rel_alt_m),
        )

    def goto_local_ned(
        self,
        x: float,
        y: float,
        z: float,
        yaw: Optional[float] = None,
        yaw_rate: Optional[float] = None
    ) -> None:
        """
        Command a local NED position setpoint (x, y, z) in meters.

        Parameters
        ----------
        x, y, z : float
            Position in LOCAL_NED relative to EKF origin (NED: N+, E+, D+).
        yaw : float, optional
            Desired yaw in radians (if None, yaw is ignored).
        yaw_rate : float, optional
            Desired yaw rate in rad/s (if None, yaw_rate is ignored).

        Notes
        -----
        • This uses SET_POSITION_TARGET_LOCAL_NED with:
          - position used,
          - velocity ignored,
          - acceleration ignored.
        """
        mask = 0
        # enable position, ignore velocity/accel
        mask |= (1 << 3) | (1 << 4) | (1 << 5)  # ignore vx,vy,vz
        mask |= (1 << 6) | (1 << 7) | (1 << 8)  # ignore ax,ay,az
        if yaw is None:
            mask |= (1 << 10)
            yaw = 0.0
        if yaw_rate is None:
            mask |= (1 << 11)
            yaw_rate = 0.0
        self._m.mav.set_position_target_local_ned_send(
            0,
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            mask,
            float(x), float(y), float(z),
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            float(yaw), float(yaw_rate),
        )

    # ---------- Velocity helpers ----------

    def send_ned_velocity(
        self,
        vx: float,
        vy: float,
        vz: float,
        *,
        body_frame: bool = False,
        yaw: Optional[float] = None,
        yaw_rate: Optional[float] = None,
    ) -> None:
        """
        Send a velocity command (LOCAL_NED or BODY_NED) using SET_POSITION_TARGET_LOCAL_NED.

        Parameters
        ----------
        vx, vy, vz : float
            Velocity components in m/s. Interpretation of axes depends on frame:
            • LOCAL_NED: vx=N+, vy=E+, vz=D+.
            • BODY_NED:  vx=forward, vy=right, vz=down (body frame).
        body_frame : bool, default False
            If True, use MAV_FRAME_BODY_NED (body frame). Otherwise LOCAL_NED.
        yaw : float, optional
            Yaw in radians. Ignored if None.
        yaw_rate : float, optional
            Yaw rate in rad/s. Ignored if None.

        Behavior
        --------
        • Position is ignored (all zeros, masked off).
        • Velocity (vx,vy,vz) is used.
        • Acceleration and force are ignored.
        """
        # IGNORE pos (0..2), USE vel (3..5), IGNORE acc (6..8), IGNORE force (9)
        mask = ((1 << 0) | (1 << 1) | (1 << 2) |
                (0 << 3) | (0 << 4) | (0 << 5) |
                (1 << 6) | (1 << 7) | (1 << 8) |
                (1 << 9))
        if yaw is None:
            mask |= (1 << 10)  # IGNORE yaw
            yaw = 0.0
        if yaw_rate is None:
            mask |= (1 << 11)  # IGNORE yaw_rate
            yaw_rate = 0.0

        frame = (
            mavutil.mavlink.MAV_FRAME_BODY_NED
            if body_frame else mavutil.mavlink.MAV_FRAME_LOCAL_NED
        )

        try:
            # time_boot_ms set to 0 avoids cross-stack overflow/format issues
            self._m.mav.set_position_target_local_ned_send(
                0,
                self.target_system,
                self.target_component,
                frame,
                mask,
                0.0, 0.0, 0.0,            # position ignored
                float(vx), float(vy), float(vz),
                0.0, 0.0, 0.0,            # acceleration ignored
                float(yaw), float(yaw_rate)
            )
        except Exception as e:
            # Intentionally only warn; in swarms we don't want a single bad send to crash the process.
            print(f"[WARN] send_ned_velocity failed: {e}")

    def send_body_velocity(
        self,
        vx: float,
        vy: float,
        vz: float,
        yaw: Optional[float] = None,
        yaw_rate: Optional[float] = None
    ) -> None:
        """
        Convenience wrapper around `send_ned_velocity` with `body_frame=True`.

        Common pattern in swarm agents:
            self.v.send_body_velocity(vx, vy, vz)
        """
        self.send_ned_velocity(vx, vy, vz, body_frame=True, yaw=yaw, yaw_rate=yaw_rate)

    # ---------- Yaw helpers ----------

    def condition_yaw(
        self,
        yaw_deg: float,
        yaw_rate_dps: float = 0.0,
        relative: bool = False
    ) -> None:
        """
        ArduPilot CONDITION_YAW helper (copters/rovers).

        Parameters
        ----------
        yaw_deg : float
            Target yaw angle in degrees.
        yaw_rate_dps : float, default 0.0
            Yaw speed in degrees/second.
        relative : bool, default False
            If True, yaw_deg is interpreted as a relative offset.
        """
        self._m.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            float(yaw_deg),       # param1 yaw angle (deg)
            float(yaw_rate_dps),  # param2 yaw speed (deg/s)
            1.0 if relative else 0.0,  # param3 direction/relative flag
            0, 0, 0, 0,
        )

    # ---------- Safety & waiting ----------

    def healthy(
        self,
        min_gps_fix: int = 3,
        min_batt_pct: int = 10,
        stale_after: float = 2.0
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate basic vehicle health and return (ok, issues_list).

        Criteria
        --------
        • Telemetry freshness:
            last_msg_time within `stale_after` seconds.
        • GPS:
            gps_fix_type >= min_gps_fix (e.g., 3=3D fix).
        • Battery:
            battery_remaining_pct >= min_batt_pct (if known).
        • EKF:
            ekf_ok must not be False (None is treated as "unknown, not bad").

        Returns
        -------
        ok : bool
            True if no issues, False otherwise.
        issues : list of str
            Human-readable issue descriptions (for logs / preflight warnings).
        """
        st = self.get_state()
        issues: List[str] = []
        now = _now()
        last = st.get("last_msg_time")
        if last is None or (now - float(last)) > stale_after:
            issues.append(f"telemetry stale ({'none' if last is None else f'{now - float(last):.1f}s old'})")
        fix = st.get("gps_fix_type")
        if fix is None or int(fix) < min_gps_fix:
            issues.append(f"gps_fix<{min_gps_fix} (got {fix})")
        bpct = st.get("battery_remaining_pct")
        if bpct is not None and int(bpct) >= 0 and int(bpct) < min_batt_pct:
            issues.append(f"battery {bpct}% < {min_batt_pct}%")
        # EKF gate if known
        ekf = st.get("ekf_ok")
        if ekf is False:
            issues.append("EKF not healthy")
        return (len(issues) == 0, issues)

    def wait_for_healthy(self, timeout: float = 20.0, **kw) -> bool:
        """
        Block until `healthy(**kw)[0]` is True or timeout passes.

        Example
        -------
            if not v.wait_for_healthy(timeout=30, min_gps_fix=3):
                raise RuntimeError("Vehicle never became healthy")
        """
        deadline = _now() + timeout
        while _now() < deadline:
            ok, _ = self.healthy(**kw)
            if ok:
                return True
            time.sleep(0.5)
        return False

    def wait_for(
        self,
        predicate: Callable[[Dict[str, Any]], bool],
        timeout: float = 10.0,
        poll_hz: float = 10.0
    ) -> bool:
        """
        Generic state waiter.

        Parameters
        ----------
        predicate : callable
            Function taking a state dict and returning True when the condition is met.
        timeout : float
            Seconds to wait before giving up.
        poll_hz : float
            How often to check the predicate.

        Returns
        -------
        bool
            True if predicate became True before timeout, False otherwise.
        """
        deadline = _now() + timeout
        interval = 1.0 / max(1.0, poll_hz)
        while _now() < deadline:
            if predicate(self.get_state()):
                return True
            time.sleep(interval)
        return False

    def brake(self, repeats: int = 5, period: float = 0.1, body_frame: bool = True) -> None:
        """
        Best-effort "brake" by sending zero velocity setpoints a few times.

        Parameters
        ----------
        repeats : int, default 5
            How many zero-velocity commands to send.
        period : float, default 0.1
            Time between commands in seconds.
        body_frame : bool, default True
            Use BODY_NED by default (more intuitive: "stop relative to my current heading").
        """
        for _ in range(max(1, repeats)):
            self.send_ned_velocity(0.0, 0.0, 0.0, body_frame=body_frame)
            time.sleep(period)

    def emergency_stop(self) -> None:
        """
        Best-effort immediate stop:
        • Issue a strong brake (zero velocities multiple times).
        • Attempt a LAND as a fallback (ignore errors).
        """
        self.brake(repeats=10, period=0.05)
        try:
            self.land()
        except Exception:
            pass

    # ---------- Scheduler for streaming setpoints ----------

    class _Streamer:
        """Internal helper: small thread that calls a function at fixed rate."""
        def __init__(self, fn: Callable[[], None], rate_hz: float, stop_evt: threading.Event):
            self.fn = fn
            self.period = 1.0 / max(1.0, rate_hz)
            self.stop_evt = stop_evt
            self.t = threading.Thread(target=self._loop, name="CV_Stream", daemon=True)

        def start(self):
            self.t.start()

        def _loop(self):
            next_t = _now()
            while not self.stop_evt.is_set():
                now = _now()
                if now >= next_t:
                    try:
                        self.fn()
                    except Exception:
                        # Swallow errors; caller owns the vehicle and can log externally.
                        pass
                    next_t += self.period
                else:
                    time.sleep(min(0.01, next_t - now))

    def start_velocity_stream(
        self,
        vx: float,
        vy: float,
        vz: float,
        *,
        body_frame: bool = True,
        yaw: Optional[float] = None,
        yaw_rate: Optional[float] = None,
        rate_hz: float = 10.0
    ) -> Callable[[], None]:
        """
        Start a background thread that periodically re-sends a velocity setpoint.

        Parameters
        ----------
        vx, vy, vz : float
            Velocity components (see `send_ned_velocity` for frame conventions).
        body_frame : bool, default True
            Use BODY_NED by default (common in swarm agents).
        yaw, yaw_rate : float, optional
            Desired yaw / yaw-rate. Ignored if None.
        rate_hz : float, default 10.0
            Frequency (Hz) at which the setpoint is re-sent.

        Returns
        -------
        stop : callable
            Zero-argument function. Call it to stop the stream.

        Example
        -------
            stop = v.start_velocity_stream(3, 0, 0, body_frame=True)
            time.sleep(5)
            stop()
        """
        stop_evt = threading.Event()

        def _tick():
            self.send_ned_velocity(vx, vy, vz, yaw=yaw, yaw_rate=yaw_rate, body_frame=body_frame)

        s = CustomVehicle._Streamer(_tick, rate_hz, stop_evt)
        s.start()

        def _stop():
            stop_evt.set()
        return _stop

    # ---------- Close ----------

    def close(self) -> None:
        """
        Cleanly shut down the RX thread and close the MAVLink connection.

        Safe to call multiple times.
        """
        self._stop_evt.set()
        if self._rx_thread.is_alive():
            self._rx_thread.join(timeout=2.0)
        try:
            self._m.close()
        except Exception:
            pass

    # ---------- RX loop ----------

    def _rx_loop(self) -> None:
        """
        Main RX loop.

        Responsibilities
        ----------------
        • `recv_match()` MAVLink messages.
        • Optionally print them if debug=True.
        • Dispatch to:
          - catch-all listeners (`on_any`),
          - named listeners (`on`),
        • Update the internal `VehicleState`.
        """
        while not self._stop_evt.is_set():
            try:
                msg = self._m.recv_match(blocking=True, timeout=1)
            except Exception:
                msg = None
            if msg is None:
                continue
            name = msg.get_type()

            # Debug print
            if self._debug:
                try:
                    print(f"[RX] {name} {msg.to_dict()}")
                except Exception:
                    print(f"[RX] {name} (to_dict failed)")

            # Catch-all listeners first (receive raw)
            for fn in list(self._catch_all):
                try:
                    fn(name, msg)
                except Exception:
                    pass

            # Named listeners
            for fn in list(self._listeners.get(name, [])):
                try:
                    fn(msg)
                except Exception:
                    pass

            # Update cache
            self._update_state_from_msg(name, msg)

    # ---------- State update ----------

    def _update_state_from_msg(self, name: str, msg: Any) -> None:
        """
        Internal: update the VehicleState fields based on a MAVLink message.
        """
        with self._state_lock:
            st = self._state
            st.last_msg_time = _now()

            if name == "HEARTBEAT":
                st.armed = bool(getattr(msg, "base_mode", 0) & 0x80)
                try:
                    mode_num = getattr(msg, "custom_mode", None)
                    if mode_num is not None:
                        mapping = self._m.mode_mapping()
                        rev = {v: k for k, v in (mapping or {}).items()}
                        st.mode = rev.get(mode_num, st.mode)
                except Exception:
                    pass

            elif name == "GLOBAL_POSITION_INT":
                st.lat = msg.lat / 1e7
                st.lon = msg.lon / 1e7
                st.alt_rel_m = msg.relative_alt / 1000.0
                st.alt_abs_m = msg.alt / 1000.0
                st.vx = msg.vx / 100.0
                st.vy = msg.vy / 100.0
                st.vz = msg.vz / 100.0

            elif name == "VFR_HUD":
                st.heading_deg = getattr(msg, "heading", None)
                st.airspeed = getattr(msg, "airspeed", None)
                st.groundspeed = getattr(msg, "groundspeed", None)

            elif name == "ATTITUDE":
                st.roll_rad = getattr(msg, "roll", None)
                st.pitch_rad = getattr(msg, "pitch", None)
                st.yaw_rad = getattr(msg, "yaw", None)

            elif name == "SYS_STATUS":
                st.battery_remaining_pct = getattr(msg, "battery_remaining", None)

            elif name == "BATTERY_STATUS":
                try:
                    v_mv = msg.voltages[0] if msg.voltages and msg.voltages[0] != 0xFFFF else None
                except Exception:
                    v_mv = None
                st.battery_voltage = (v_mv / 1000.0) if v_mv is not None else st.battery_voltage
                cur_cA = getattr(msg, "current_battery", None)
                st.battery_current = (cur_cA / 100.0) if (cur_cA is not None and cur_cA != -1) else st.battery_current

            elif name == "GPS_RAW_INT":
                st.gps_fix_type = getattr(msg, "fix_type", None)
                st.gps_satellites = getattr(msg, "satellites_visible", None)

            elif name == "EKF_STATUS_REPORT":
                # Use flags to infer overall EKF OK; conservative: require velocity & pos OK bits
                try:
                    flags = int(getattr(msg, "flags", 0))
                    # Bitfield subset: 2: vel_horiz, 4: vel_vert, 8: pos_horiz_abs, 16: pos_horiz_rel
                    vel_ok = (flags & 0x2) and (flags & 0x4)
                    pos_ok = (flags & 0x8) or (flags & 0x10)
                    st.ekf_ok = bool(vel_ok and pos_ok)
                except Exception:
                    pass

    # ---------- Context manager ----------

    def __enter__(self) -> "CustomVehicle":
        """Allow `with CustomVehicle(...) as v:` usage."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """On context manager exit, cleanly close the connection."""
        self.close()


# ========== Example (optional) ==========

if __name__ == "__main__":
    # Minimal smoke test (SITL/QGC must be available on 14550)
    v = CustomVehicle("udp:127.0.0.1:14550", autorequest_rates=True, debug=False)
    try:
        ok, issues = v.healthy()
        print("Healthy:", ok, issues)
        st = v.get_state()
        print("State keys:", sorted(k for k, v_ in st.items() if v_ is not None))
    finally:
        v.close()
