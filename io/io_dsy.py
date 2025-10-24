#!/usr/bin/env python3
"""DSY-RS servo drive Modbus helper.

This module wraps the low-voltage DSY-RS servo drive so deployments can:
  * connect over RS-485 / Modbus RTU,
  * read the incremental encoder position,
  * send motion commands (speed-based by default),
  * expose CLI hooks for zeroing and spot moves during bring-up.

Registers follow the DSY-RS user manual (see COTS/DSY-RS...pdf).  Most defaults
assume position feedback via P18.07 (absolute position counter) and speed
command via P05.03 (keyboard speed command).  Adjust the register addresses if
your drive is configured differently.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

try:
    import minimalmodbus  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    minimalmodbus = None

LOGGER = logging.getLogger(__name__)


@dataclass
class DsyStatus:
    """Point-in-time snapshot of drive telemetry."""

    position_rad: float
    position_deg: float
    velocity_rpm: float
    last_command: float


def _deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi


class DsyDrive:
    """Minimal Modbus RTU interface to the DSY servo drive.

    The drive exposes most telemetry in parameter group P18 (display).  We read
    P18.07 (absolute position) and optionally P18.01 (speed feedback) while
    commanding speed via P05.03 (keyboard speed reference).  The mapping is
    configurable via constructor arguments.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        *,
        baudrate: int = 115200,
        unit_id: int = 1,
        timeout: float = 0.05,
        counts_per_rev: int = 131072,
        position_register: int = 0x1207,  # P18.07 absolute position counter
        speed_feedback_register: int = 0x1201,  # P18.01 motor speed feedback
        speed_command_register: int = 0x0503,  # P05.03 keyboard speed setpoint
        relative_move_register: int = 0x1401,  # P14.01 segment target (optional)
        speed_scale: float = 10.0,  # register units per RPM (default 0.1 rpm)
        simulate: bool = False,
    ) -> None:
        if port is None:
            simulate = True
        if simulate and port is not None:
            LOGGER.warning(
                "simulate=True with port=%s; hardware connection will be skipped.", port
            )
        if not simulate and minimalmodbus is None:
            raise RuntimeError(
                "minimalmodbus is required for hardware communication. "
                "Install it (`pip install minimalmodbus`) or set simulate=True."
            )
        self.port = port
        self.baudrate = baudrate
        self.unit_id = unit_id
        self.timeout = timeout
        self.counts_per_rev = counts_per_rev
        self.position_register = position_register
        self.speed_feedback_register = speed_feedback_register
        self.speed_command_register = speed_command_register
        self.relative_move_register = relative_move_register
        self.speed_scale = speed_scale
        self.simulate = simulate

        self._instrument: Optional["minimalmodbus.Instrument"] = None
        self._zero_offset_counts: float = 0.0
        self._last_command: float = 0.0  # rad/s²
        self._command_velocity_rad: float = 0.0
        self._position_counts: float = 0.0
        self._velocity_rpm: float = 0.0
        self._simulate_velocity: float = 0.0
        self._simulate_position: float = 0.0
        self._last_read_time: float = time.monotonic()

    # ---------------------------------------------------------------------- #
    # Connection helpers
    # ---------------------------------------------------------------------- #
    def connect(self) -> None:
        if self.simulate:
            LOGGER.info("DSY drive running in simulation mode.")
            return
        if self._instrument is not None:
            return
        instrument = minimalmodbus.Instrument(self.port, self.unit_id)
        instrument.serial.baudrate = self.baudrate
        instrument.serial.timeout = self.timeout
        instrument.mode = minimalmodbus.MODE_RTU
        instrument.clear_buffers_before_each_transaction = True
        self._instrument = instrument
        LOGGER.info(
            "Connected to DSY drive on %s (baud=%d, unit=%d)",
            self.port,
            self.baudrate,
            self.unit_id,
        )

    def close(self) -> None:
        if self._instrument:
            try:
                self._instrument.serial.close()
            except Exception:  # pragma: no cover - best effort
                pass
            self._instrument = None

    # ---------------------------------------------------------------------- #
    # Core telemetry
    # ---------------------------------------------------------------------- #
    def _read_position_counts(self) -> float:
        if self.simulate:
            return self._simulate_position * self.counts_per_rev / (2.0 * math.pi)
        assert self._instrument is not None, "Drive not connected"
        raw = self._instrument.read_long(
            self.position_register, functioncode=3, signed=False
        )
        # Normalize to unsigned 32-bit space to avoid wrap sign flips.
        return float(raw & 0xFFFFFFFF)

    def _read_speed_rpm(self) -> float:
        if self.simulate:
            return self._simulate_velocity * 60.0 / (2.0 * math.pi)
        assert self._instrument is not None, "Drive not connected"
        raw = self._instrument.read_long(
            self.speed_feedback_register, functioncode=3, signed=True
        )
        # Register is typically 0.1 rpm units -> divide by 10
        return raw / self.speed_scale

    def read_position_rad(self) -> float:
        counts = self._read_position_counts()
        counts -= self._zero_offset_counts
        if counts > 0x80000000:
            counts -= 0x100000000
        self._position_counts = counts
        angle = counts * (2.0 * math.pi) / self.counts_per_rev
        return angle

    def read_position_deg(self) -> float:
        return _rad_to_deg(self.read_position_rad())

    def read_velocity_rpm(self) -> float:
        rpm = self._read_speed_rpm()
        self._velocity_rpm = rpm
        return rpm

    @property
    def last_command(self) -> float:
        return self._last_command

    def status(self) -> DsyStatus:
        pos_rad = self.read_position_rad()
        vel_rpm = self.read_velocity_rpm()
        return DsyStatus(
            position_rad=pos_rad,
            position_deg=_rad_to_deg(pos_rad),
            velocity_rpm=vel_rpm,
            last_command=self._last_command,
        )

    # ---------------------------------------------------------------------- #
    # Control primitives
    # ---------------------------------------------------------------------- #
    def command_acceleration(self, accel_rad_s2: float, dt: float) -> None:
        """Integrate acceleration and update the drive speed command."""
        self._last_command = accel_rad_s2
        self._command_velocity_rad += accel_rad_s2 * dt
        target_rpm = self._command_velocity_rad * 60.0 / (2.0 * math.pi)
        target_reg = int(round(target_rpm * self.speed_scale))

        if self.simulate:
            self._simulate_velocity = self._command_velocity_rad
            self._simulate_position += self._simulate_velocity * dt
            return

        assert self._instrument is not None, "Drive not connected"
        self._instrument.write_register(
            self.speed_command_register,
            target_reg,
            number_of_decimals=0,
            signed=True,
        )

    def move_relative_degrees(self, delta_deg: float) -> None:
        """Jog the motor by a relative angle (best effort implementation)."""
        delta_rad = _deg_to_rad(delta_deg)
        if self.simulate:
            self._simulate_position += delta_rad
            return
        counts_delta = int(round(delta_rad * self.counts_per_rev / (2.0 * math.pi)))
        assert self._instrument is not None, "Drive not connected"
        low = counts_delta & 0xFFFF
        high = (counts_delta >> 16) & 0xFFFF
        payload = [low, high]
        self._instrument.write_registers(self.relative_move_register, payload)

    # ---------------------------------------------------------------------- #
    # Homing / zeroing
    # ---------------------------------------------------------------------- #
    def zero_out(
        self,
        *,
        settle_time: float = 0.5,
        tolerance_deg: float = 0.02,
        timeout: float = 10.0,
        poll_rate_hz: float = 50.0,
    ) -> float:
        """Wait for the motor to settle, then treat the current encoder value as zero."""
        tolerance_rad = _deg_to_rad(tolerance_deg)
        dt = 1.0 / poll_rate_hz
        start = time.time()
        stable_time = 0.0
        previous = self.read_position_rad()
        while True:
            time.sleep(dt)
            current = self.read_position_rad()
            if abs(current - previous) < tolerance_rad:
                stable_time += dt
            else:
                stable_time = 0.0
            previous = current
            if stable_time >= settle_time:
                break
            if time.time() - start > timeout:
                LOGGER.warning("Zero-out timed out; using last position as zero.")
                break
        counts = self._read_position_counts()
        self._zero_offset_counts = counts
        LOGGER.info("Encoder zeroed at raw counts %.1f", counts)
        return counts

    # ---------------------------------------------------------------------- #
    # CLI helpers
    # ---------------------------------------------------------------------- #
    def _ensure_connected(self) -> None:
        if self.simulate:
            return
        if self._instrument is None:
            self.connect()

    def cli_zero(self) -> None:
        self._ensure_connected()
        self.zero_out()
        status = self.status()
        print(
            f"Zero complete. position={status.position_deg:.4f} deg, velocity={status.velocity_rpm:.3f} rpm"
        )

    def cli_move(self, delta_deg: float) -> None:
        self._ensure_connected()
        self.move_relative_degrees(delta_deg)
        print(f"Move command issued: Δθ={delta_deg:.2f} deg")

    def cli_status(self) -> None:
        self._ensure_connected()
        status = self.status()
        print(
            f"Position: {status.position_deg:.4f} deg ({status.position_rad:.4f} rad)\n"
            f"Velocity: {status.velocity_rpm:.3f} rpm\n"
            f"Last accel command: {status.last_command:.3f} rad/s²"
        )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DSY-RS servo drive helper.")
    parser.add_argument("--port", help="Serial port for the drive (e.g., /dev/ttyUSB0).")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--unit-id", type=int, default=1)
    parser.add_argument(
        "--counts-per-rev", type=int, default=131072, help="Encoder counts per revolution."
    )
    parser.add_argument(
        "--simulate", action="store_true", help="Force simulation even if a port is provided."
    )
    parser.add_argument("--zero", action="store_true", help="Zero the encoder on startup.")
    parser.add_argument(
        "--motormove",
        type=float,
        help="Issue a relative move (degrees). Positive = CCW in motor frame.",
    )
    parser.add_argument(
        "--status", action="store_true", help="Print the current drive status after actions."
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = _build_argparser()
    args = parser.parse_args(argv)

    drive = DsyDrive(
        port=args.port,
        baudrate=args.baudrate,
        unit_id=args.unit_id,
        counts_per_rev=args.counts_per_rev,
        simulate=args.simulate,
    )
    drive.connect()

    if args.zero:
        drive.cli_zero()
    if args.motormove is not None:
        drive.cli_move(args.motormove)
        # Give the simulated drive a moment to integrate
        time.sleep(0.1)
    if args.status or (not args.zero and args.motormove is None):
        drive.cli_status()

    drive.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
