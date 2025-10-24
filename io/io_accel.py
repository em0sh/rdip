#!/usr/bin/env python3
"""
Acceleration-driven control helper for the DSY-RS servo drive.

Usage examples:
    python io/io_accel.py --target 180
        → command a 180° incremental move using the drive's internal positioning.

    python io/io_accel.py --observe
        → zero the incremental encoder, then print its position every 0.25 s without moving.

This module builds on the minimal Modbus helper in io/io_dsy.py, providing a
control path that matches the rotary pendulum actor interface (angular
acceleration commands).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Optional

from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from io_dsy import DsyDrive

try:
    import minimalmodbus  # type: ignore
except ImportError:  # pragma: no cover
    minimalmodbus = None


CONTROL_DT = 0.25  # seconds between encoder polls / acceleration updates
DEFAULT_PORT = "/dev/ttyUSB0"
# Edit this constant to match your encoder's counts-per-revolution (P18.07 scaling).
# Common values: 360, 10000, 131072.  Set it once here instead of passing a flag.
COUNTS_PER_REV = 360

# Drive registers used for mode / forced DI control
P0000 = 0                # P00.00 control mode selection
P0201 = 513              # P02.01 DI1 function code
P1110 = 2826             # P11.10 Forced DI enable
P1111 = 2827             # P11.11 Forced DI mask (bit0 = DI1)
DI1_MASK = 0x0001
FUNIN_SERVO_ON = 1


def deg_to_rad(value: float) -> float:
    return value * math.pi / 180.0


def rad_to_deg(value: float) -> float:
    return value * 180.0 / math.pi


def zero_incremental_encoder(drive: DsyDrive) -> None:
    """Latch the current encoder counts as zero without waiting for settle."""
    print("Zeroing incremental encoder (instantaneous)...")
    counts = drive._read_position_counts()
    drive._zero_offset_counts = counts
    drive._command_velocity_rad = 0.0
    drive._last_command = 0.0
    drive._position_counts = counts
    print(f"Raw counts at zero: {counts:.0f}")


def print_encoder(drive: DsyDrive) -> None:
    status = drive.status()
    counts = getattr(drive, "_position_counts", float("nan"))
    offset = getattr(drive, "_zero_offset_counts", float("nan"))
    print(
        f"Encoder: {status.position_deg:+9.3f} deg "
        f"({status.position_rad:+8.4f} rad) | velocity: {status.velocity_rpm:+8.2f} rpm "
        f"| counts={counts:.0f} offset={offset:.0f}"
    )


def run_incremental_move(drive: DsyDrive, delta_deg: float) -> None:
    """Command a precise incremental position using the drive's P14.01 register."""
    drive.move_relative_degrees(delta_deg)
    time.sleep(CONTROL_DT)
    print_encoder(drive)


def _instrument(drive: DsyDrive):
    if drive.simulate:
        return None
    if minimalmodbus is None:
        raise RuntimeError("minimalmodbus is required for hardware control; please install it.")
    if drive._instrument is None:
        drive.connect()
    return drive._instrument


def _write_register(drive: DsyDrive, register: int, value: int, **kwargs) -> None:
    inst = _instrument(drive)
    if inst is None:
        return
    inst.write_register(register, value, number_of_decimals=0, functioncode=6, signed=True, **kwargs)


def servo_enable_via_forced_di(drive: DsyDrive) -> None:
    """Ensure the servo-on line is asserted using forced DI."""
    if drive.simulate:
        print("[simulate] Pretending to enable servo (forced DI).")
        return
    print("Configuring forced DI for S-ON...")
    _write_register(drive, P0201, FUNIN_SERVO_ON)
    _write_register(drive, P1110, 1)          # enable forced DI control
    _write_register(drive, P1111, DI1_MASK)   # set DI1 high → S-ON
    time.sleep(0.05)


def servo_disable_via_forced_di(drive: DsyDrive) -> None:
    if drive.simulate:
        return
    try:
        _write_register(drive, P1111, 0)
        _write_register(drive, P1110, 0)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"Warning: could not disable forced DI cleanly: {exc}")


def set_position_mode(drive: DsyDrive) -> None:
    if drive.simulate:
        print("[simulate] Pretending to set control mode to position.")
        return
    print("Setting control mode to position (P00.00 = 0)...")
    _write_register(drive, P0000, 0)
    time.sleep(0.05)


def periodic_observe(drive: DsyDrive, duration: Optional[float] = None) -> None:
    """Print encoder data every CONTROL_DT seconds. If duration is None, run indefinitely."""
    print("Observing encoder position...")
    start = time.time()
    iteration = 0
    while True:
        iteration += 1
        status = drive.status()
        print(
            f"[{iteration:04d}] encoder={status.position_deg:+9.3f} deg "
            f"vel={status.velocity_rpm:+8.2f} rpm last_cmd={status.last_command:+8.2f} rad/s²"
        )
        time.sleep(CONTROL_DT)
        if duration is not None and time.time() - start >= duration:
            break


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acceleration-based DSY drive helper.")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--unit-id", type=int, default=1)
    parser.add_argument("--simulate", action="store_true", help="Run without real hardware.")
    parser.add_argument(
        "--target",
        type=float,
        help="Target rotation in degrees (positive CCW). Omit to skip motion.",
    )
    parser.add_argument(
        "--observe",
        action="store_true",
        help="Print encoder position every 0.25 s after zeroing.",
    )
    parser.add_argument(
        "--observe-duration",
        type=float,
        help="Optional duration (seconds) for observation mode.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    drive = DsyDrive(
        port=DEFAULT_PORT,
        baudrate=args.baudrate,
        unit_id=args.unit_id,
        counts_per_rev=COUNTS_PER_REV,
        simulate=args.simulate,
    )
    drive.connect()

    try:
        set_position_mode(drive)
        servo_enable_via_forced_di(drive)
        zero_incremental_encoder(drive)
        print_encoder(drive)

        if args.target is not None:
            run_incremental_move(drive, args.target)

        if args.observe:
            periodic_observe(drive, duration=args.observe_duration)
    finally:
        servo_disable_via_forced_di(drive)
        drive.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
