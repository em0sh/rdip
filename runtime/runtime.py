#!/usr/bin/env python3
"""Runtime entry point for RDIP motor control.

Supports two modes:
  * hardware (default): interfaces with a DSY-RS servo drive over Modbus to
    stream encoder feedback and send torque commands.
  * simulation (--simulate): runs the real actor against RDIPEnv for software
    tests.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure project root is on sys.path so imports work when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from io.io_dsy import DsyDrive
from rdip_env import RDIPEnv, wrap_pi
from runtime.runtime_actor import (
    EncoderInterface,
    EncoderSample,
    RuntimeActor,
    build_observation,
)


class DriveEncoder(EncoderInterface):
    """Wrap the DSY drive as an EncoderInterface source."""

    def __init__(self, drive: DsyDrive) -> None:
        self.drive = drive
        self._prev_theta = self.drive.read_position_rad()
        self._prev_time = time.monotonic()
        self._last_sample = EncoderSample(
            theta=self._prev_theta,
            alpha=0.0,
            beta=0.0,
            theta_dot=0.0,
            alpha_dot=0.0,
            beta_dot=0.0,
        )

    def read(self) -> EncoderSample:
        theta = self.drive.read_position_rad()
        now = time.monotonic()
        dt = max(now - self._prev_time, 1e-6)
        theta_dot = wrap_pi(theta - self._prev_theta) / dt
        self._prev_theta = theta
        self._prev_time = now
        self._last_sample = EncoderSample(
            theta=theta,
            alpha=0.0,
            beta=0.0,
            theta_dot=theta_dot,
            alpha_dot=0.0,
            beta_dot=0.0,
        )
        return self._last_sample

    @property
    def last_sample(self) -> EncoderSample:
        return self._last_sample

def set_cpu_affinity(core_id: int = 0) -> None:
    """Place the current process on one CPU core when supported."""
    try:
        os.sched_setaffinity(0, {core_id})
    except AttributeError:
        pass  # Not available on macOS / Windows
    except OSError:
        pass  # Affinity request rejected (ignore)


def hardware_loop(
    actor: RuntimeActor,
    drive: DsyDrive,
    encoder: DriveEncoder,
    ep_mode: int,
    loop_hz: float,
    duration: Optional[float],
    verbose: bool,
) -> None:
    loop_period = 1.0 / loop_hz
    stop_time = time.monotonic() + duration if duration else None
    latencies = []
    ticks = 0
    verbose_latencies = []
    last_step_delta = 0.0
    interval_delta = 0.0
    last_report = time.monotonic()
    prev_theta = encoder.last_sample.theta
    step_count = 0

    try:
        while True:
            start = time.monotonic()

            sample = encoder.read()
            state = sample.as_state()
            obs = build_observation(state, ep_mode)
            action, latency = actor.infer(obs)
            drive.command_acceleration(action, loop_period)

            latencies.append(latency)
            if verbose:
                verbose_latencies.append(latency)
                current_theta = sample.theta
                last_step_delta = abs(wrap_pi(current_theta - prev_theta))
                interval_delta += last_step_delta
                prev_theta = current_theta
                step_count += 1
            ticks += 1

            now = time.monotonic()

            if verbose and now - last_report >= 1.0:
                arr = np.array(verbose_latencies) if verbose_latencies else np.array([0.0])
                mean_us = arr.mean() * 1e6
                p95_us = np.percentile(arr, 95) * 1e6
                max_us = arr.max() * 1e6
                degrees = np.degrees(last_step_delta)
                interval_degrees = np.degrees(interval_delta)
                encoder_deg = np.degrees(prev_theta)
                print(
                    f"[hardware][1s] ticks={step_count} | "
                    f"Δθ_step={last_step_delta:.4f} rad ({degrees:.2f} deg) | "
                    f"Δθ_interval={interval_delta:.4f} rad ({interval_degrees:.2f} deg) | "
                    f"encoder={encoder_deg:.2f} deg | "
                    f"last_cmd={drive.last_command:.2f} rad/s² | "
                    f"latency mean={mean_us:.1f} µs | "
                    f"p95={p95_us:.1f} µs | "
                    f"max={max_us:.1f} µs"
                )
                verbose_latencies.clear()
                interval_delta = 0.0
                step_count = 0
                last_report = now

            sleep_remaining = loop_period - (now - start)
            if sleep_remaining > 0:
                time.sleep(sleep_remaining)

            if stop_time and now >= stop_time:
                break
    except KeyboardInterrupt:
        pass

    if latencies:
        arr = np.array(latencies)
        print(
            f"[hardware] ticks={ticks} | "
            f"mean={arr.mean() * 1e6:.2f} µs | "
            f"p95={np.percentile(arr, 95) * 1e6:.2f} µs | "
            f"max={arr.max() * 1e6:.2f} µs"
        )
    status = drive.status()
    print(
        f"[hardware] last command={status.last_command:.3f} rad/s² | "
        f"encoder={status.position_deg:.2f} deg"
    )


def simulation_loop(
    actor: RuntimeActor,
    ep_mode: int,
    duration: Optional[float],
    verbose: bool,
) -> None:
    env = RDIPEnv(seed=0)
    obs = env.reset(ep_mode=ep_mode)
    state = env.x.copy()

    t_end = env.T if duration is None else min(duration, env.T)
    max_steps = int(t_end / env.control_dt)
    latencies = []
    traj_actions = []
    verbose_latencies = []
    last_step_delta = 0.0
    interval_delta = 0.0
    last_report = time.monotonic()
    prev_theta: float = env.x[0]
    step_count = 0

    print(
        f"[sim] Running RDIPEnv for {max_steps} steps "
        f"(dt={env.control_dt:.3f}s, duration={max_steps * env.control_dt:.2f}s)"
    )

    for step in range(max_steps):
        action, latency = actor.infer(obs)
        latencies.append(latency)
        traj_actions.append(action)
        if verbose:
            verbose_latencies.append(latency)
        obs, _, done, _ = env.step(action)
        state = env.x.copy()

        if verbose:
            theta = state[0]
            last_step_delta = abs(wrap_pi(theta - prev_theta))
            interval_delta += last_step_delta
            prev_theta = theta
            step_count += 1

            now = time.monotonic()
            if now - last_report >= 1.0:
                arr = np.array(verbose_latencies) if verbose_latencies else np.array([0.0])
                mean_us = arr.mean() * 1e6
                p95_us = np.percentile(arr, 95) * 1e6
                max_us = arr.max() * 1e6
                degrees = np.degrees(last_step_delta)
                interval_degrees = np.degrees(interval_delta)
                print(
                    f"[sim][1s] steps={step_count} | "
                    f"Δθ_step={last_step_delta:.4f} rad ({degrees:.2f} deg) | "
                    f"Δθ_interval={interval_delta:.4f} rad ({interval_degrees:.2f} deg) | "
                    f"last_cmd={traj_actions[-1]:.2f} rad/s² | "
                    f"latency mean={mean_us:.1f} µs | "
                    f"p95={p95_us:.1f} µs | "
                    f"max={max_us:.1f} µs"
                )
                verbose_latencies.clear()
                interval_delta = 0.0
                step_count = 0
                last_report = now

        if done:
            break

    latencies = np.array(latencies)
    print(
        f"[sim] steps={len(latencies)} | "
        f"mean={latencies.mean() * 1e6:.2f} µs | "
        f"p95={np.percentile(latencies, 95) * 1e6:.2f} µs | "
        f"max={latencies.max() * 1e6:.2f} µs"
    )
    print(f"[sim] final state={state}")
    print(f"[sim] last action={traj_actions[-1]:.3f} rad/s²")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RDIP runtime control loop (hardware or simulation)."
    )
    parser.add_argument(
        "--actor",
        type=Path,
        required=True,
        help="Path to TorchScript actor (rdip_tqc_actor_*.pt).",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="Target equilibrium mode fed into the policy (default: EP3).",
    )
    parser.add_argument(
        "--loop-hz",
        type=float,
        default=100.0,
        help="Control loop frequency for hardware mode.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Optional duration in seconds (hardware or simulation).",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run against RDIPEnv instead of hardware stubs.",
    )
    parser.add_argument(
        "--no-affinity",
        action="store_true",
        help="Skip setting CPU affinity to core 0.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actor outputs (disable sampling).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print 1-second summaries (angular delta and latency stats).",
    )
    parser.add_argument(
        "--drive-port",
        type=str,
        help="Serial port for the DSY servo drive (e.g., /dev/ttyUSB0). Omit for simulation.",
    )
    parser.add_argument(
        "--drive-baud",
        type=int,
        default=115200,
        help="Modbus baud rate for the DSY drive.",
    )
    parser.add_argument(
        "--drive-unit",
        type=int,
        default=1,
        help="Modbus unit ID for the DSY drive.",
    )
    parser.add_argument(
        "--drive-counts",
        type=int,
        default=131072,
        help="Encoder counts per revolution reported by the DSY motor.",
    )
    parser.add_argument(
        "--drive-sim",
        action="store_true",
        help="Force simulated drive I/O even if a serial port is provided.",
    )
    parser.add_argument(
        "--zero",
        action="store_true",
        help="Zero the incremental encoder via the drive before starting the control loop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.no_affinity:
        set_cpu_affinity(0)

    actor = RuntimeActor(
        actor_path=args.actor,
        deterministic=args.deterministic,
        num_warmup=16,
    )

    if args.simulate:
        simulation_loop(
            actor=actor,
            ep_mode=args.ep,
            duration=args.duration,
            verbose=args.verbose,
        )
        return

    drive = DsyDrive(
        port=args.drive_port,
        baudrate=args.drive_baud,
        unit_id=args.drive_unit,
        counts_per_rev=args.drive_counts,
        simulate=args.drive_sim,
    )
    drive.connect()
    try:
        if args.zero:
            drive.zero_out()
        encoder = DriveEncoder(drive)
        hardware_loop(
            actor=actor,
            drive=drive,
            encoder=encoder,
            ep_mode=args.ep,
            loop_hz=args.loop_hz,
            duration=args.duration,
            verbose=args.verbose,
        )
    finally:
        drive.close()


if __name__ == "__main__":
    main()
