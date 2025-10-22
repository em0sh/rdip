#!/usr/bin/env python3
"""Measure controller inference latency for the RDIP motor control stack."""

import argparse
import csv
import math
import os
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from rdip_env import RDIPEnv


def _is_interactive_backend(name: str) -> bool:
    """Return True if the Matplotlib backend name supports interactive windows."""
    if not name:
        return False
    lower = name.lower()
    if lower.startswith("module://"):
        lower = lower.split("module://", 1)[1]
    interactive_aliases = {
        "tkagg",
        "qt5agg",
        "qt6agg",
        "qtagg",
        "gtk3agg",
        "gtk4agg",
        "wxagg",
        "macosx",
        "webagg",
        "nbagg",
    }
    if lower in interactive_aliases:
        return True
    if lower.endswith("agg") and lower not in interactive_aliases:
        return False
    return lower in interactive_aliases


class LatencyBuffer:
    """Thread-safe buffer for storing per-step latency samples (µs)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: List[float] = []

    def append(self, value: float) -> None:
        with self._lock:
            self._data.append(value)

    def snapshot(self) -> np.ndarray:
        with self._lock:
            return np.array(self._data, dtype=np.float64)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def _project_pendulum(theta: float, alpha: float, beta: float, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D pendulum configuration into 2D coordinates for plotting."""
    L1 = params["L1"]
    l1 = params["l1"]
    l2 = params["l2"]

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    base = np.array([0.0, 0.0, 0.0])
    arm_tip = np.array([L1 * cos_t, L1 * sin_t, 0.0])

    alpha_total = alpha
    beta_total = alpha + beta

    first_tip = arm_tip + np.array(
        [l1 * math.sin(alpha_total) * cos_t,
         l1 * math.sin(alpha_total) * sin_t,
         l1 * math.cos(alpha_total)]
    )
    second_tip = first_tip + np.array(
        [l2 * math.sin(beta_total) * cos_t,
         l2 * math.sin(beta_total) * sin_t,
         l2 * math.cos(beta_total)]
    )

    yaw = math.radians(30.0)
    c, s = math.cos(yaw), math.sin(yaw)

    def project(point):
        x, y, z = point
        x_proj = c * x - s * y
        return x_proj, z

    points = np.stack([base, arm_tip, first_tip, second_tip], axis=0)
    xz = np.array([project(p) for p in points])
    return xz[:, 0], xz[:, 1]


class LatencyPlotter:
    """Main-thread Matplotlib visualizer for latency samples."""

    def __init__(self, buffer: LatencyBuffer, interval: float = 1.0):
        self._buffer = buffer
        self._interval = interval
        self._plt = None
        self.fig = None
        self.ax_pend = None
        self.ax_latency = None
        self._latency_line = None
        self._stats_text = None
        self._pend_line = None
        self._pend_text = None
        self._last_refresh = 0.0

    def start(self) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(f"matplotlib unavailable: {exc}") from exc

        self._plt = plt
        self._plt.ion()
        self.fig, (self.ax_pend, self.ax_latency) = self._plt.subplots(
            1, 2, figsize=(12, 6), num="RDIP Controller Latency"
        )
        self._plt.subplots_adjust(wspace=0.3)

        self.ax_pend.set_title("RDIP (projected)")
        self.ax_pend.set_aspect("equal", adjustable="box")
        self.ax_pend.set_xlabel("Projected horizontal (m)")
        self.ax_pend.set_ylabel("Vertical (m)")
        self._pend_line, = self.ax_pend.plot([], [], "-o", lw=2)
        self._pend_text = self.ax_pend.text(
            0.5, 0.5, "Final pose will display after run", transform=self.ax_pend.transAxes,
            ha="center", va="center"
        )

        self.ax_latency.set_title("Controller latency")
        self.ax_latency.set_xlabel("Step")
        self.ax_latency.set_ylabel("Latency (µs)")
        self.ax_latency.grid(True, linestyle="--", alpha=0.3)
        self._latency_line, = self.ax_latency.plot([], [], lw=1.6)
        self._stats_text = self.ax_latency.text(0.02, 0.98, "", transform=self.ax_latency.transAxes, va="top")

        self.fig.canvas.draw_idle()
        self._plt.show(block=False)
        self._last_refresh = 0.0

    def refresh(self, force: bool = False) -> None:
        if not self.fig:
            return
        now = time.time()
        if not force and now - self._last_refresh < self._interval:
            return
        samples = self._buffer.snapshot()
        if samples.size:
            xs = np.arange(samples.size)
            self._latency_line.set_data(xs, samples)
            self.ax_latency.relim()
            self.ax_latency.autoscale_view()
            self.ax_latency.set_ylim(bottom=0.0)
            mean_val = samples.mean()
            p95_val = np.percentile(samples, 95)
            max_val = samples.max()
            self._stats_text.set_text(
                f"mean: {mean_val:.1f} µs\n"
                f"p95:  {p95_val:.1f} µs\n"
                f"max:  {max_val:.1f} µs"
            )
        self.fig.canvas.draw_idle()
        self._last_refresh = now

    def pause(self, dt: float = 0.001) -> None:
        if self._plt:
            self._plt.pause(dt)

    def show_final_state(self, state: np.ndarray, params: dict, label: str) -> None:
        if not self.fig:
            return
        theta, alpha, beta = state[:3]
        xs, ys = _project_pendulum(theta, alpha, beta, params)
        self._pend_line.set_data(xs, ys)
        pad = 0.05
        self.ax_pend.set_xlim(xs.min() - pad, xs.max() + pad)
        self.ax_pend.set_ylim(ys.min() - pad, ys.max() + pad)
        if self._pend_text:
            self._pend_text.set_text(label)
            self._pend_text.set_position((0.02, 0.95))
            self._pend_text.set_ha("left")
            self._pend_text.set_va("top")
        self.fig.canvas.draw_idle()

    def close(self) -> None:
        if not self.fig or not self._plt:
            return
        self.refresh(force=True)
        self._plt.ioff()
        self._plt.close(self.fig)
        self.fig = None

    def hold_open(self) -> None:
        if not self.fig or not self._plt:
            return
        self.refresh(force=True)
        self._plt.ioff()
        fig_num = self.fig.number
        while self._plt.fignum_exists(fig_num):
            self._plt.pause(0.1)
        self.fig = None


def _load_actor(actor_path: Path) -> torch.jit.ScriptModule:
    actor = torch.jit.load(str(actor_path))
    actor.eval()
    actor.to("cpu")
    return actor


def _step_actor(actor, obs: np.ndarray, deterministic: bool) -> Tuple[float, float]:
    tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        start_ns = time.perf_counter_ns()
        out = actor(tensor, deterministic=deterministic, with_logprob=False)
        end_ns = time.perf_counter_ns()
    if isinstance(out, (tuple, list)):
        action_tensor = out[0]
    else:
        action_tensor = out
    action = float(action_tensor.squeeze().cpu().numpy())
    latency_us = (end_ns - start_ns) / 1000.0
    return action, latency_us


def _choose_unit(latencies_us: np.ndarray) -> Tuple[float, str]:
    if not latencies_us.size:
        return 1.0, "µs"
    median_us = float(np.median(latencies_us))
    if median_us >= 1000.0:
        return 1.0 / 1000.0, "ms"
    return 1.0, "µs"


def _print_summary(latencies_us: np.ndarray) -> None:
    scale, unit = _choose_unit(latencies_us)
    scaled = latencies_us * scale
    mean_val = float(np.mean(scaled))
    median_val = float(np.median(scaled))
    p95_val = float(np.percentile(scaled, 95))
    max_val = float(np.max(scaled))
    std_val = float(np.std(scaled))
    print(
        f"\nController latency across {latencies_us.size} steps:\n"
        f"  mean   : {mean_val:.3f} {unit}\n"
        f"  median : {median_val:.3f} {unit}\n"
        f"  p95    : {p95_val:.3f} {unit}\n"
        f"  max    : {max_val:.3f} {unit}\n"
        f"  stddev : {std_val:.3f} {unit}"
    )


def profile_controller(
    actor_path: Path,
    episodes: int,
    steps_per_episode: int,
    deterministic: bool,
    seed: Optional[int],
    warmup: int,
    csv_path: Optional[Path],
    enable_plot: bool,
    plot_interval: float,
) -> None:
    env = RDIPEnv(seed=seed)
    actor = _load_actor(actor_path)
    buffer = LatencyBuffer()

    if episodes != 1:
        print(f"[info] Parameter episodes={episodes} ignored; running a single EP3 episode.")

    csv_file = None
    csv_writer = None
    if csv_path:
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["episode", "step", "latency_us", "action"])

    plotter: Optional[LatencyPlotter] = None
    if enable_plot:
        try:
            import matplotlib
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[plot] Disabled (matplotlib unavailable: {exc})")
        else:
            backend_name = matplotlib.get_backend() or ""
            backend_lower = backend_name.lower()
            interactive = _is_interactive_backend(backend_lower)
            if not interactive:
                try:
                    from matplotlib.backends import BackendFilter, backend_registry
                except Exception:
                    interactive_names = set()
                else:
                    interactive_names = {
                        name.lower()
                        for name in backend_registry.list_builtin(BackendFilter.INTERACTIVE)
                    }
                if backend_lower in interactive_names and not (
                    backend_lower.endswith("agg") and not _is_interactive_backend(backend_lower)
                ):
                    interactive = True
            if not interactive:
                display_hint = ""
                if os.name != "nt" and not os.environ.get("DISPLAY"):
                    display_hint = " (DISPLAY is unset; enable X forwarding or a virtual display)"
                print(
                    f"[plot] Disabled: backend '{backend_name}' cannot open windows{display_hint}. "
                    "Set an interactive backend (e.g., MPLBACKEND=TkAgg or MPLBACKEND=QtAgg) "
                    "and ensure the GUI toolkit is installed, or run with --no-plot."
                )
            else:
                if backend_lower == "tkagg":
                    try:
                        import tkinter  # noqa: F401
                    except ModuleNotFoundError:
                        print(
                            "[plot] Disabled: Tk backend requested but Tk/Tcl bindings are missing "
                            "(ModuleNotFoundError: _tkinter). Install Tk support or choose another backend."
                        )
                    else:
                        plotter = LatencyPlotter(buffer, interval=plot_interval)
                else:
                    plotter = LatencyPlotter(buffer, interval=plot_interval)

                if plotter:
                    try:
                        plotter.start()
                    except RuntimeError as exc:
                        print(f"[plot] Disabled ({exc})")
                        plotter = None

    worker_errors: List[BaseException] = []
    final_state: Optional[np.ndarray] = None
    final_time = 0.0
    total_steps_recorded = 0

    def measurement_loop() -> None:
        nonlocal csv_file, csv_writer, final_state, final_time, total_steps_recorded
        try:
            target_mode = 3
            obs = env.reset(ep_mode=target_mode)

            for _ in range(max(warmup, 0)):
                _step_actor(actor, obs, deterministic)

            print(
                f"[info] Running EP3 latency test for up to {steps_per_episode} step(s)."
            )

            done = False
            for step_idx in range(steps_per_episode):
                action, latency_us = _step_actor(actor, obs, deterministic)
                buffer.append(latency_us)
                if csv_writer:
                    csv_writer.writerow([0, step_idx, f"{latency_us:.3f}", f"{action:.6f}"])
                obs, _, done, _ = env.step(action)
                total_steps_recorded = step_idx + 1

                if done:
                    break
        except BaseException as exc:  # pragma: no cover - ensure visibility in main thread
            worker_errors.append(exc)
        finally:
            if csv_file:
                csv_file.close()
                csv_file = None
                csv_writer = None
            final_state = env.x.copy()
            final_time = env.t

    worker = threading.Thread(target=measurement_loop, name="latency-worker", daemon=True)
    worker.start()

    try:
        while worker.is_alive():
            worker.join(timeout=0.1)
            if plotter:
                plotter.refresh()
                plotter.pause(0.001)
    finally:
        worker.join()

    if worker_errors:
        raise worker_errors[0]

    if plotter:
        plotter.refresh(force=True)
        if final_state is not None:
            plotter.show_final_state(final_state, env.p, label="Final EP3 pose")
        print("[plot] Close the window to finish.")
        plotter.hold_open()

    print(
        f"[info] Completed {total_steps_recorded} step(s); final t={final_time:.3f}s."
    )

    latencies = buffer.snapshot()
    if latencies.size == 0:
        print("No controller steps recorded.")
        return
    _print_summary(latencies)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile controller response time for RDIP motor control."
    )
    parser.add_argument(
        "--actor",
        type=Path,
        required=True,
        help="Path to TorchScript actor (e.g., rdip_tqc_actor.pt).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Compatibility flag (ignored). Profiler always runs a single EP3 episode.",
    )
    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=10_000,
        help="Number of controller steps to execute (caps the profiling horizon).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actor outputs (skip sampling).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for environment randomization.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Initial actor evaluations to discard from timing statistics.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write per-step latency samples as CSV.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable background latency plot.",
    )
    parser.add_argument(
        "--plot-interval",
        type=float,
        default=1.0,
        help="Seconds between plot refreshes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_controller(
        actor_path=args.actor,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        deterministic=args.deterministic,
        seed=args.seed,
        warmup=args.warmup,
        csv_path=args.csv,
        enable_plot=not args.no_plot,
        plot_interval=args.plot_interval,
    )


if __name__ == "__main__":
    main()
