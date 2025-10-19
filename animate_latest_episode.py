import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def load_episode(path: Path):
    """Load a saved rollout (.npz) and normalize optional fields."""
    data = np.load(path, allow_pickle=True)
    states = data["state"]
    times = data["time"]
    actions = data["action"]
    rewards = data["reward"]
    mode = int(data["mode"])
    episode_idx = int(data["episode"])
    control_dt = float(data["control_dt"])
    if "params" in data:
        params = data["params"]
        if isinstance(params, np.ndarray):
            params = params[0]
    else:
        params = {}
    return dict(
        states=states,
        times=times,
        actions=actions,
        rewards=rewards,
        mode=mode,
        control_dt=control_dt,
        episode=episode_idx,
        params=params,
    )


def wrap_pi_array(arr):
    """Vectorized angle wrapping helper."""
    return (arr + np.pi) % (2 * np.pi) - np.pi


def compute_geometry(states, params):
    """Convert state trajectory into projected coordinates for plotting."""
    L1 = params.get("L1", 0.16)
    l1 = params.get("l1", 0.07)
    l2 = params.get("l2", 0.13)

    theta = wrap_pi_array(states[:, 0])
    alpha_raw = wrap_pi_array(states[:, 1])
    beta = wrap_pi_array(states[:, 2])

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # 3D positions for each link endpoint
    base = np.zeros((len(theta), 3))
    arm_tip = np.stack([L1 * cos_t, L1 * sin_t, np.zeros_like(theta)], axis=1)

    # First link: offset from the rotating arm by alpha.
    horiz1 = l1 * np.sin(alpha_raw)
    vert1 = l1 * np.cos(alpha_raw)
    first_tip = arm_tip + np.stack([horiz1 * cos_t, horiz1 * sin_t, vert1], axis=1)

    # Second link angle includes alpha + beta as defined in the paper.
    total_second = alpha_raw + beta
    horiz2 = l2 * np.sin(total_second)
    vert2 = l2 * np.cos(total_second)
    second_tip = first_tip + np.stack([horiz2 * cos_t, horiz2 * sin_t, vert2], axis=1)

    # Simple oblique projection to 2D (rotate around vertical axis)
    # Fixed oblique viewing angle for the simple 3D→2D projection.
    view_yaw = np.deg2rad(30)
    c, s = np.cos(view_yaw), np.sin(view_yaw)

    def project(points):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        x_proj = c * x - s * y
        return x_proj, z

    base_s, base_z = project(base)
    arm_s, arm_z = project(arm_tip)
    first_s, first_z = project(first_tip)
    second_s, second_z = project(second_tip)

    s_coords = np.stack([base_s, arm_s, first_s, second_s], axis=1)
    z_coords = np.stack([base_z, arm_z, first_z, second_z], axis=1)

    # Shift alpha by 90° so the trace aligns with the paper's visualizations.
    alpha_plot = wrap_pi_array(alpha_raw + (np.pi / 2.0))

    return s_coords, z_coords, theta, alpha_plot, beta


def animate_episode(path: Path):
    """Animate a single episode using Matplotlib's FuncAnimation."""
    info = load_episode(path)
    states = info["states"]
    if len(states) == 0:
        print(f"{path} contains no steps to animate.")
        return

    times = info["times"]
    control_dt = info["control_dt"]
    params = info["params"]
    mode = info["mode"]
    episode_idx = info["episode"]

    s_coords, z_coords, theta, alpha, beta = compute_geometry(states, params)

    fig, (ax_pend, ax_angles) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Episode {episode_idx} | Mode EP{mode}")

    # Pendulum view
    ax_pend.set_title("Pendulum (side view)")
    s_min = np.min(s_coords) - 0.1
    s_max = np.max(s_coords) + 0.1
    z_min = np.min(z_coords) - 0.1
    z_max = np.max(z_coords) + 0.1
    ax_pend.set_xlim(s_min, s_max)
    ax_pend.set_ylim(z_min, z_max)
    ax_pend.set_xlabel("Projected horizontal (m)")
    ax_pend.set_ylabel("Vertical (m)")
    pend_line, = ax_pend.plot([], [], "-o", lw=2)
    time_text = ax_pend.text(0.02, 0.95, "", transform=ax_pend.transAxes)

    # Angle traces
    ax_angles.set_title("Angles")
    ax_angles.set_xlim(times[0], times[-1])
    ax_angles.set_ylim(-np.pi - 0.5, np.pi + 0.5)
    ax_angles.set_xlabel("Time (s)")
    ax_angles.set_ylabel("Angle (rad)")
    line_theta, = ax_angles.plot([], [], label=r"$\theta$")
    line_alpha, = ax_angles.plot([], [], label=r"$\alpha$")
    line_beta, = ax_angles.plot([], [], label=r"$\beta$")
    ax_angles.legend(loc="upper right")
    time_marker = ax_angles.axvline(times[0], color="k", linestyle="--", alpha=0.5)

    def init():
        pend_line.set_data([], [])
        line_theta.set_data([], [])
        line_alpha.set_data([], [])
        line_beta.set_data([], [])
        time_text.set_text("")
        time_marker.set_xdata([times[0], times[0]])
        return pend_line, line_theta, line_alpha, line_beta, time_marker, time_text

    def update(idx):
        pend_line.set_data(s_coords[idx], z_coords[idx])
        line_theta.set_data(times[: idx + 1], theta[: idx + 1])
        line_alpha.set_data(times[: idx + 1], alpha[: idx + 1])
        line_beta.set_data(times[: idx + 1], beta[: idx + 1])
        time_marker.set_xdata([times[idx], times[idx]])
        time_text.set_text(f"t = {times[idx]:.2f} s")
        return pend_line, line_theta, line_alpha, line_beta, time_marker, time_text

    interval_ms = max(int(control_dt * 1000), 1)
    anim = FuncAnimation(
        fig,
        update,
        frames=len(times),
        init_func=init,
        interval=interval_ms,
        blit=True,
        repeat=True,
    )
    # Keep a reference to prevent garbage collection
    setattr(fig, "_anim", anim)
    plt.show()


def wait_for_new_file(path: Path, last_mtime: float, poll: float = 1.0):
    """Block until the file timestamp changes."""
    while True:
        time.sleep(poll)
        if not path.exists():
            continue
        mtime = path.stat().st_mtime
        if mtime > last_mtime:
            return mtime


def main():
    parser = argparse.ArgumentParser(description="Animate the latest recorded RDIP episode.")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to a specific episode .npz file.",
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Run directory (if not specified, newest directory in runs/ is used).",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Episode index to visualize (requires --run).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep watching for updates and animate whenever the file changes.",
    )
    args = parser.parse_args()

    runs_dir = Path("runs")
    episode_path = args.path

    if episode_path is None:
        # Resolve which episode file to visualize based on CLI arguments.
        run_dir = args.run
        if run_dir is None:
            if not runs_dir.exists():
                raise FileNotFoundError("No runs/ directory found.")
            run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
            if not run_dirs:
                raise FileNotFoundError("No run subdirectories found in runs/.")
            run_dir = run_dirs[-1]
        if args.episode is not None:
            episode_path = run_dir / f"episode_{args.episode:05d}.npz"
            if not episode_path.exists():
                raise FileNotFoundError(f"{episode_path} does not exist.")
        elif args.loop:
            episode_path = run_dir / "latest_episode.npz"
        else:
            ep_files = sorted(run_dir.glob("episode_*.npz"))
            if not ep_files:
                raise FileNotFoundError(f"No episode_*.npz files found in {run_dir}")
            episode_path = ep_files[-1]

    if not episode_path.exists():
        raise FileNotFoundError(f"{episode_path} does not exist. Run training first.")

    while True:
        # Draw once, optionally wait on disk changes if --loop is active.
        mtime = episode_path.stat().st_mtime
        print(f"Animating {episode_path} (mtime={mtime})")
        animate_episode(episode_path)
        if not args.loop:
            break
        print("Waiting for new episode data...")
        try:
            new_mtime = wait_for_new_file(episode_path, mtime)
        except KeyboardInterrupt:
            break
        print(f"Detected update (mtime={new_mtime}). Reloading.")


if __name__ == "__main__":
    main()
