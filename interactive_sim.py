"""Interactive RDIP policy viewer.

Usage:
    python interactive_sim.py --actor rdip_tqc_actor.pt

Left subplot: rotary double inverted pendulum (projected 3D view)
Right subplot: angle traces over time (last N seconds)

Controls (right side):
    * Reset: reset environment using values from the TextBoxes below
    * Pause/Resume: toggle simulation
    * Step: advance one control step while paused
    * Disturbance slider + Apply buttons: inject additive acceleration once
    * EP selector: choose target equilibrium mode fed into the policy
    * State TextBoxes: initial angles/velocities for next reset
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
import numpy as np
import torch

from rdip_env import RDIPEnv, wrap_pi


def project_pendulum(theta: float, alpha: float, beta: float, params: Dict[str, float]):
    """Project 3D configuration into 2D coordinates for plotting."""
    L1 = params["L1"]
    l1 = params["l1"]
    l2 = params["l2"]

    # 3D positions
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


class InteractiveSimulator:
    def __init__(self, actor_path: Path, history_window: float = 10.0, disturbance_limit: float = 10.0):
        self.actor = torch.jit.load(str(actor_path)).eval()
        self.device = torch.device("cpu")
        self.actor.to(self.device)

        self.env = RDIPEnv(seed=0)
        self.history_window = history_window
        self.disturbance_limit = disturbance_limit

        self.target_ep = 0
        self.pending_disturbance = 0.0
        self.paused = False
        self.elapsed_time = 0.0
        self.history_steps = int(self.history_window / self.env.control_dt)

        self._init_state_arrays()
        self._build_figure()
        self.reset_env()

    def _init_state_arrays(self):
        self.time_hist: List[float] = []
        self.theta_hist: List[float] = []
        self.alpha_hist: List[float] = []
        self.beta_hist: List[float] = []

    def _build_figure(self):
        self.fig, (self.ax_pend, self.ax_angles) = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(left=0.08, right=0.78, bottom=0.25)

        self.ax_pend.set_title("RDIP (projected)")
        self.ax_pend.set_xlabel("Projected horizontal (m)")
        self.ax_pend.set_ylabel("Vertical (m)")
        self.pend_line, = self.ax_pend.plot([], [], "-o", lw=2)
        self.time_text = self.ax_pend.text(0.02, 0.95, "", transform=self.ax_pend.transAxes)

        self.ax_angles.set_title("Angles")
        self.ax_angles.set_xlabel("Time (s)")
        self.ax_angles.set_ylabel("Angle (rad)")
        self.theta_line, = self.ax_angles.plot([], [], label=r"$\theta$")
        self.alpha_line, = self.ax_angles.plot([], [], label=r"$\alpha$")
        self.beta_line, = self.ax_angles.plot([], [], label=r"$\beta$")
        self.ax_angles.legend(loc="upper right")

        # Controls
        btn_reset_ax = plt.axes([0.82, 0.88, 0.15, 0.05])
        btn_pause_ax = plt.axes([0.82, 0.82, 0.15, 0.05])
        btn_step_ax = plt.axes([0.82, 0.76, 0.15, 0.05])
        radio_ax = plt.axes([0.82, 0.55, 0.15, 0.18])

        slider_ax = plt.axes([0.82, 0.48, 0.15, 0.03])
        btn_apply_pos_ax = plt.axes([0.82, 0.43, 0.07, 0.04])
        btn_apply_neg_ax = plt.axes([0.90, 0.43, 0.07, 0.04])
        disturbance_info_ax = plt.axes([0.82, 0.38, 0.15, 0.03])

        textbox_axes = {
            "theta": plt.axes([0.82, 0.34, 0.15, 0.035]),
            "alpha": plt.axes([0.82, 0.30, 0.15, 0.035]),
            "beta": plt.axes([0.82, 0.26, 0.15, 0.035]),
            "theta_dot": plt.axes([0.82, 0.22, 0.15, 0.035]),
            "alpha_dot": plt.axes([0.82, 0.18, 0.15, 0.035]),
            "beta_dot": plt.axes([0.82, 0.14, 0.15, 0.035]),
        }

        self.btn_reset = Button(btn_reset_ax, "Reset")
        self.btn_pause = Button(btn_pause_ax, "Pause")
        self.btn_step = Button(btn_step_ax, "Step")
        self.radio_ep = RadioButtons(radio_ax, ("EP0", "EP1", "EP2", "EP3"), active=0)
        self.slider_dist = Slider(
            slider_ax,
            "Impulse (rad/s²)",
            -self.disturbance_limit,
            self.disturbance_limit,
            valinit=0.0,
        )
        self.btn_apply_pos = Button(btn_apply_pos_ax, "Inject +")
        self.btn_apply_neg = Button(btn_apply_neg_ax, "Inject -")
        disturbance_note = (
            "Applies a single-step\nangular acceleration\nimpulse (additive)"
        )
        disturbance_info_ax.axis("off")
        disturbance_info_ax.text(
            0.0,
            0.5,
            disturbance_note,
            fontsize=8,
            verticalalignment="center",
            transform=disturbance_info_ax.transAxes,
        )

        self.textboxes = {
            name: TextBox(axis, name.replace("_", " "), initial="0.0")
            for name, axis in textbox_axes.items()
        }

        self.btn_reset.on_clicked(self.handle_reset)
        self.btn_pause.on_clicked(self.handle_pause)
        self.btn_step.on_clicked(self.handle_step)
        self.radio_ep.on_clicked(self.handle_ep_change)
        self.btn_apply_pos.on_clicked(self.handle_apply_positive)
        self.btn_apply_neg.on_clicked(self.handle_apply_negative)

        self.anim = FuncAnimation(
            self.fig,
            self.update,
            interval=int(self.env.control_dt * 1000),
            blit=False,
            cache_frame_data=False,
        )

    def parse_initial_state(self) -> np.ndarray:
        state = np.zeros(6, dtype=np.float64)
        keys = ["theta", "alpha", "beta", "theta_dot", "alpha_dot", "beta_dot"]
        for i, key in enumerate(keys):
            try:
                state[i] = float(self.textboxes[key].text)
            except ValueError:
                state[i] = 0.0
        return state

    def reset_env(self, custom_state: np.ndarray = None):
        self.env.reset(ep_mode=self.target_ep)
        if custom_state is not None:
            self.env.x = custom_state.astype(np.float64)
        self.env.t = 0.0
        self.elapsed_time = 0.0
        self._init_state_arrays()
        self._append_history_sample()
        self.pending_disturbance = 0.0

    def handle_reset(self, _event):
        self.reset_env(self.parse_initial_state())

    def handle_pause(self, _event):
        self.paused = not self.paused
        self.btn_pause.label.set_text("Resume" if self.paused else "Pause")

    def handle_step(self, _event):
        if self.paused:
            self._simulation_step()
            self._update_plot()

    def handle_ep_change(self, label: str):
        self.target_ep = int(label[-1])

    def handle_apply_positive(self, _event):
        self.pending_disturbance += self.slider_dist.val

    def handle_apply_negative(self, _event):
        self.pending_disturbance -= self.slider_dist.val

    def _append_history_sample(self):
        self.time_hist.append(self.elapsed_time)
        self.theta_hist.append(wrap_pi(self.env.x[0]))
        self.alpha_hist.append(wrap_pi(self.env.x[1]))
        self.beta_hist.append(wrap_pi(self.env.x[2]))

        if len(self.time_hist) > self.history_steps:
            self.time_hist = self.time_hist[-self.history_steps:]
            self.theta_hist = self.theta_hist[-self.history_steps:]
            self.alpha_hist = self.alpha_hist[-self.history_steps:]
            self.beta_hist = self.beta_hist[-self.history_steps:]

    def _simulation_step(self):
        self.env.ep = self.target_ep
        obs = self.env._obs().copy()
        obs[-1] = float(self.target_ep)

        obs_tensor = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            actor_out = self.actor(obs_tensor, deterministic=False, with_logprob=False)
        if isinstance(actor_out, (tuple, list)):
            action_tensor = actor_out[0]
        else:
            action_tensor = actor_out
        action = float(action_tensor.squeeze().cpu().item())
        action = np.clip(action + self.pending_disturbance, -self.env.max_action, self.env.max_action)
        self.pending_disturbance = 0.0

        self.env.step(action)
        self.elapsed_time += self.env.control_dt
        self._append_history_sample()

    def _update_plot(self):
        theta = self.theta_hist[-1]
        alpha = self.alpha_hist[-1]
        beta = self.beta_hist[-1]

        x_coords, z_coords = project_pendulum(theta, alpha, beta, self.env.p)
        self.pend_line.set_data(x_coords, z_coords)
        self.ax_pend.set_xlim(x_coords.min() - 0.2, x_coords.max() + 0.2)
        self.ax_pend.set_ylim(z_coords.min() - 0.2, z_coords.max() + 0.4)
        self.time_text.set_text(f"t = {self.elapsed_time:5.2f}s | EP{self.target_ep}")

        self.theta_line.set_data(self.time_hist, self.theta_hist)
        self.alpha_line.set_data(self.time_hist, self.alpha_hist)
        self.beta_line.set_data(self.time_hist, self.beta_hist)
        if self.time_hist:
            t_min = max(0.0, self.time_hist[-1] - self.history_window)
            self.ax_angles.set_xlim(t_min, t_min + self.history_window)
        self.ax_angles.set_ylim(-math.pi - 0.5, math.pi + 0.5)

    def update(self, _frame):
        if not self.paused:
            self._simulation_step()
        self._update_plot()
        return self.pend_line, self.theta_line, self.alpha_line, self.beta_line


def main():
    parser = argparse.ArgumentParser(description="Interactive RDIP simulator")
    parser.add_argument("--actor", type=Path, default=Path("rdip_tqc_actor.pt"), help="Path to TorchScript actor (.pt)")
    parser.add_argument("--history", type=float, default=10.0, help="Angle trace window (seconds)")
    parser.add_argument("--disturbance", type=float, default=10.0, help="Maximum disturbance magnitude")
    args = parser.parse_args()

    if not args.actor.exists():
        raise FileNotFoundError(f"Actor file not found: {args.actor}")

    sim = InteractiveSimulator(args.actor, history_window=args.history, disturbance_limit=args.disturbance)
    plt.show()


if __name__ == "__main__":
    main()
