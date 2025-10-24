#!/usr/bin/env python3
"""Runtime actor utilities for RDIP control on embedded targets.

This module wraps a TorchScript policy so parent processes can query it with
encoder readings at low latency.  It mirrors the simulator’s observation
packing, but avoids any simulator dependencies to keep the hot path lean.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from rdip_env import EP_TARGETS, wrap_pi

OBS_DIM = 17  # matches RDIPEnv observation size


def build_observation(state: Sequence[float], ep_mode: int) -> np.ndarray:
    """Convert raw pendulum state into the 17-D observation vector."""
    theta, alpha, beta, theta_dot, alpha_dot, beta_dot = state
    alpha_star, beta_star = EP_TARGETS[int(ep_mode)]

    theta = wrap_pi(theta)
    alpha = wrap_pi(alpha)
    beta = wrap_pi(beta)
    alpha_err = wrap_pi(alpha - alpha_star)
    beta_err = wrap_pi(beta - beta_star)

    obs = np.array(
        [
            np.sin(theta),
            np.cos(theta),
            np.sin(alpha),
            np.cos(alpha),
            np.sin(beta),
            np.cos(beta),
            theta_dot,
            alpha_dot,
            beta_dot,
            np.sin(alpha_star),
            np.cos(alpha_star),
            np.sin(beta_star),
            np.cos(beta_star),
            np.sin(alpha_err),
            np.cos(alpha_err),
            np.sin(beta_err),
            np.cos(beta_err),
        ],
        dtype=np.float32,
    )
    return obs


class RuntimeActor:
    """Optimized TorchScript actor wrapper for low-latency inference."""

    def __init__(
        self,
        actor_path: Path,
        *,
        deterministic: bool = True,
        num_warmup: int = 8,
        device: Optional[torch.device] = None,
    ) -> None:
        self.actor_path = Path(actor_path)
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.deterministic = deterministic

        torch.set_num_threads(1)
        self._actor = torch.jit.load(str(self.actor_path), map_location=device).eval()
        try:
            self._actor = torch.jit.optimize_for_inference(self._actor)
        except (AttributeError, RuntimeError):
            pass

        self._obs_tensor = torch.empty((1, OBS_DIM), dtype=torch.float32, device=device)
        self._out_tensor = torch.empty((1, 1), dtype=torch.float32, device=device)

        if num_warmup > 0:
            dummy = torch.zeros((1, OBS_DIM), dtype=torch.float32, device=device)
            with torch.inference_mode():
                for _ in range(num_warmup):
                    self._actor(dummy)

    @torch.inference_mode()
    def infer(self, observation: np.ndarray) -> Tuple[float, float]:
        """Run the actor on a single observation.

        Returns (action, latency_seconds).
        """
        if observation.dtype != np.float32:
            observation = observation.astype(np.float32, copy=False)
        obs_tensor = torch.from_numpy(observation)
        self._obs_tensor[0].copy_(obs_tensor)
        start = time.perf_counter_ns()
        output = self._actor(self._obs_tensor)
        end = time.perf_counter_ns()

        if isinstance(output, (tuple, list)):
            out_tensor = output[0]
        else:
            out_tensor = output
        if out_tensor.dim() == 2 and out_tensor.shape == (1, 1):
            action = float(out_tensor.item())
        else:
            self._out_tensor.copy_(out_tensor.reshape(1, 1))
            action = float(self._out_tensor.item())
        latency = (end - start) * 1e-9
        return action, latency

    def infer_from_state(self, state: Sequence[float], ep_mode: int) -> Tuple[float, float]:
        """Helper that builds the observation from a 6-D state before inferring."""
        obs = build_observation(state, ep_mode)
        return self.infer(obs)


class EncoderSample:
    """Lightweight container for encoder telemetry."""

    __slots__ = ("theta", "alpha", "beta", "theta_dot", "alpha_dot", "beta_dot")

    def __init__(
        self,
        theta: float,
        alpha: float,
        beta: float,
        theta_dot: float,
        alpha_dot: float,
        beta_dot: float,
    ) -> None:
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.theta_dot = theta_dot
        self.alpha_dot = alpha_dot
        self.beta_dot = beta_dot

    def as_state(self) -> Tuple[float, float, float, float, float, float]:
        return (
            self.theta,
            self.alpha,
            self.beta,
            self.theta_dot,
            self.alpha_dot,
            self.beta_dot,
        )


class EncoderInterface:
    """Abstract interface for reading encoder data."""

    def read(self) -> EncoderSample:
        raise NotImplementedError


class StubEncoder(EncoderInterface):
    """Fallback encoder returning zeros (placeholder for hardware integration)."""

    def read(self) -> EncoderSample:
        return EncoderSample(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
