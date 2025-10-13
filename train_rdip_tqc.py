# train_rdip_tqc.py
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rdip_env import RDIPEnv
from tqc import DEVICE, Replay, TQC


def train(total_steps=5_000_000, seed=0):
    print(f"[train] Device: {DEVICE}")

    # Instantiate a probe environment to read constants and act_limit
    probe_env = RDIPEnv(seed=seed)
    algo = TQC(obs_dim=10, act_limit=probe_env.max_action, target_entropy=-0.4, n_critics=5)
    buf = Replay(size=1_000_000)

    run_name = f"TQC_{time.strftime('%Y%m%d-%H%M%S')}_seed{seed}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"[train] Logging TensorBoard data to {run_dir}")

    start_ep_steps = 10_000
    warmup_remaining = start_ep_steps
    batch = 512
    updates_per_step = 1

    # Determine how many environments to run in parallel
    cpu_count = os.cpu_count() or 1
    max_envs = min(max(cpu_count // 2, 1), 8)
    num_envs = max_envs if DEVICE.type == "cuda" and max_envs > 1 else 1
    print(f"[train] Parallel environments: {num_envs}")

    envs = [probe_env] + [RDIPEnv(seed=seed + (i + 1) * 1000) for i in range(num_envs - 1)]
    obs_list = []
    episode_times = [[] for _ in range(num_envs)]
    episode_states = [[] for _ in range(num_envs)]
    episode_actions = [[] for _ in range(num_envs)]
    episode_rewards = [[] for _ in range(num_envs)]
    episode_modes = []
    ep_returns = np.zeros(num_envs, dtype=np.float64)
    ep_start_times = [time.time() for _ in range(num_envs)]
    ep = 0
    ema_ret = None
    ema_beta = 0.05
    total_ep_time = 0.0
    mode_counts = defaultdict(int)
    mode_return_totals = defaultdict(float)
    stats = None

    for idx, env in enumerate(envs):
        obs_list.append(env.reset(ep_mode=np.random.randint(0, 4)))
        episode_modes.append(env.ep)
        ep_start_times[idx] = time.time()

    steps_goal = total_steps
    total_steps = 0
    latest_episode_path = run_dir / "latest_episode.npz"

    executor = ThreadPoolExecutor(max_workers=num_envs) if num_envs > 1 else None

    try:
        while total_steps < steps_goal:
            steps_remaining = steps_goal - total_steps
            active_envs = min(num_envs, steps_remaining)

            actions = []
            for idx in range(active_envs):
                env = envs[idx]
                episode_times[idx].append(env.t)
                episode_states[idx].append(env.x.copy())
                if warmup_remaining > 0:
                    a = np.array([np.random.uniform(-env.max_action, env.max_action)], dtype=np.float32)
                    warmup_remaining -= 1
                else:
                    a = algo.act(obs_list[idx], deterministic=False).astype(np.float32)
                episode_actions[idx].append(float(a[0]))
                actions.append(a)

            action_scalars = [float(a[0]) for a in actions]

            if executor:
                futures = [
                    executor.submit(envs[idx].step, action_scalars[idx])
                    for idx in range(active_envs)
                ]
            else:
                futures = [None] * active_envs

            for idx in range(active_envs):
                env = envs[idx]
                if executor:
                    next_obs, base_r, done, _ = futures[idx].result()
                else:
                    next_obs, base_r, done, _ = env.step(action_scalars[idx])

                action_scalar = action_scalars[idx]
                reward = float(math.exp(-0.005 * abs(action_scalar)) * base_r)
                episode_rewards[idx].append(reward)
                buf.add(obs_list[idx], actions[idx], reward, next_obs, float(done))
                obs_list[idx] = next_obs
                ep_returns[idx] += reward
                total_steps += 1

                if done:
                    episode_times[idx].append(env.t)
                    episode_states[idx].append(env.x.copy())
                    ep += 1
                    ep_time = time.time() - ep_start_times[idx]
                    total_ep_time += ep_time
                    avg_ep_time = total_ep_time / ep
                    ep_return = ep_returns[idx]
                    ema_ret = ep_return if ema_ret is None else (1.0 - ema_beta) * ema_ret + ema_beta * ep_return
                    ema_str = f"{ema_ret:8.2f}" if ema_ret is not None else "       -"
                    if stats is not None:
                        alpha_str = f"{stats['alpha']:.6f}"
                        entropy_str = f"{stats['entropy']:.3f}"
                    else:
                        alpha_str = "-"
                        entropy_str = "-"

                    mode = episode_modes[idx]
                    mode_counts[mode] += 1
                    mode_return_totals[mode] += ep_return
                    total_eps = sum(mode_counts.values())

                    print(
                        f"ep {ep:04d} | steps {total_steps:7d} | ret {ep_return:8.2f} | ema {ema_str} | "
                        f"H {entropy_str} | EP={mode} | alpha={alpha_str} | "
                        f"dt/ep {ep_time:6.2f}s | avg_dt/ep {avg_ep_time:6.2f}s"
                    )

                    writer.add_scalar("episode/return", ep_return, ep)
                    if ema_ret is not None:
                        writer.add_scalar("episode/ema_return", ema_ret, ep)
                    writer.add_scalar("episode/mode", mode, ep)
                    writer.add_scalar(f"episode/return_mode_{mode}", ep_return, ep)
                    writer.add_scalar("episode/duration", ep_time, ep)
                    writer.add_scalar("episode/duration_avg", avg_ep_time, ep)
                    for m in range(4):
                        if total_eps:
                            writer.add_scalar(f"episode/mode_pct_{m}", mode_counts[m] / total_eps, ep)
                        if mode_counts[m]:
                            writer.add_scalar(
                                f"episode/return_mode_avg_{m}",
                                mode_return_totals[m] / mode_counts[m],
                                ep,
                            )

                    data = {
                        "time": np.array(episode_times[idx], dtype=np.float32),
                        "state": np.array(episode_states[idx], dtype=np.float32),
                        "action": np.array(episode_actions[idx], dtype=np.float32),
                        "reward": np.array(episode_rewards[idx], dtype=np.float32),
                        "mode": np.array(mode, dtype=np.int32),
                        "episode": np.array(ep, dtype=np.int32),
                        "control_dt": np.array(env.control_dt, dtype=np.float32),
                        "internal_dt": np.array(env.h, dtype=np.float32),
                        "params": np.array([env.p], dtype=object),
                    }
                    np.savez(run_dir / f"episode_{ep:05d}.npz", **data)
                    np.savez(latest_episode_path, **data)

                    obs_list[idx] = env.reset(ep_mode=np.random.randint(0, 4))
                    episode_modes[idx] = env.ep
                    episode_times[idx] = []
                    episode_states[idx] = []
                    episode_actions[idx] = []
                    episode_rewards[idx] = []
                    ep_returns[idx] = 0.0
                    ep_start_times[idx] = time.time()
                if total_steps >= steps_goal:
                    break
            if total_steps >= steps_goal:
                break

            if warmup_remaining <= 0 and len(buf) >= batch:
                updates_to_run = updates_per_step
                for _ in range(updates_to_run):
                    stats = algo.train_step(buf, batch=batch)
                    if stats is not None:
                        writer.add_scalar("train/q_loss", stats["q_loss"], total_steps)
                        writer.add_scalar("train/pi_loss", stats["pi_loss"], total_steps)
                        writer.add_scalar("train/alpha", stats["alpha"], total_steps)
                        writer.add_scalar("train/entropy", stats["entropy"], total_steps)
                        writer.add_scalar("train/log_prob", stats["logp"], total_steps)
        actor = algo.actor.cpu().eval()
        scripted = torch.jit.script(actor)
        scripted.save("rdip_tqc_actor.pt")
        print("Saved policy to rdip_tqc_actor.pt")
    finally:
        if executor:
            executor.shutdown(wait=True)
        writer.close()


if __name__ == "__main__":
    train(total_steps=5_000_000, seed=42)
