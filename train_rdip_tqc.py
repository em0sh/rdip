# train_rdip_tqc.py
import time
import numpy as np
from rdip_env import RDIPEnv
from tqc import TQC, Replay, DEVICE
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def train(total_steps=500_000, seed=0):
    print(f"[train] Device: {DEVICE}")
    env = RDIPEnv(seed=seed)
    algo = TQC(obs_dim=10, act_limit=env.max_action, target_entropy=-1.0)  # 1D action
    buf = Replay(size=200_000)
    run_name = f"TQC_{time.strftime('%Y%m%d-%H%M%S')}_seed{seed}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"[train] Logging TensorBoard data to {run_dir}")

    start_ep_steps = 10000  # pure exploration warmup
    batch = 512
    updates_per_step = 1
    ep_len_ctrl = int(env.T / env.control_dt)  # 10s / 0.01s = 1000 steps/episode

    s = env.reset(ep_mode=np.random.randint(0, 4))
    ep_step = 0
    ep_ret = 0.0
    ep = 0
    ema_ret = None
    ema_beta = 0.05  # smoothing factor for EMA of episode returns
    stats = None
    episode_times = []
    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_mode = env.ep
    episode_snapshots = {}

    try:
        for t in range(1, total_steps+1):
            if ep_step == 0:
                episode_times = []
                episode_states = []
                episode_actions = []
                episode_rewards = []
                episode_mode = env.ep

            # action
            if t < start_ep_steps:
                a = np.array([np.random.uniform(-env.max_action, env.max_action)], dtype=np.float32)
            else:
                a = algo.act(s, deterministic=False).astype(np.float32)

            # step with control-cost inside reward (matches Eq. (16))
            # We compute reward here by replacing Ru in env with actual |u|:
            s2, base_r, done, _ = env.step(a[0])
            u_cost = np.exp(-0.005*abs(a[0]))
            r = float(u_cost) * base_r

            episode_times.append(env.t)
            episode_states.append(env.x.copy())
            episode_actions.append(float(a[0]))
            episode_rewards.append(r)

            d = done
            buf.add(s, a, r, s2, float(d))
            s = s2
            ep_ret += r
            ep_step += 1

            # update
            if t >= start_ep_steps:
                for _ in range(updates_per_step):
                    stats = algo.train_step(buf, batch=batch)
                if stats is not None:
                    writer.add_scalar("train/q_loss", stats["q_loss"], t)
                    writer.add_scalar("train/pi_loss", stats["pi_loss"], t)
                    writer.add_scalar("train/alpha", stats["alpha"], t)
                    writer.add_scalar("train/entropy", stats["entropy"], t)
                    writer.add_scalar("train/log_prob", stats["logp"], t)

            # episode end
            if done or ep_step >= ep_len_ctrl:
                ep += 1
                ema_ret = ep_ret if ema_ret is None else (1.0 - ema_beta)*ema_ret + ema_beta*ep_ret
                ema_str = f"{ema_ret:8.2f}" if ema_ret is not None else "       -"
                alpha_str = f"{stats['alpha']:.6f}" if (stats is not None and t >= start_ep_steps) else "-"
                entropy_str = f"{stats['entropy']:.3f}" if (stats is not None and t >= start_ep_steps) else "-"
                print(f"ep {ep:04d} | steps {t:7d} | ret {ep_ret:8.2f} | ema {ema_str} | H {entropy_str} | EP={episode_mode} | alpha={alpha_str}")
                writer.add_scalar("episode/return", ep_ret, ep)
                if ema_ret is not None:
                    writer.add_scalar("episode/ema_return", ema_ret, ep)
                writer.add_scalar("episode/mode", episode_mode, ep)

                episode_array = np.array(episode_states, dtype=np.float32)
                data = {
                    "time": np.array(episode_times, dtype=np.float32),
                    "state": episode_array,
                    "action": np.array(episode_actions, dtype=np.float32),
                    "reward": np.array(episode_rewards, dtype=np.float32),
                    "mode": np.array(episode_mode, dtype=np.int32),
                    "episode": np.array(ep, dtype=np.int32),
                    "control_dt": np.array(env.control_dt, dtype=np.float32),
                    "internal_dt": np.array(env.h, dtype=np.float32),
                    "params": np.array([env.p], dtype=object),
                }
                episode_snapshots[ep] = data
                np.savez(run_dir / f"episode_{ep:05d}.npz", **data)
                np.savez(run_dir / "latest_episode.npz", **data)

                next_mode = np.random.randint(0, 4)
                s = env.reset(ep_mode=next_mode)
                ep_step = 0
                ep_ret  = 0.0

        # export actor for deployment (PyTorch script)
        actor = algo.actor.cpu().eval()
        scripted = torch.jit.script(actor)
        scripted.save("rdip_tqc_actor.pt")
        print("Saved policy to rdip_tqc_actor.pt")
    finally:
        writer.close()

if __name__ == "__main__":
    train(total_steps=5_000_000, seed=42)
