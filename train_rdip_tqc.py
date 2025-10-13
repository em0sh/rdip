# train_rdip_tqc.py
import time
import numpy as np
from rdip_env import RDIPEnv
from tqc import TQC, Replay, DEVICE
import torch
from torch.utils.tensorboard import SummaryWriter

def train(total_steps=500_000, seed=0):
    print(f"[train] Device: {DEVICE}")
    env = RDIPEnv(seed=seed)
    algo = TQC(obs_dim=10, act_limit=env.max_action, target_entropy=-1.0)  # 1D action
    buf = Replay(size=200_000)
    run_name = f"TQC_{time.strftime('%Y%m%d-%H%M%S')}_seed{seed}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    print(f"[train] Logging TensorBoard data to runs/{run_name}")

    start_ep_steps = 10000  # pure exploration warmup
    batch = 512
    updates_per_step = 1
    ep_len_ctrl = int(env.T / env.control_dt)  # 10s / 0.01s = 1000 steps/episode

    s = env.reset(ep_mode=0)
    ep_step = 0
    ep_ret = 0.0
    ep = 0
    ema_ret = None
    ema_beta = 0.05  # smoothing factor for EMA of episode returns
    stats = None

    try:
        for t in range(1, total_steps+1):
            # sample new EP mode at episode boundary to learn all transitions
            if ep_step == 0:
                env.ep = np.random.randint(0,4)

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
                print(f"ep {ep:04d} | steps {t:7d} | ret {ep_ret:8.2f} | ema {ema_str} | H {entropy_str} | EP={env.ep} | alpha={alpha_str}")
                writer.add_scalar("episode/return", ep_ret, ep)
                if ema_ret is not None:
                    writer.add_scalar("episode/ema_return", ema_ret, ep)
                writer.add_scalar("episode/mode", env.ep, ep)
                s = env.reset()  # random EP mode next episode
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
    train(total_steps=1_000_000, seed=42)

