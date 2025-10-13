# train_rdip_tqc.py
from torch.utils.tensorboard import SummaryWriter

import time
import numpy as np
from rdip_env import RDIPEnv
from tqc import TQC, Replay, DEVICE
import torch

def train(total_steps=500_000, seed=0):
    env = RDIPEnv(seed=seed)
    writer = SummaryWriter()
    # DIAG: replacing this line for fixed alpha
    algo = TQC(obs_dim=10, act_limit=env.max_action, target_entropy=-1.0)  # 1D action
    # DIAG: with this
	# A/B test with fixed alpha (e.g., 0.1). Comment this in for the experiment:
    algo = TQC(obs_dim=10,
               act_limit=env.max_action,
               target_entropy=-1.0,      # kept for when you turn auto back on
               fixed_alpha=0.1)          # <— enable fixed α here

    buf = Replay(size=200_000)

    start_ep_steps = 10000  # pure exploration warmup
    batch = 512
    updates_per_step = 1
    ep_len_ctrl = int(env.T / env.control_dt)  # 10s / 0.01s = 1000 steps/episode

    s = env.reset(ep_mode=0)
    ep_step = 0
    ep_ret = 0.0
    ep = 0

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

                # >>> Per-update TensorBoard logs <<<
                writer.add_scalar("loss/actor",  float(stats.get("actor_loss", float("nan"))), global_step=t)
                writer.add_scalar("loss/critic", float(stats.get("critic_loss", float("nan"))), global_step=t)
                writer.add_scalar("sac/alpha",   float(stats.get("alpha",      float("nan"))), global_step=t)
                writer.add_scalar("policy/entropy", float(stats.get("policy_entropy", float("nan"))), global_step=t)

        # episode end
        if done or ep_step >= ep_len_ctrl:
            ep += 1
            # compute the rolling average using the EMA method
            avg_ret = 0.95 * avg_ret + 0.05 * ep_ret if ep > 1 else ep_ret
            
            print(f"ep {ep:04d} | steps {t:7d} | ret {ep_ret:8.2f} | avg {avg_ret:8.2f} | EP={env.ep} | alpha={stats['alpha'] if t>=start_ep_steps else '-'}")

            # >>> Per-update TensorBoard logs <<<
            alpha_val = float(stats['alpha']) if (t >= start_ep_steps and 'alpha' in stats) else float("nan")
            writer.add_scalar("train/ep_return", float(ep_ret), global_step=t)
            writer.add_scalar("train/ep_length", int(ep_step),  global_step=t)
            writer.add_scalar("sac/alpha_ep",    alpha_val,     global_step=t)

            s = env.reset()  # random EP mode next episode
            ep_step = 0
            ep_ret  = 0.0

    # export actor for deployment (PyTorch script)
    actor = algo.actor.cpu().eval()
    scripted = torch.jit.script(actor)
    scripted.save("rdip_tqc_actor.pt")
    print("Saved policy to rdip_tqc_actor.pt")

if __name__ == "__main__":
    train(total_steps=1_000_000, seed=42)

