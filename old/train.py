import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv


def rdip_ode(t, x, u, p):
    # x = [θ, α, β, θ̇, α̇, β̇]
    θ, α, β, θd, αd, βd = x
    # -- compute matrices M(q), C(q,dq), G(q) using equations from the paper --
    M = M_matrix(θ, α, β, p)
    C = C_vector(θ, α, β, θd, αd, βd, p)
    G = G_vector(θ, α, β, p)
    B = np.array([[1.0], [0.0], [0.0]])  # actuation at θ only
    qdd = np.linalg.solve(M, B*u - (C + G))
    return np.array([θd, αd, βd, *qdd])

def rk4_step(f, t, x, u, dt, p):
    k1 = f(t, x, u, p)
    k2 = f(t+dt/2, x + dt*k1/2, u, p)
    k3 = f(t+dt/2, x + dt*k2/2, u, p)
    k4 = f(t+dt,   x + dt*k3,   u, p)
    return x + dt*(k1 + 2*k2 + 2*k3 + k4)/6

def reward(x, u, α_star, β_star):
    θ, α, β, θd, αd, βd = x
    Ru   = np.exp(-0.005*np.abs(u))
    Rθ   = 0.5 + 0.5*np.cos(θ)
    Rα   = 0.5 + 0.5*np.cos(α - α_star)
    Rβ   = 0.5 + 0.5*np.cos(β - β_star)
    Rθd  = np.exp(-0.02*np.abs(θd))
    Rαd  = np.exp(-0.02*np.abs(αd))
    Rβd  = np.exp(-0.02*np.abs(βd))
    return Ru * Rθ * Rα * Rβ * Rθd * Rαd * Rβd

def simulate_episode(policy, p, α_star, β_star, Tmax=10.0, dt=0.001):
    t, x, total_reward = 0.0, random_init_state(), 0.0
    while t < Tmax:
        u = policy(x)                   # continuous scalar
        for _ in range(10):             # 10 physics steps = 10 ms
            x = rk4_step(rdip_ode, t, x, u, dt, p)
            t += dt
        r = reward(x, u, α_star, β_star)
        total_reward += r
        # store (s,a,r,s') in replay buffer
    return total_reward


env = DummyVecEnv([lambda: YourCustomRDIPEnv(params)])  # wrap your loop above
model = TQC("MlpPolicy", env, learning_rate=3e-4,
            batch_size=256, gamma=0.995, tau=0.005,
            policy_kwargs=dict(net_arch=[256,256]), verbose=1)
model.learn(total_timesteps=5_000_000)

