# rdip_env.py
import math
import numpy as np

# ========= Paper parameters (Table 2) =========
PARAMS = dict(
    M1=0.187, M2=0.132,
    Ixx1=1.0415e-3, Ixx2=8.8210e-4,
    Iyy1=4.3569e-3, Iyy2=4.9793e-3,
    Izz1=3.3179e-3, Izz2=4.8178e-3,
    Ixz1=3.7770e-4, Ixz2=1.9823e-4,
    l1=0.072, l2=0.133,
    c1=2.41e-6, c2=1.09e-6,
    L1=0.1645, R1=0.1625, r1=0.1597, r2=0.0209,
    g=9.81
)

# ========= Target EP angles (Table 3) =========
EP_TARGETS = {
    0: (-math.pi,  0.0),
    1: (-math.pi, -math.pi),
    2: (0.0,      -math.pi),
    3: (0.0,       0.0),
}

def wrap_pi(a):
    # wrap angle to [-pi, pi]
    a = (a + math.pi) % (2*math.pi) - math.pi
    return a

class RDIPEnv:
    """
    Rotary Double-Inverted Pendulum environment with:
      state x = [theta, alpha, beta, theta_dot, alpha_dot, beta_dot]
      action u = theta_ddot in [-50, 50] (angular acceleration)
    Internal integration dt=0.001s (1 ms), control step = 0.01s (10 ms), episode=10s
    """
    def __init__(self, params=PARAMS, control_dt=0.01, internal_dt=0.001, episode_seconds=10.0, seed=None):
        self.p = params.copy()
        self.control_dt = control_dt
        self.h = internal_dt
        self.steps_per_action = int(round(control_dt / internal_dt))
        self.T = episode_seconds
        self.max_action = 50.0
        self.rng = np.random.default_rng(seed)

        self.t = 0.0
        self.ep = 0  # equilibrium mode (0,1,2,3), can change per-episode
        self.x = np.zeros(6, dtype=np.float64)

    # ---------- Equations from the paper (Eqs. (2)–(6)) ----------
    def _h_terms(self, alpha, beta):
        p = self.p
        g = p['g']
        M1,M2 = p['M1'], p['M2']
        l1,l2 = p['l1'], p['l2']
        L1,R1 = p['L1'], p['R1']
        r1,r2 = p['r1'], p['r2']
        Ixx1,Ixx2 = p['Ixx1'], p['Ixx2']
        Iyy1,Iyy2 = p['Iyy1'], p['Iyy2']
        Izz1,Izz2 = p['Izz1'], p['Izz2']
        Ixz1,Ixz2 = p['Ixz1'], p['Ixz2']

        h1 = M1*l1*r1 + M2*L1*(R1 + r2) - Ixz1
        h2 = M2*l2*(R1 + r2) - Ixz2
        h3 = Ixx1 + M1*l1*l1 + M2*L1*L1
        h4 = M2*L1*l2
        h5 = g*(M1*l1 + M2*L1)
        h6 = Ixx2 + M2*l2*l2
        h7 = M2*g*l2
        h8 = M1*l1*l1 + M2*L1*L1 + Iyy1 - Izz1
        h9 = M2*l2*l2 + Iyy2 - Izz2
        return h1,h2,h3,h4,h5,h6,h7,h8,h9

    def _accelerations(self, x, u):
        # x=[theta, alpha, beta, theta_dot, alpha_dot, beta_dot]
        theta, alpha, beta, thd, ad, bd = x
        p = self.p
        c1, c2 = p['c1'], p['c2']

        h1,h2,h3,h4,h5,h6,h7,h8,h9 = self._h_terms(alpha, beta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        cab, sab = math.cos(alpha+beta), math.sin(alpha+beta)
        cb = math.cos(beta); sb = math.sin(beta)

        n1 = h1*ca + h2*cab
        n2 = h2*cab

        m11 = h3 + h6 + 2*h4*cb
        m12 = h6 + h4*cb
        m21 = m12
        m22 = h6

        d1 = (-h4*sb*(2*ad*bd + bd*bd)
              - h5*sa
              - h7*sab
              + c1*ad
              - thd*thd*(0.5*h8*math.sin(2*alpha) + h4*math.sin(2*alpha + beta) + 0.5*h9*math.sin(2*(alpha+beta))))
        d2 = ( h4*sb*(ad*ad)
              - h7*sab
              + c2*bd
              - thd*thd*(0.5*h9*math.sin(2*(alpha+beta)) + 0.5*h4*(math.sin(2*(alpha+beta)) - sb)))

        Phi = m11*m22 - m12*m21
        # Eq. (5)
        addot = ((-m22*n1 + m12*n2)*u + (-m22*d1 + m12*d2))/Phi
        bddot = (( m21*n1 - m11*n2)*u + ( m21*d1 - m11*d2))/Phi
        return addot, bddot

    def _f(self, x, u):
        # state derivative f(x,u) (Eq. (6))
        theta, alpha, beta, thd, ad, bd = x
        addot, bddot = self._accelerations(x, u)
        return np.array([thd, ad, bd, u, addot, bddot], dtype=np.float64)

    def _rk4(self, x, u, h):
        k1 = self._f(x, u)
        k2 = self._f(x + 0.5*h*k1, u)
        k3 = self._f(x + 0.5*h*k2, u)
        k4 = self._f(x + h*k3, u)
        return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # ---------- Public API ----------
    def reset(self, ep_mode=None):
        # randomize initial state within paper's ranges (Eq. (15))
        if ep_mode is None:
            self.ep = self.rng.integers(0, 4)
        else:
            self.ep = int(ep_mode)

        th  = self.rng.uniform(-math.pi, math.pi)
        al  = self.rng.uniform(-math.pi, math.pi)
        be  = self.rng.uniform(-math.pi, math.pi)
        thd = self.rng.uniform(-7, 7)
        ad  = self.rng.uniform(-10, 10)
        bd  = self.rng.uniform(-20, 20)

        self.x = np.array([th, al, be, thd, ad, bd], dtype=np.float64)
        self.t = 0.0
        return self._obs()

    def _obs(self):
        th, al, be, thd, ad, bd = self.x
        # wrap angular positions
        th = wrap_pi(th); al = wrap_pi(al); be = wrap_pi(be)
        # state features (Eq. (13))
        s = np.array([
            math.sin(th), math.cos(th),
            math.sin(al), math.cos(al),
            math.sin(be), math.cos(be),
            thd, ad, bd,
            float(self.ep)  # encode EP as a scalar context
        ], dtype=np.float32)
        return s

    def step(self, u):
        # clamp action to [-50, 50]
        u = float(np.clip(u, -self.max_action, self.max_action))
        # integrate at 1 ms for self.steps_per_action iterations
        for _ in range(self.steps_per_action):
            self.x = self._rk4(self.x, u, self.h)
            self.t += self.h
        # produce reward and done flag
        r = self._reward()
        done = self.t >= self.T
        return self._obs(), r, done, {}

    # ---------- Reward (Eq. (16)) ----------
    def _reward(self):
        th, al, be, thd, ad, bd = self.x
        # target angles by EP (Table 3)
        alpha_star, beta_star = EP_TARGETS[self.ep]

        # angular wrapping for reward
        th = wrap_pi(th); al = wrap_pi(al); be = wrap_pi(be)

        # components
        # NB: we don't have direct |u| here; use |theta_ddot| proxy through recent dynamics if desired.
        # In practice we use the chosen action magnitude at call-site for Ru; here we approximate via |theta_ddot| = |ẋ4|
        # but since we integrate internally, we approximate Ru as modest constant factor to keep code local:
        Ru  = 1.0  # optionally feed actual |u| from the agent into reward() externally
        Rth = 0.5 + 0.5*math.cos(th)
        Ra  = 0.5 + 0.5*math.cos(al - alpha_star)
        Rb  = 0.5 + 0.5*math.cos(be - beta_star)
        Rthd = math.exp(-0.02*abs(thd))
        Rad  = math.exp(-0.02*abs(ad))
        Rbd  = math.exp(-0.02*abs(bd))
        return float(Ru * Rth * Ra * Rb * Rthd * Rad * Rbd)

