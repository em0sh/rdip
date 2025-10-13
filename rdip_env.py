# rdip_env.py
import math
import numpy as np

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMBA_AVAILABLE = False

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

PARAM_VEC_TEMPLATE = np.array([
    PARAMS['M1'], PARAMS['M2'],
    PARAMS['Ixx1'], PARAMS['Ixx2'],
    PARAMS['Iyy1'], PARAMS['Iyy2'],
    PARAMS['Izz1'], PARAMS['Izz2'],
    PARAMS['Ixz1'], PARAMS['Ixz2'],
    PARAMS['l1'], PARAMS['l2'],
    PARAMS['c1'], PARAMS['c2'],
    PARAMS['L1'], PARAMS['R1'],
    PARAMS['r1'], PARAMS['r2'],
    PARAMS['g']
], dtype=np.float64)

if _NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True, nogil=True)
    def _accelerations_numba(x, u, p):
        theta, alpha, beta, thd, ad, bd = x
        M1, M2 = p[0], p[1]
        Ixx1, Ixx2 = p[2], p[3]
        Iyy1, Iyy2 = p[4], p[5]
        Izz1, Izz2 = p[6], p[7]
        Ixz1, Ixz2 = p[8], p[9]
        l1, l2 = p[10], p[11]
        c1, c2 = p[12], p[13]
        L1, R1 = p[14], p[15]
        r1, r2 = p[16], p[17]
        g = p[18]

        ca = math.cos(alpha)
        sa = math.sin(alpha)
        cab = math.cos(alpha + beta)
        sab = math.sin(alpha + beta)
        cb = math.cos(beta)
        sb = math.sin(beta)

        h1 = M1 * l1 * r1 + M2 * L1 * (R1 + r2) - Ixz1
        h2 = M2 * l2 * (R1 + r2) - Ixz2
        h3 = Ixx1 + M1 * l1 * l1 + M2 * L1 * L1
        h4 = M2 * L1 * l2
        h5 = g * (M1 * l1 + M2 * L1)
        h6 = Ixx2 + M2 * l2 * l2
        h7 = M2 * g * l2
        h8 = M1 * l1 * l1 + M2 * L1 * L1 + Iyy1 - Izz1
        h9 = M2 * l2 * l2 + Iyy2 - Izz2

        n1 = h1 * ca + h2 * cab
        n2 = h2 * cab

        m11 = h3 + h6 + 2.0 * h4 * cb
        m12 = h6 + h4 * cb
        m22 = h6

        thd2 = thd * thd
        bd2 = bd * bd
        ad2 = ad * ad

        d1 = (-h4 * sb * (2.0 * ad * bd + bd2)
              - h5 * sa
              - h7 * sab
              + c1 * ad
              - thd2 * (0.5 * h8 * math.sin(2.0 * alpha)
                        + h4 * math.sin(2.0 * alpha + beta)
                        + 0.5 * h9 * math.sin(2.0 * (alpha + beta))))

        d2 = (h4 * sb * ad2
              - h7 * sab
              + c2 * bd
              - thd2 * (0.5 * h9 * math.sin(2.0 * (alpha + beta))
                        + 0.5 * h4 * (math.sin(2.0 * (alpha + beta)) - sb)))

        phi = m11 * m22 - m12 * m12
        addot = ((-m22 * n1 + m12 * n2) * u + (-m22 * d1 + m12 * d2)) / phi
        bddot = (((m12) * n1 - m11 * n2) * u + (m12 * d1 - m11 * d2)) / phi
        return addot, bddot

    @njit(cache=True, fastmath=True, nogil=True)
    def _f_numba(x, u, p):
        addot, bddot = _accelerations_numba(x, u, p)
        res = np.empty(6, dtype=np.float64)
        res[0] = x[3]
        res[1] = x[4]
        res[2] = x[5]
        res[3] = u
        res[4] = addot
        res[5] = bddot
        return res

    @njit(cache=True, fastmath=True, nogil=True)
    def _rk4_numba(x, u, h, p):
        k1 = _f_numba(x, u, p)
        k2 = _f_numba(x + 0.5 * h * k1, u, p)
        k3 = _f_numba(x + 0.5 * h * k2, u, p)
        k4 = _f_numba(x + h * k3, u, p)
        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
else:
    _rk4_numba = None  # type: ignore


def _accelerations_py(x, u, params):
    theta, alpha, beta, thd, ad, bd = x
    p = params
    c1, c2 = p['c1'], p['c2']

    M1, M2 = p['M1'], p['M2']
    l1, l2 = p['l1'], p['l2']
    L1, R1 = p['L1'], p['R1']
    r1, r2 = p['r1'], p['r2']
    Ixx1, Ixx2 = p['Ixx1'], p['Ixx2']
    Iyy1, Iyy2 = p['Iyy1'], p['Iyy2']
    Izz1, Izz2 = p['Izz1'], p['Izz2']
    Ixz1, Ixz2 = p['Ixz1'], p['Ixz2']
    g = p['g']

    h1 = M1*l1*r1 + M2*L1*(R1 + r2) - Ixz1
    h2 = M2*l2*(R1 + r2) - Ixz2
    h3 = Ixx1 + M1*l1*l1 + M2*L1*L1
    h4 = M2*L1*l2
    h5 = g*(M1*l1 + M2*L1)
    h6 = Ixx2 + M2*l2*l2
    h7 = M2*g*l2
    h8 = M1*l1*l1 + M2*L1*L1 + Iyy1 - Izz1
    h9 = M2*l2*l2 + Iyy2 - Izz2

    ca, sa = math.cos(alpha), math.sin(alpha)
    cab, sab = math.cos(alpha+beta), math.sin(alpha+beta)
    cb = math.cos(beta)
    sb = math.sin(beta)

    n1 = h1*ca + h2*cab
    n2 = h2*cab

    m11 = h3 + h6 + 2*h4*cb
    m12 = h6 + h4*cb
    m22 = h6

    thd2 = thd*thd
    d1 = (-h4*sb*(2*ad*bd + bd*bd)
          - h5*sa
          - h7*sab
          + c1*ad
          - thd2*(0.5*h8*math.sin(2*alpha) + h4*math.sin(2*alpha + beta) + 0.5*h9*math.sin(2*(alpha+beta))))
    d2 = ( h4*sb*(ad*ad)
          - h7*sab
          + c2*bd
          - thd2*(0.5*h9*math.sin(2*(alpha+beta)) + 0.5*h4*(math.sin(2*(alpha+beta)) - sb)))

    phi = m11*m22 - m12*m12
    addot = ((-m22*n1 + m12*n2)*u + (-m22*d1 + m12*d2))/phi
    bddot = (((m12)*n1 - m11*n2)*u + (m12*d1 - m11*d2))/phi
    return addot, bddot


def _f_py(x, u, params):
    addot, bddot = _accelerations_py(x, u, params)
    return np.array([x[3], x[4], x[5], u, addot, bddot], dtype=np.float64)


def _rk4_py(x, u, h, params):
    k1 = _f_py(x, u, params)
    k2 = _f_py(x + 0.5*h*k1, u, params)
    k3 = _f_py(x + 0.5*h*k2, u, params)
    k4 = _f_py(x + h*k3, u, params)
    return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

class RDIPEnv:
    """
    Rotary Double-Inverted Pendulum environment with:
      state x = [theta, alpha, beta, theta_dot, alpha_dot, beta_dot]
      action u = theta_ddot in [-50, 50] (angular acceleration)
    Internal integration dt=0.001s (1 ms), control step = 0.01s (10 ms), episode=10s
    """
    def __init__(self, params=PARAMS, control_dt=0.01, internal_dt=0.001, episode_seconds=10.0, seed=None):
        self.p = params.copy()
        self.param_vec = np.array([
            self.p['M1'], self.p['M2'],
            self.p['Ixx1'], self.p['Ixx2'],
            self.p['Iyy1'], self.p['Iyy2'],
            self.p['Izz1'], self.p['Izz2'],
            self.p['Ixz1'], self.p['Ixz2'],
            self.p['l1'], self.p['l2'],
            self.p['c1'], self.p['c2'],
            self.p['L1'], self.p['R1'],
            self.p['r1'], self.p['r2'],
            self.p['g']
        ], dtype=np.float64)
        self.control_dt = control_dt
        self.h = internal_dt
        self.steps_per_action = int(round(control_dt / internal_dt))
        self.T = episode_seconds
        self.max_action = 50.0
        self.rng = np.random.default_rng(seed)

        self.t = 0.0
        self.ep = 0  # equilibrium mode (0,1,2,3), can change per-episode
        self.x = np.zeros(6, dtype=np.float64)
        self._use_numba = _NUMBA_AVAILABLE
        self._rk4_impl = _rk4_numba if self._use_numba else _rk4_py

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
            if self._use_numba:
                self.x = self._rk4_impl(self.x, u, self.h, self.param_vec)
            else:
                self.x = self._rk4_impl(self.x, u, self.h, self.p)
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
