# tqc.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    """Construct a simple feed-forward network with configurable activations."""
    layers=[]
    for i in range(len(sizes)-1):
        layers += [nn.Linear(sizes[i], sizes[i+1]),
                   act() if i < len(sizes)-2 else out_act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_limit, hidden=256):
        """Gaussian policy head producing tanh-squashed actions."""
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, 2], act=nn.ReLU)  # outputs mean, log_std (2-d because action is 1-d)
        self.log_std_min, self.log_std_max = -5, 2
        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor, deterministic: bool = False, with_logprob: bool = True):
        """Forward pass returning sampled action and optional log-prob."""
        mu_logstd = self.net(obs)
        mu, log_std = mu_logstd[:, :1], mu_logstd[:, 1:]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if deterministic:
            pi = mu
        else:
            pi = mu + std * torch.randn_like(mu)

        # tanh squashing to [-1,1], then scale to action limit
        pi_tanh = torch.tanh(pi)
        action = self.act_limit * pi_tanh

        if with_logprob:
            # log-prob correction for tanh (SAC)
            # Gaussian log-prob
            pre_sum = -0.5*(((pi - mu)/std)**2 + 2*log_std + math.log(2*math.pi))
            logp = pre_sum.sum(dim=-1, keepdim=True)
            # change-of-variables
            logp_pi = logp - torch.log(1 - pi_tanh.pow(2) + 1e-6)
        else:
            logp_pi = torch.zeros_like(mu)
        return action, logp_pi

class QuantileCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, n_quant=25, hidden=512):
        """Critic head outputting quantile values for the TQC ensemble."""
        super().__init__()
        self.nq = n_quant
        self.q_net = mlp([obs_dim + act_dim, hidden, hidden, hidden, n_quant], act=nn.ReLU)

    def forward(self, obs, act):
        """Return quantile estimates for given state-action pairs."""
        x = torch.cat([obs, act], dim=-1)
        q = self.q_net(x)  # (B, nq), quantile values
        return q

class Replay:
    def __init__(self, size=1_000_000, obs_dim=10):
        """Circular replay buffer mirroring SAC-style storage layout."""
        self.S = np.zeros((size, obs_dim), np.float32)
        self.A = np.zeros((size, 1),  np.float32)
        self.R = np.zeros((size, 1),  np.float32)
        self.S2= np.zeros((size, obs_dim), np.float32)
        self.D = np.zeros((size, 1),  np.float32)
        self.ptr = 0; self.full=False; self.size=size

    def add(self, s,a,r,s2,d):
        """Insert a single transition (s, a, r, s2, done)."""
        self.S[self.ptr]=s; self.A[self.ptr]=a; self.R[self.ptr]=r; self.S2[self.ptr]=s2; self.D[self.ptr]=d
        self.ptr = (self.ptr+1)%self.size
        if self.ptr==0: self.full=True

    def sample(self, batch):
        """Random minibatch with tensors located on the active DEVICE."""
        n = self.size if self.full else self.ptr
        idx = np.random.randint(0, n, size=batch)
        return (
            torch.as_tensor(self.S[idx]).to(DEVICE),
            torch.as_tensor(self.A[idx]).to(DEVICE),
            torch.as_tensor(self.R[idx]).to(DEVICE),
            torch.as_tensor(self.S2[idx]).to(DEVICE),
            torch.as_tensor(self.D[idx]).to(DEVICE),
        )
    
    def __len__(self):
        return self.size if self.full else self.ptr

class TQC:
    def __init__(self, obs_dim=10, act_limit=50.0, gamma=0.99, lr=3e-4,
                 n_quant=25, n_critics=2, truncate_top_frac=0.1, tau=0.005, target_entropy=-1.0):
        """Implementation of Truncated Quantile Critics (TQC) for 1-D torque control."""
        self.act_limit=act_limit
        self.gamma=gamma
        self.tau=tau
        self.nq=n_quant
        self.trunc_frac = truncate_top_frac
        self.n_critics = n_critics

        self.actor=Actor(obs_dim, act_limit).to(DEVICE)
        self.critics=nn.ModuleList([QuantileCritic(obs_dim, 1, n_quant) for _ in range(n_critics)]).to(DEVICE)
        self.critics_tgt=nn.ModuleList([QuantileCritic(obs_dim, 1, n_quant) for _ in range(n_critics)]).to(DEVICE)
        self.critics_tgt.load_state_dict(self.critics.state_dict())

        # Separate optimizers for actor and critic ensemble.
        self.pi_opt=optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt=optim.Adam(self.critics.parameters(), lr=lr)

        # temperature for entropy
        self.log_alpha = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        self.a_opt = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = target_entropy

        # fixed quantile fractions (uniform)
        # Fixed quantile fractions with uniform spacing between 0 and 1.
        taus = (torch.arange(self.nq, device=DEVICE, dtype=torch.float32)+0.5)/self.nq
        self.taus = taus.view(1, self.nq)  # (1, nq)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """Policy inference helper that returns a NumPy action."""
        obs = torch.as_tensor(obs, device=DEVICE).unsqueeze(0)
        a, _ = self.actor(obs, deterministic=deterministic, with_logprob=False)
        return a.squeeze(0).cpu().numpy()

    def _soft_update(self, net, tgt):
        """Polyak-average parameters into the target network."""
        for p, tp in zip(net.parameters(), tgt.parameters()):
            tp.data.mul_(1-self.tau).add_(self.tau*p.data)

    def _huber(self, x, k=1.0):
        """Differentiable Huber loss used inside the quantile regression term."""
        return torch.where(x.abs()<=k, 0.5*x.pow(2), k*(x.abs()-0.5*k))

    def train_step(self, replay: Replay, batch=256):
        """Run one gradient step for critics, actor, and entropy temperature."""
        S,A,R,S2,D = replay.sample(batch)

        # ------- Critic update (distributional quantile regression + truncation) -------
        with torch.no_grad():
            # next action + entropy
            A2, logp2 = self.actor(S2, deterministic=False, with_logprob=True)
            alpha = self.log_alpha.exp()
            # target critics: compute quantiles, then take elementwise min across critics (TQC concatenates & drops top quantiles)
            target_quants=[]
            for ct in self.critics_tgt:
                target_quants.append(ct(S2, A2))  # (B, nq)
            target_quants = torch.stack(target_quants, dim=1)  # (B, Ncrit, nq)
            # Concatenate quantiles along quantile axis
            tq = target_quants.reshape(S2.shape[0], -1)  # (B, Ncrit*nq)
            # sort and drop largest top-k quantiles
            tq_sorted, _ = torch.sort(tq, dim=1)
            total_quants = tq_sorted.shape[1]
            # Drop the highest quantiles across the ensemble as per TQC.
            drop_per = max(1, int(math.floor(self.trunc_frac * self.nq)))
            drop = min(total_quants - 1, drop_per * self.n_critics)
            tq_kept = tq_sorted[:, :total_quants-drop]
            # Soft value: mean of kept quantiles minus entropy term
            V_tgt = tq_kept.mean(dim=1, keepdim=True) - alpha*logp2
            # 1-step distributional target = r + gamma*(1-d)*V_tgt (broadcasted as quantile targets)
            target = R + (1.0 - D)*self.gamma*V_tgt

        q_loss=0.0
        for qc in self.critics:
            q = qc(S, A)  # (B, nq)
            # quantile regression loss: tilted Huber
            # expand target to all quantiles
            y = target.expand_as(q).detach()
            u = y - q  # TD error per quantile
            # taus shape (1,nq) -> (B,nq)
            taus = self.taus.expand_as(q)
            delta = (u < 0.0).float()  # indicator for asymmetric quantile weighting
            quantile_weight = torch.abs(taus - delta)
            loss = (quantile_weight * self._huber(u)).mean()
            q_loss = q_loss + loss

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # ------- Actor + temperature update -------
        a, logp = self.actor(S, deterministic=False, with_logprob=True)
        alpha = self.log_alpha.exp().detach()

        # Allow gradients from critics to flow back to the actor (required for policy improvement)
        q_vals = [qc(S, a).mean(dim=1, keepdim=True) for qc in self.critics]  # mean across quantiles per critic
        q_pi = torch.stack(q_vals, dim=0).min(dim=0)[0]
        pi_loss = (alpha*logp - q_pi).mean()

        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        # temperature
        # Temperature update (SAC-style); pushes entropy toward target_entropy
        logp_detached = logp.detach()
        # Encourage entropy toward the desired target (SAC temperature update).
        alpha_loss = -(self.log_alpha * (logp_detached + self.target_entropy)).mean()
        self.a_opt.zero_grad()
        alpha_loss.backward()
        self.a_opt.step()

        # targets
        self._soft_update(self.critics, self.critics_tgt)

        return dict(q_loss=float(q_loss.item()),
                    pi_loss=float(pi_loss.item()),
                    alpha=float(self.log_alpha.exp().item()),
                    logp=float(logp_detached.mean().item()),
                    entropy=float((-logp_detached).mean().item()))
