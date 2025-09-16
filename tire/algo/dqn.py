import numpy as np, random
import torch
import torch.nn.functional as F

from ..models.dqn import CNNDQN
from ..models.encoder import RandomEncoder
from ..tire_bonus import TIREConfig, EpisodicNovelty, mix_rewards
from ..utils.replay import Replay

def to_torch(x, device): return torch.as_tensor(x, dtype=torch.float32, device=device)

class DQNRunner:
    def __init__(self, env, device="cpu", cfg=TIREConfig(),
                 gamma=0.99, batch=32, lr=1e-4, target_tau=10_000,
                 start_eps=1.0, end_eps=0.01, eps_decay_steps=1_000_000):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.batch = batch
        self.online = CNNDQN(in_channels=4, num_actions=env.action_space.n, dueling=True).to(device)
        self.target = CNNDQN(in_channels=4, num_actions=env.action_space.n, dueling=True).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.enc = RandomEncoder(in_channels=4, out_dim=128).to(device).eval()
        self.tire_cfg = cfg
        self.novelty = EpisodicNovelty(cfg, emb_dim=128, device=device)
        self.replay = Replay()
        self.target_tau = target_tau
        self.start_eps, self.end_eps, self.eps_decay = start_eps, end_eps, eps_decay_steps

    def epsilon(self, step):
        t = min(1.0, step / self.eps_decay)
        return self.start_eps + t * (self.end_eps - self.start_eps)

    def _reset(self, seed=0):
        o, _ = self.env.reset(seed=seed)
        self.novelty.reset_envs(1)
        return np.array(o)

    def train(self, total_steps=500_000, learn_start=50_000):
        s = self._reset()
        step = 0
        while step < total_steps:
            eps = self.epsilon(step)
            if random.random() < eps:
                a = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    q = self.online(to_torch(np.transpose(s, (2,0,1))[None, ...], self.device))
                    a = int(q.argmax(dim=1).item())

            s2, r_ext, done, tr, _ = self.env.step(a)
            done = done or tr
            s2 = np.array(s2)

            with torch.no_grad():
                z = self.enc(to_torch(np.transpose(s, (2,0,1))[None, ...], self.device))
            _, r_int = self.novelty.novelty_and_bonus(z)
            self.novelty.step_time(np.array([done], bool))
            r = float(r_ext) + self.tire_cfg.lam * float(r_int.item())

            self.replay.add(s, a, r, s2, float(done))
            s = s2
            step += 1

            if done:
                s = self._reset()

            if len(self.replay) >= max(self.batch, learn_start):
                sb, ab, rb, s2b, db = self.replay.sample(self.batch)
                sb = to_torch(np.transpose(sb, (0,3,1,2)), self.device)
                s2b = to_torch(np.transpose(s2b, (0,3,1,2)), self.device)
                ab = torch.as_tensor(ab, device=self.device, dtype=torch.long)
                rb = torch.as_tensor(rb, device=self.device)
                db = torch.as_tensor(db, device=self.device)

                with torch.no_grad():
                    qn = self.online(s2b)
                    amax = qn.argmax(dim=1)
                    q_tgt = self.target(s2b).gather(1, amax.unsqueeze(1)).squeeze(1)
                    y = rb + self.gamma * (1.0 - db) * q_tgt

                q = self.online(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(q, y)

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
                self.optim.step()

                if step % self.target_tau == 0:
                    self.target.load_state_dict(self.online.state_dict())
                    print(f"[DQN] steps={step:,}  loss={loss.item():.4f}")
