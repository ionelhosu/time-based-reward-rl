import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..models.policy_value import CNNPolicyValue
from ..models.encoder import RandomEncoder
from ..tire_bonus import TIREConfig, EpisodicNovelty, mix_rewards

def to_torch(x, device): return torch.as_tensor(x, dtype=torch.float32, device=device)

class PPORunner:
    def __init__(self, envs, action_space, device="cpu", cfg=TIREConfig(),
                 rollout_len=128, epochs=4, minibatch=256, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.1, ent_coef=0.01, vf_coef=0.5, lr=2.5e-4):
        self.envs = envs
        self.N = len(envs)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollout_len = rollout_len
        self.epochs = epochs
        self.minibatch = minibatch
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.model = CNNPolicyValue(in_channels=4, num_actions=action_space.n).to(device)
        self.enc = RandomEncoder(in_channels=4, out_dim=128).to(device).eval()
        self.tire_cfg = cfg
        self.novelty = EpisodicNovelty(cfg, emb_dim=128, device=device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

    def _reset_all(self, seed=0):
        obs = []
        for i, env in enumerate(self.envs):
            o, _ = env.reset(seed=seed+i)
            obs.append(np.array(o))
        self.novelty.reset_envs(self.N)
        return np.stack(obs, axis=0)

    def _step_all(self, actions):
        obs2, rew, done = [], [], []
        for (env, a) in zip(self.envs, actions):
            o2, r, d, tr, _ = env.step(int(a))
            d = d or tr
            if d:
                o2, _ = env.reset()
            obs2.append(np.array(o2)); rew.append(r); done.append(d)
        return np.stack(obs2, 0), np.array(rew, np.float32), np.array(done, bool)

    def train(self, total_steps=200_000, log_every=10_000):
        obs = self._reset_all()
        step = 0
        while step < total_steps:
            obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
            for t in range(self.rollout_len):
                x = to_torch(np.transpose(obs, (0,3,1,2)), self.device)
                logits, v = self.model(x)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample(); logp = dist.log_prob(a)

                obs2, r_ext_np, done_np = self._step_all(a.detach().cpu().numpy())

                with torch.no_grad():
                    z = self.enc(x)
                _, r_int = self.novelty.novelty_and_bonus(z)
                self.novelty.step_time(done_np)

                r_mix = mix_rewards(to_torch(r_ext_np, self.device), r_int, self.tire_cfg.lam)

                obs_buf.append(x.cpu().numpy()); act_buf.append(a.cpu().numpy())
                logp_buf.append(logp.cpu().numpy()); val_buf.append(v.cpu().numpy())
                rew_buf.append(r_mix.cpu().numpy()); done_buf.append(done_np.astype(np.float32))

                obs = obs2
                step += self.N

            with torch.no_grad():
                x_last = to_torch(np.transpose(obs, (0,3,1,2)), self.device)
                _, v_last = self.model(x_last)

            obs_b = torch.from_numpy(np.concatenate(obs_buf,0)).to(self.device)
            act_b = torch.from_numpy(np.concatenate(act_buf,0)).to(self.device).long()
            logp_b = torch.from_numpy(np.concatenate(logp_buf,0)).to(self.device)
            val_b = torch.from_numpy(np.concatenate(val_buf,0)).to(self.device).squeeze(-1)
            rew_b = torch.from_numpy(np.concatenate(rew_buf,0)).to(self.device)
            done_b = torch.from_numpy(np.concatenate(done_buf,0)).to(self.device)

            B = obs_b.size(0)
            adv = torch.zeros(B, device=self.device); ret = torch.zeros(B, device=self.device)
            next_value = v_last.mean().item()
            gae = 0.0
            for i in reversed(range(B)):
                mask = 1.0 - done_b[i].item()
                delta = rew_b[i].item() + self.gamma * next_value * mask - val_b[i].item()
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                adv[i] = gae; next_value = val_b[i].item()
            ret = adv + val_b
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            inds = np.arange(B)
            for _ in range(self.epochs):
                np.random.shuffle(inds)
                for start in range(0, B, self.minibatch):
                    mb = inds[start:start+self.minibatch]
                    logits, v = self.model(obs_b[mb])
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(act_b[mb])
                    ratio = (logp - logp_b[mb]).exp()
                    surr1 = ratio * adv[mb]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv[mb]
                    loss_pi = -torch.min(surr1, surr2).mean()
                    loss_v = F.mse_loss(v, ret[mb])
                    loss_ent = dist.entropy().mean()
                    loss = loss_pi + self.vf_coef * loss_v - self.ent_coef * loss_ent
                    self.optim.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optim.step()

            if step % log_every == 0:
                print(f"[PPO] steps={step:,}")
