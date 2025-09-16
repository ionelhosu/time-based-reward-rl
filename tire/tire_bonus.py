from dataclasses import dataclass
from typing import Tuple, List, Optional
import math
import torch

@dataclass
class TIREConfig:
    beta: float = 0.1
    kappa: float = 500.0
    lam: float = 0.5
    delta: float = 0.5
    max_ep_memory: int = 4096
    decay: str = "exp"  # "exp" | "hyper" | "sqrt"

class EpisodicNovelty:
    """Per-env episodic memory of embeddings with L2 threshold gating."""
    def __init__(self, cfg: TIREConfig, emb_dim=128, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.emb_dim = emb_dim
        self.buffers: List[torch.Tensor] = []
        self.ts: List[int] = []

    def reset_envs(self, n_envs: int):
        self.buffers = [torch.empty((0, self.emb_dim), device=self.device) for _ in range(n_envs)]
        self.ts = [0 for _ in range(n_envs)]

    def reset_one(self, env_id: int):
        self.buffers[env_id] = torch.empty((0, self.emb_dim), device=self.device)
        self.ts[env_id] = 0

    def step_time(self, dones):
        for i, d in enumerate(dones):
            if d: self.ts[i] = 0
            else: self.ts[i] += 1

    def novelty_and_bonus(self, z: torch.Tensor):
        N = z.size(0)
        novelty = torch.zeros(N, device=z.device)
        bonus = torch.zeros(N, device=z.device)
        for i in range(N):
            buf = self.buffers[i]
            is_novel = 1
            if buf.numel() > 0:
                dists = torch.cdist(z[i:i+1], buf, p=2).squeeze(0)
                if (dists <= self.cfg.delta).any():
                    is_novel = 0
            if is_novel:
                if buf.size(0) >= self.cfg.max_ep_memory:
                    buf = buf[1:]
                self.buffers[i] = torch.cat([buf, z[i:i+1]], dim=0)
                novelty[i] = 1.0
            tau = float(self.ts[i])
            if self.cfg.decay == "exp":
                f = math.exp(-tau / self.cfg.kappa)
            elif self.cfg.decay == "hyper":
                f = 1.0 / (1.0 + tau / self.cfg.kappa)
            else:
                f = 1.0 / math.sqrt(1.0 + tau / self.cfg.kappa)
            bonus[i] = self.cfg.beta * f * novelty[i]
        return novelty, bonus

def mix_rewards(r_ext: torch.Tensor, r_int: torch.Tensor, lam: float) -> torch.Tensor:
    return r_ext + lam * r_int
