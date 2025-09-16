import torch, numpy as np
from tire.envs.wrappers import make_atari_env
from tire.algo.ppo import PPORunner
from tire.tire_bonus import TIREConfig

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    envs = [make_atari_env("MontezumaRevengeNoFrameskip-v4", frameskip=4, sticky=0.25) for _ in range(8)]
    runner = PPORunner(envs, envs[0].action_space, device=device,
                       cfg=TIREConfig(beta=0.1, kappa=500.0, lam=0.5, delta=0.5))
    runner.train(total_steps=200_000)
