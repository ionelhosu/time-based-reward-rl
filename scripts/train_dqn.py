import torch
from tire.envs.wrappers import make_atari_env
from tire.algo.dqn import DQNRunner
from tire.tire_bonus import TIREConfig

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_atari_env("GravitarNoFrameskip-v4", frameskip=4, sticky=0.25)
    runner = DQNRunner(env, device=device, cfg=TIREConfig(beta=0.1, kappa=500.0, lam=0.5, delta=0.5))
    runner.train(total_steps=500_000)
