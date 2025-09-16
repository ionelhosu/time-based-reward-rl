import gymnasium as gym

def make_atari_env(game_id="MontezumaRevengeNoFrameskip-v4", frameskip=4, sticky=0.25):
    from gymnasium.wrappers import AtariPreprocessing, FrameStack
    env = gym.make(game_id, frameskip=frameskip, repeat_action_probability=sticky)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1,
                             noop_max=30, terminal_on_life_loss=False)
    env = FrameStack(env, 4)
    return env
