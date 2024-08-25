from gym.wrappers import gray_scale_observation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt
from config.gym import ZeldaGymEnv
from gym.spaces import Box


def Preprocess(config: dict) -> DummyVecEnv:
    env = ZeldaGymEnv(config, debug=True)
    env = gray_scale_observation.GrayScaleObservation(env, keep_dim=True)
    # env = DummyVecEnv([lambda: env])
    # env = VecFrameStack(env, n_stack=4, channel_order='last')

    return env
