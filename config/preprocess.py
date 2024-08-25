from gymnasium.wrappers import gray_scale_observation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt
from config.gym import ZeldaGymEnv
from gym.spaces import Box


def PreprocessEnv(config: dict) -> VecFrameStack:
    """
    Preprocesses the environment for reinforcement learning.

    Args:
        config (dict): Configuration parameters for the environment.

    Returns:
        VecFrameStack: Preprocessed environment with stacked frames.
    """

    env = ZeldaGymEnv(config, debug=True)
    env = gray_scale_observation.GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    return env
