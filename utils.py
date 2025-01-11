import os

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3.common.callbacks import \
    BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from config.gym import ZeldaGymEnv


def PreprocessEnv(config: dict) -> VecFrameStack:
    """
    Preprocesses the environment for reinforcement learning.

    Args:
        config (dict): Configuration parameters for the environment.

    Returns:
        VecFrameStack: Preprocessed environment with stacked frames.
    """

    env = ZeldaGymEnv(config, debug=True)
    env = DictGrayScaleObservation(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4)

    return env


class DictGrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, keep_dim: bool = False):
        super().__init__(env)
        self.keep_dim = keep_dim

        assert isinstance(env.observation_space,
                          gym.spaces.Dict), "Observation space must be a Dict"

        obs_shape = env.observation_space["screen"].shape[:2]

        if self.keep_dim:
            self.observation_space["screen"] = Box(
                low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
            )
        else:
            self.observation_space["screen"] = Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8
            )

    def observation(self, observation):
        import cv2

        observation["screen"] = cv2.cvtColor(
            observation["screen"], cv2.COLOR_RGB2GRAY)

        if self.keep_dim:
            observation["screen"] = np.expand_dims(observation["screen"], -1)

        return observation


class CheckpointAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_lengths = []

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        # Log episode info
        if self.locals.get('done'):
            self.logger.record('game/episode_reward',
                               self.locals.get('rewards')[0])
            self.logger.record('game/episode_length', self.n_calls)
            self.logger.record('game/current_health',
                               self.training_env.get_attr('pyboy')[0].memory[0xDB5A])

        # Save model checkpoint
        if self.n_calls % self.check_freq == 0:
            model_path = f"{self.save_path}/best_model_{self.n_calls}.zip"
            self.model.save(model_path)

        return True
