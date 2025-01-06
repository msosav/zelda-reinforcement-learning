import os

from gymnasium.wrappers import gray_scale_observation
from stable_baselines3.common.callbacks import BaseCallback  # For saving models
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
    env = gray_scale_observation.GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order="last")

    return env


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
