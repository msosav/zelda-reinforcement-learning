import os

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import unwrap_vec_normalize
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack

from config.gym import FRAMES_PER_SECOND, PENALTY_COMPONENTS, ZeldaGymEnv


def PreprocessEnv(
    config: dict, debug: bool = True, n_envs: int = 1, show_window: bool = True
) -> VecFrameStack:
    """
    Preprocesses the environment for reinforcement learning.

    Args:
        config (dict): Configuration parameters for the environment.
        debug (bool): Keep real-time emulation speed; False runs unthrottled.
        n_envs (int): Number of emulators; more than one runs each in its own process.
        show_window (bool): Show the game window (only with a single emulator).

    Returns:
        VecFrameStack: Preprocessed environment with stacked frames.
    """
    if isinstance(n_envs, bool) or not isinstance(n_envs, int) or n_envs < 1:
        raise ValueError(f"n_envs must be a positive integer, got {n_envs!r}")
    if n_envs > 1 and (show_window or config["game_with_sound"]):
        raise ValueError("The game window and sound require a single environment")
    if config["game_with_sound"] and not show_window:
        raise ValueError("Sound requires the game window")

    def make_env():
        env = ZeldaGymEnv(config, debug=debug, show_window=show_window)
        # Keep a channel axis so VecFrameStack yields a 144x160x4 image for the CNN.
        return DictGrayScaleObservation(env, keep_dim=True)

    vec_env_class = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    env = vec_env_class([make_env] * n_envs)
    env = VecFrameStack(env, 4)

    return env


class DictGrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, keep_dim: bool = False):
        super().__init__(env)
        self.keep_dim = keep_dim

        assert isinstance(
            env.observation_space, gym.spaces.Dict
        ), "Observation space must be a Dict"

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

        observation["screen"] = cv2.cvtColor(observation["screen"], cv2.COLOR_RGB2GRAY)

        if self.keep_dim:
            observation["screen"] = np.expand_dims(observation["screen"], -1)

        return observation


class CheckpointAndLoggingCallback(BaseCallback):
    """Saves checkpoints and logs per-episode game metrics from vectorized terminal infos."""

    def __init__(self, check_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.last_save = 0
        self.episode_returns = None
        self.episode_lengths = None

    def _init_callback(self) -> None:
        # Save into a folder named after the TensorBoard run (e.g. checkpoints/PPO_10).
        log_dir = getattr(self.logger, "dir", None)
        if self.save_path is not None and log_dir:
            self.save_path = os.path.join(self.save_path, os.path.basename(log_dir))
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        self.episode_returns = np.zeros(self.training_env.num_envs)
        self.episode_lengths = np.zeros(self.training_env.num_envs, dtype=np.int64)

    def _on_step(self):
        # Log raw game rewards even when VecNormalize scales them for training.
        vec_normalize = unwrap_vec_normalize(self.training_env)
        if vec_normalize is None:
            rewards = self.locals["rewards"]
        else:
            rewards = vec_normalize.get_original_reward()
        self.episode_returns += rewards
        self.episode_lengths += 1
        for index, done in enumerate(self.locals["dones"]):
            if done:
                # The terminal info was produced before DummyVecEnv reset the emulator.
                self._log_episode(
                    self.episode_returns[index],
                    self.episode_lengths[index],
                    self.locals["infos"][index],
                )
                self.episode_returns[index] = 0.0
                self.episode_lengths[index] = 0

        # Counted in environment steps across all parallel environments.
        if self.num_timesteps - self.last_save >= self.check_freq:
            self.last_save = self.num_timesteps
            model_path = f"{self.save_path}/best_model_{self.num_timesteps}.zip"
            self.model.save(model_path)

        return True

    def _log_episode(self, episode_return, episode_length, info):
        # Go-Explore episodes start mid-game; keep them apart from bed starts.
        start = info.get("start", "initial")
        prefix = "game/" if start == "initial" else "game/archive/"

        def record(key, value):
            self.logger.record_mean(prefix + key[len("game/") :], value)

        self.logger.record_mean("game/archive_start_rate", float(start == "archive"))
        if "archive_cells" in info:
            self.logger.record("game/archive_cells", info["archive_cells"])
        record("game/episode_return", float(episode_return))
        record("game/episode_length", int(episode_length))
        if "milestones" not in info:
            return

        milestones = info["milestones"]
        record("game/episode_seconds", info["elapsed_frames"] / FRAMES_PER_SECOND)
        for name in ("house_exit", "sword"):
            seconds = milestones[f"{name}_seconds"]
            record(f"game/{name}_rate", float(seconds is not None))
            if seconds is not None:
                record(f"game/{name}_seconds", seconds)
        record("game/instruments", milestones["instruments"])
        record("game/unique_rooms", info["unique_rooms"])
        record("game/new_rooms", info["new_rooms"])
        record("game/critical_health_seconds", info["critical_health_seconds"])
        for name, seconds in info["stagnation_seconds"].items():
            record(f"game/stagnation_seconds/{name}", seconds)

        record("game/death_rate", float(info["termination_reason"] == "death"))
        for reason in ("no_progress", "max_duration"):
            record(
                f"game/timeout_rate/{reason}",
                float(info["truncation_reason"] == reason),
            )

        for name, available in info["availability"].items():
            if not available or name not in info["credits"]:
                continue
            if name in PENALTY_COMPONENTS:
                record(f"game/debits/{name}", info["debits"][name])
            else:
                record(f"game/credits/{name}", info["credits"][name])
        for name, remaining in info["budgets_remaining"].items():
            record(f"game/budget_saturated/{name}", float(remaining <= 1e-9))
        for name, available in info["availability"].items():
            self.logger.record(f"game/availability/{name}", float(available))
