import tempfile
import unittest
import warnings
from collections import defaultdict
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from test_rewards import FakePyBoy, make_config, wake_memory

from config import memory_addresses as ram
from utils import CheckpointAndLoggingCallback, PreprocessEnv


class FakeLogger:
    def __init__(self):
        self.means = defaultdict(list)
        self.values = {}

    def record_mean(self, key, value):
        self.means[key].append(value)

    def record(self, key, value):
        self.values[key] = value


def make_vec_env(**overrides):
    pyboy = FakePyBoy()
    wake_memory(pyboy.memory)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with patch("config.gym.PyBoy", return_value=pyboy):
            venv = PreprocessEnv(make_config(**overrides))
    venv.venv.envs[0].unwrapped._initial_state = BytesIO(bytes(pyboy.memory))
    return venv, pyboy.memory


class LegacySpaceEnv(gym.Env):
    """Observation space used by checkpoints trained before the reward rework."""

    observation_space = Dict(
        {
            "screen": Box(0, 255, (144, 160), np.uint8),
            "current_room_layout": Box(0, 255, (156,), np.uint8),
            "items_in_hand": Box(0, 255, (2,), np.uint8),
            "items_in_inventory": Box(0, 255, (9,)),
            "health": Box(0, 16, (1,), np.uint8),
            "rupees": Box(0, 255, (1,), np.uint8),
        }
    )
    action_space = Discrete(7)


class VectorStackTests(unittest.TestCase):
    def test_timeout_reaches_sb3_with_final_observation_and_episode_metrics(self):
        venv, memory = make_vec_env(episode__max_seconds=2)
        logger = FakeLogger()
        model = SimpleNamespace(
            get_env=lambda: venv, logger=logger, save=Mock(), num_timesteps=0
        )
        with tempfile.TemporaryDirectory() as directory:
            callback = CheckpointAndLoggingCallback(
                check_freq=1000, save_path=directory
            )
            callback.init_callback(model)
            observation = venv.reset()

            episode_return, steps = 0.0, 0
            while True:
                if steps == 1:
                    memory[ram.ADDR_IS_INDOOR] = 0
                observation, rewards, dones, infos = venv.step(np.array([0]))
                callback.update_locals(
                    {"rewards": rewards, "dones": dones, "infos": infos}
                )
                callback.on_step()
                episode_return += rewards[0]
                steps += 1
                if dones[0]:
                    break

        info = infos[0]
        self.assertEqual(info["truncation_reason"], "max_duration")
        self.assertTrue(info["TimeLimit.truncated"])
        terminal = info["terminal_observation"]
        self.assertEqual(terminal["screen"].shape, observation["screen"].shape[1:])
        self.assertEqual(terminal["health"].shape, (4,))
        # Leaving the house (+10), the new room (+2), and the first overworld tile.
        self.assertAlmostEqual(episode_return, 12.02, places=5)
        self.assertAlmostEqual(logger.means["game/episode_return"][0], 12.02, places=5)
        self.assertEqual(logger.means["game/episode_length"], [steps])
        self.assertEqual(logger.means["game/house_exit_rate"], [1.0])
        self.assertEqual(logger.means["game/sword_rate"], [0.0])
        self.assertEqual(logger.means["game/timeout_rate/max_duration"], [1.0])
        self.assertEqual(logger.values["game/availability/enemy_defeat"], 0.0)
        self.assertNotIn("game/credits/enemy_defeat", logger.means)
        self.assertEqual(logger.means["game/credits/house_exit"], [10])
        self.assertNotIn("game/debits/house_exit", logger.means)
        self.assertIn("game/debits/stagnation", logger.means)
        self.assertEqual(callback.episode_returns[0], 0)

    def test_callback_logs_raw_returns_under_reward_normalization(self):
        venv, memory = make_vec_env(episode__max_seconds=2)
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True)
        logger = FakeLogger()
        model = SimpleNamespace(
            get_env=lambda: venv, logger=logger, save=Mock(), num_timesteps=0
        )
        with tempfile.TemporaryDirectory() as directory:
            callback = CheckpointAndLoggingCallback(
                check_freq=1000, save_path=directory
            )
            callback.init_callback(model)
            venv.reset()
            memory[ram.ADDR_IS_INDOOR] = 0
            normalized = 0.0
            while True:
                _, rewards, dones, infos = venv.step(np.array([0]))
                normalized += rewards[0]
                callback.update_locals(
                    {"rewards": rewards, "dones": dones, "infos": infos}
                )
                callback.on_step()
                if dones[0]:
                    break
        raw = logger.means["game/episode_return"][0]
        self.assertGreater(raw, 12)
        self.assertNotAlmostEqual(raw, normalized)

    def test_window_and_sound_require_a_single_environment(self):
        invalid = [
            ({"n_envs": 2, "show_window": True}, False),
            ({"n_envs": 2, "show_window": False}, True),
            ({"n_envs": 1, "show_window": False}, True),
            ({"n_envs": 0, "show_window": False}, False),
        ]
        for kwargs, sound in invalid:
            with self.subTest(kwargs=kwargs, sound=sound):
                with self.assertRaises(ValueError):
                    PreprocessEnv(make_config(game_with_sound=sound), **kwargs)

    def test_checkpoints_load_only_with_matching_observation_spaces(self):
        venv, _ = make_vec_env()
        legacy = VecFrameStack(DummyVecEnv([LegacySpaceEnv]), 4)
        with tempfile.TemporaryDirectory() as directory:
            PPO("MultiInputPolicy", venv, n_steps=8).save(f"{directory}/current")
            PPO("MultiInputPolicy", legacy, n_steps=8).save(f"{directory}/legacy")

            PPO.load(f"{directory}/current", env=venv)
            with self.assertRaisesRegex(ValueError, "Observation spaces do not match"):
                PPO.load(f"{directory}/legacy", env=venv)


if __name__ == "__main__":
    unittest.main()
