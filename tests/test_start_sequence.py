import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import main
from config import memory_addresses as ram
from config.gym import ZeldaGymEnv


class StartupPyBoy:
    def __init__(self, delay=60, occupied_slots=0, stall=None):
        self.memory = bytearray(65536)
        self.memory[ram.ADDR_SAVE_FILES_EXIST] = occupied_slots
        self.screen = SimpleNamespace(ndarray=np.zeros((144, 160, 3), dtype=np.uint8))
        self.frames = 0
        self.delay = delay
        self.stall = stall
        self.running = True
        self.buttons = []
        self.stage = "boot"
        self.ready_at = delay
        self.pending_button = None
        self.held_button = None
        self.release_at = 0
        self.save_calls = 0
        self.load_calls = 0

    def button(self, button, delay):
        self.buttons.append((self.stage, button, delay))
        self.pending_button = (button, delay)

    def tick(self, count, render):
        if not self.running:
            return False
        for _ in range(count):
            self.frames += 1
            if self.frames >= self.release_at:
                self.held_button = None
            if self.ready_at is not None and self.frames >= self.ready_at:
                self._finish_loading()
            if self.pending_button is not None:
                button, delay = self.pending_button
                self.pending_button = None
                if self.held_button != button:
                    self._press(button)
                self.held_button = button
                self.release_at = self.frames + delay
        if render:
            self.screen.ndarray.fill(self.frames % 256)
        return True

    def _load(self, stage, gameplay_type):
        self.stage = stage
        self.memory[ram.ADDR_GAMEPLAY_TYPE] = gameplay_type
        self.memory[ram.ADDR_GAMEPLAY_SUBTYPE] = 0
        self.ready_at = None if stage == self.stall else self.frames + self.delay

    def _finish_loading(self):
        self.ready_at = None
        if self.stage == "boot":
            self.stage = "title"
        elif self.stage == "file-loading":
            self.stage = "file"
            self.memory[ram.ADDR_GAMEPLAY_SUBTYPE] = 7
        elif self.stage == "name-loading":
            self.stage = "name"
            self.memory[ram.ADDR_GAMEPLAY_SUBTYPE] = 2
        elif self.stage == "world-loading":
            self.stage = "waking"
            self.memory[ram.ADDR_GAMEPLAY_SUBTYPE] = 7
            self.memory[ram.ADDR_CURRENT_HEALTH] = 24
            if self.stall != "waking":
                self.ready_at = self.frames + self.delay
        elif self.stage == "waking":
            self.stage = "awake"
            self.memory[ram.ADDR_DIALOG_STATE] = 0x40

    def _press(self, button):
        if self.stage == "title" and button == "start":
            self._load("file-loading", 2)
        elif self.stage == "file":
            if button == "down":
                rows = 4 if self.memory[ram.ADDR_SAVE_FILES_EXIST] else 3
                self.memory[ram.ADDR_SAVE_SLOT] = (
                    self.memory[ram.ADDR_SAVE_SLOT] + 1
                ) % rows
            elif button == "start":
                slot = self.memory[ram.ADDR_SAVE_SLOT]
                if self.memory[ram.ADDR_SAVE_FILES_EXIST] & (1 << slot):
                    self._load("world-loading", 0x0B)
                else:
                    self._load("name-loading", 3)
        elif self.stage == "name":
            if button == "a":
                self.memory[ram.ADDR_NAME_ENTRY_POSITION] += 1
            elif button == "start":
                self.memory[ram.ADDR_SAVE_FILES_EXIST] |= (
                    1 << self.memory[ram.ADDR_SAVE_SLOT]
                )
                self.memory[ram.ADDR_SAVE_SLOT] = 0
                self._load("file-loading", 2)

    def save_state(self, state):
        self.save_calls += 1
        state.write(self.memory)
        state.write(self.frames.to_bytes(4, "little"))

    def load_state(self, state):
        self.load_calls += 1
        self.memory[:] = state.read(65536)
        self.frames = int.from_bytes(state.read(4), "little")
        self.screen.ndarray.fill(self.frames % 256)


class StartSequenceTests(unittest.TestCase):
    def make_env(self, pyboy):
        config = copy.deepcopy(main.config)
        config["rom_path"] = "unused.gb"
        config["game_with_sound"] = False
        with patch("config.gym.PyBoy", return_value=pyboy):
            return ZeldaGymEnv(config, debug=True)

    def test_startup_waits_for_menus_and_wake_up(self):
        for delay in (1, 75, 310):
            with self.subTest(delay=delay):
                pyboy = StartupPyBoy(delay=delay)
                env = self.make_env(pyboy)

                observation, info = env.reset()

                self.assertEqual(pyboy.stage, "awake")
                self.assertAlmostEqual(observation["health"][0], 3 / 14)
                self.assertEqual(sum(info["credits"].values()), 0)
                self.assertEqual(pyboy.memory[ram.ADDR_NAME_ENTRY_POSITION], 1)
                self.assertEqual(pyboy.save_calls, 1)
                self.assertIsNone(pyboy.held_button)
                self.assertTrue(all(delay == 8 for _, _, delay in pyboy.buttons))
                self.assertFalse(
                    any(stage.endswith("loading") for stage, _, _ in pyboy.buttons)
                )
                self.assertFalse(
                    any(stage in ("waking", "awake") for stage, _, _ in pyboy.buttons)
                )

    def test_existing_saves_are_preserved(self):
        for occupied_slots, expected_slot in ((1, 1), (3, 2), (5, 1)):
            with self.subTest(occupied_slots=occupied_slots):
                pyboy = StartupPyBoy(occupied_slots=occupied_slots)
                env = self.make_env(pyboy)

                env.reset()

                self.assertEqual(pyboy.memory[ram.ADDR_SAVE_SLOT], expected_slot)
                self.assertEqual(
                    pyboy.memory[ram.ADDR_SAVE_FILES_EXIST],
                    occupied_slots | (1 << expected_slot),
                )
                self.assertIn(("name", "a", 8), pyboy.buttons)

    def test_reset_restores_wake_up_without_replaying_inputs(self):
        pyboy = StartupPyBoy()
        env = self.make_env(pyboy)
        observation, _ = env.reset()
        initial_screen = observation["screen"].copy()
        initial_memory = bytes(pyboy.memory)
        initial_frames = pyboy.frames
        initial_buttons = list(pyboy.buttons)

        for _ in range(2):
            pyboy.memory[ram.ADDR_CURRENT_HEALTH] = 0
            pyboy.memory[ram.ADDR_DIALOG_STATE] = 0
            pyboy.frames += 100
            pyboy.screen.ndarray.fill(123)
            env._fitness = 100
            env._previous_fitness = 90
            env._ep.ledger.credit("new_room", 2)
            env._ep.visited_rooms.add((1, 0x55, 0x55))

            observation, _ = env.reset()

            self.assertEqual(bytes(pyboy.memory), initial_memory)
            np.testing.assert_array_equal(observation["screen"], initial_screen)
            self.assertEqual(pyboy.frames, initial_frames)
            self.assertEqual(pyboy.buttons, initial_buttons)
            self.assertEqual(env._fitness, 0)
            self.assertEqual(env._previous_fitness, 0)
            self.assertEqual(env._ep.ledger.fitness(), 0)
            self.assertNotIn((1, 0x55, 0x55), env._ep.visited_rooms)
        self.assertEqual(pyboy.save_calls, 1)
        self.assertEqual(pyboy.load_calls, 2)

    def test_full_save_slots_fail_without_loading_an_existing_game(self):
        pyboy = StartupPyBoy(occupied_slots=7)
        env = self.make_env(pyboy)

        with self.assertRaisesRegex(RuntimeError, "all three save slots are occupied"):
            env.reset()

        self.assertEqual(pyboy.stage, "file")
        self.assertIsNone(env._initial_state)

    def test_stalled_menu_or_missing_wake_up_dialogue_times_out(self):
        for stage in ("file-loading", "waking"):
            with self.subTest(stage=stage):
                pyboy = StartupPyBoy(stall=stage)
                env = self.make_env(pyboy)

                with self.assertRaisesRegex(RuntimeError, "Startup timed out"):
                    env.reset()

                self.assertEqual(pyboy.frames, 7200)
                self.assertIsNone(env._initial_state)

    def test_stopped_emulator_fails_explicitly(self):
        pyboy = StartupPyBoy()
        pyboy.running = False
        env = self.make_env(pyboy)

        with self.assertRaisesRegex(RuntimeError, "Emulation stopped"):
            env.reset()

        self.assertIsNone(env._initial_state)


if __name__ == "__main__":
    unittest.main()
