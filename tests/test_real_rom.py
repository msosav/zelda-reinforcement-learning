import copy
import os
import sys
import unittest
import warnings
from io import BytesIO
from unittest.mock import patch

import main
from config import memory_addresses as ram
from config.gym import FRAMES_PER_SECOND, ZeldaGymEnv

DOWN, LEFT, RIGHT, UP, A, B, START, DOWN_B, LEFT_B, RIGHT_B, UP_B = range(11)

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
    ),
)
ROM = main.config["rom_path"]


@unittest.skipUnless(
    os.environ.get("ZELDA_REAL_ROM_TESTS") == "1" and os.path.exists(ROM),
    "set ZELDA_REAL_ROM_TESTS=1 and provide the ROM to run real-ROM traces",
)
class RealRomTraceTests(unittest.TestCase):
    """Drives the real game through the scenarios the reward signals depend on."""

    def setUp(self):
        from pyboy import PyBoy

        config = copy.deepcopy(main.config)
        config["game_with_sound"] = False
        config["go_explore"]["start_probability"] = 0.0
        pyboy = PyBoy(ROM, window="null", sound_emulated=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with patch("config.gym.PyBoy", return_value=pyboy):
                self.env = ZeldaGymEnv(config)
        self.memory = pyboy.memory
        self.trace = []
        self.last = None

    def tearDown(self):
        # stop() would otherwise write cartridge RAM (save slots) next to the ROM.
        self.env.pyboy.stop(save=False)

    def step(self, action):
        result = self.env.step(action)
        self.last = result
        info = result[4]
        events = {k: round(v, 4) for k, v in info["reward_components"].items() if v}
        if events or info["termination_reason"] or info["truncation_reason"]:
            self.trace.append(
                (
                    info["elapsed_frames"],
                    f"x={self.memory[ram.ADDR_LINK_X]:02X}",
                    f"y={self.memory[ram.ADDR_LINK_Y]:02X}",
                    events,
                    info["stagnation_categories"],
                    info["termination_reason"] or info["truncation_reason"],
                )
            )
        return result

    def dialog_open(self):
        return self.memory[ram.ADDR_DIALOG_STATE] & ram.DIALOG_STATE_MASK != 0

    def close_dialog(self, limit=300):
        # Follow-up messages (such as the item message) can open after a short gap.
        closed_steps = 0
        for index in range(limit):
            closed_steps = 0 if self.dialog_open() else closed_steps + 1
            if closed_steps > 10:
                return
            self.step(A if index % 2 == 0 else B)
        self.fail(f"dialogue did not close: {self.trace}")

    def goto(self, x, y, limit=120):
        for _ in range(limit):
            if self.dialog_open():
                self.close_dialog()
            dx = x - self.memory[ram.ADDR_LINK_X]
            dy = y - self.memory[ram.ADDR_LINK_Y]
            if abs(dx) <= 4 and abs(dy) <= 4:
                return
            if abs(dx) > 4:
                self.step(RIGHT if dx > 0 else LEFT)
            else:
                self.step(DOWN if dy > 0 else UP)
        self.fail(f"could not reach ({x:02X}, {y:02X}): {self.trace}")

    def walk_until_room(self, action, room, limit=60):
        for _ in range(limit):
            self.step(action)
            if self.env._ep.room == room:
                return
        self.fail(f"did not reach room {room}: {self.trace}")

    def test_wake_up_to_overworld_and_back(self):
        _, info = self.env.reset()
        self.assertEqual(self.env._ep.room, ram.MARIN_HOUSE_ROOM)
        self.assertEqual(info["unique_rooms"], 1)
        self.assertTrue(info["availability"]["house_exit"])

        # Marin's dialogue is input-advanceable: it stays controllable and records progress.
        self.close_dialog()
        self.assertGreaterEqual(len(self.env._ep.dialog_keys), 5)
        self.assertEqual(
            self.last[4]["controllable_frames"], self.last[4]["elapsed_frames"]
        )

        # Pushing the south wall is stationary stagnation once the window fills.
        self.goto(0x46, 0x70)
        for _ in range(50):
            self.step(DOWN)
        self.assertEqual(self.memory[ram.ADDR_LINK_Y], 0x70)
        self.assertIn("stationary", self.last[4]["stagnation_categories"])
        self.assertGreater(self.env._ep.ledger.debits["stagnation"], 0)

        # Tarin's shield arrives as item ID 04 with shield level 1, credited once.
        self.goto(0x50, 0x70)
        self.goto(0x50, 0x54)
        self.goto(0x68, 0x54)
        self.step(RIGHT)
        self.step(A)
        self.close_dialog()
        credits = self.env._ep.ledger.credits
        self.assertEqual(credits["unique_equipment"], 20)
        self.assertEqual(credits["equipment_upgrade"], 0)
        self.assertEqual(self.memory[ram.ADDR_SHIELD_LEVEL], 1)

        # Repeatedly opening the inventory: slide frames are forced, open time is not,
        # so the 15-second window needs well over 15 seconds of toggling to fill.
        before = self.last[4]
        for _ in range(160):
            self.step(START)
        for _ in range(10):
            window_y = self.memory[ram.ADDR_WINDOW_Y]
            if window_y == ram.WINDOW_Y_HIDDEN:
                break
            self.step(START if window_y == ram.WINDOW_Y_INVENTORY_OPEN else B)
        after = self.last[4]
        forced = (after["elapsed_frames"] - before["elapsed_frames"]) - (
            after["controllable_frames"] - before["controllable_frames"]
        )
        self.assertGreater(forced, 0)
        self.assertIn("menu_loop", after["stagnation_categories"], self.trace[-3:])

        # Leaving the house is a completed overworld entry worth the exit and a new room.
        self.goto(0x50, 0x54)
        self.goto(0x50, 0x70)
        self.walk_until_room(DOWN, (0, 0, 0xA2))
        self.assertEqual(credits["house_exit"], 10)
        self.assertEqual(credits["new_room"], 2)
        self.assertIsNotNone(self.last[4]["milestones"]["house_exit_seconds"])
        self.walk_until_room(DOWN, (0, 0, 0xB2))
        self.assertEqual(credits["new_room"], 4)
        self.assertTrue(self.memory[0xD8A2] & 0x80 and self.memory[0xD8B2] & 0x80)

        # Alternating between the two known rooms becomes a room loop.
        start = self.last[4]["controllable_frames"]
        # Scrolling is forced, so each crossing adds only a few controllable frames.
        for _ in range(100):
            self.walk_until_room(UP, (0, 0, 0xA2))
            self.walk_until_room(DOWN, (0, 0, 0xB2))
            if "room_loop" in self.last[4]["stagnation_categories"]:
                break
        self.assertEqual(credits["new_room"], 4)
        self.assertIn("room_loop", self.last[4]["stagnation_categories"])
        looped = (self.last[4]["controllable_frames"] - start) / FRAMES_PER_SECOND
        self.assertGreaterEqual(looped, 15)

        # Damage and healing through the game's own health buffers.
        self.memory[0xDB94] = self.memory[ram.ADDR_CURRENT_HEALTH] - 8
        for _ in range(4):
            self.step(B)
        self.assertEqual(self.memory[ram.ADDR_CURRENT_HEALTH], 8)
        self.assertTrue(self.last[4]["critical_health"])
        critical = self.env._ep.ledger.debits["critical_health"]
        self.assertGreater(critical, 0)
        self.memory[0xDB93] = 8
        for _ in range(4):
            self.step(B)
        self.assertFalse(self.last[4]["critical_health"])
        self.assertEqual(self.last[4]["reward_components"]["critical_health"], 0)

        self.memory[0xDB94] = 0x20
        for _ in range(10):
            observation, reward, terminated, truncated, info = self.step(B)
            if terminated:
                break
        self.assertTrue(terminated, self.trace)
        self.assertEqual(info["reward_components"]["death"], -20)
        self.assertEqual(self.env._ep.ledger.debits["death"], 20)
        self.assertEqual(self.memory[ram.ADDR_LINK_MOTION_STATE], 0x07)

        # Reset restores the wake-up snapshot and reseeds every baseline.
        observation, info = self.env.reset()
        self.assertEqual(sum(info["credits"].values()), 0)
        self.assertEqual(sum(info["debits"].values()), 0)
        self.assertEqual(info["unique_rooms"], 1)
        self.assertTrue(self.env.observation_space.contains(observation))
        self.assertEqual(self.step(B)[1], 0)

    def test_shield_push_moves_sea_urchins_on_the_beach(self):
        import map_reachability as mapper

        # Find room E1 (above the beach) with the item-free search, then start there.
        probe = mapper.Probe()
        try:
            probe.leave_house_with_shield()
            _, rooms, _, _ = mapper.explore(probe, 40, 2000, target=(0, 0, 0xE1))
        finally:
            probe.pyboy.stop(save=False)
        self.env._initial_state = BytesIO(rooms[(0, 0, 0xE1)])

        def push(action):
            self.env.reset()
            self.goto(0x58, 0x3C)
            self.goto(0x6C, 0x3C)
            for _ in range(60):
                self.step(action)
                self.memory[ram.ADDR_CURRENT_HEALTH] = 0x18
                if self.memory[ram.ADDR_LINK_Y] >= 0x60:
                    break
            return self.memory[ram.ADDR_LINK_Y]

        # The urchins below the gap block plain walking but yield to the raised shield.
        self.assertLess(push(DOWN), 0x50)
        self.assertGreaterEqual(push(DOWN_B), 0x60)


if __name__ == "__main__":
    unittest.main()
