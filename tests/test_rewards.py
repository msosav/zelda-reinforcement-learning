import copy
import math
import unittest
import warnings
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import main
from config import memory_addresses as ram
from config.gym import FRAMES_PER_SECOND, ZeldaGymEnv

WINDOW_FRAMES = round(15 * FRAMES_PER_SECOND)


class FakePyBoy:
    """RAM-only emulator stand-in; tests mutate memory between steps."""

    def __init__(self):
        self.memory = bytearray(0x10000)
        self.screen = SimpleNamespace(ndarray=np.zeros((144, 160, 4), dtype=np.uint8))
        self.inputs = []
        self.on_frame = None

    def tick(self, count=1, render=True):
        for _ in range(count):
            if self.on_frame is not None:
                self.on_frame(self.memory)
        return True

    def send_input(self, event):
        self.inputs.append(event)

    def set_emulation_speed(self, speed):
        pass

    def save_state(self, state):
        state.write(bytes(self.memory))

    def load_state(self, state):
        self.memory[:] = state.read(0x10000)

    def stop(self):
        pass


def wake_memory(memory):
    memory[ram.ADDR_GAMEPLAY_TYPE] = 0x0B
    memory[ram.ADDR_GAMEPLAY_SUBTYPE] = 0x07
    memory[ram.ADDR_CURRENT_HEALTH] = 0x18
    memory[ram.ADDR_MAX_HEALTH] = 3
    memory[ram.ADDR_WINDOW_Y] = 0x80
    memory[ram.ADDR_IS_INDOOR] = 1
    memory[ram.ADDR_MAP_ID] = 0x10
    memory[ram.ADDR_MAP_ROOM] = 0xA3
    memory[ram.ADDR_LINK_X] = 0x38
    memory[ram.ADDR_LINK_Y] = 0x34


def make_config(**overrides):
    config = copy.deepcopy(main.config)
    config["rom_path"] = "unused.gb"
    config["game_with_sound"] = False
    for path, value in overrides.items():
        section, _, key = path.partition("__")
        if key:
            config[section][key] = value
        else:
            config[section] = value
    return config


def make_env(setup=None, **overrides):
    pyboy = FakePyBoy()
    wake_memory(pyboy.memory)
    if setup is not None:
        setup(pyboy.memory)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with patch("config.gym.PyBoy", return_value=pyboy):
            env = ZeldaGymEnv(make_config(**overrides))
        env._initial_state = BytesIO(bytes(pyboy.memory))
        env.reset()
    return env, pyboy.memory


def bcd(value):
    return (value // 10) << 4 | value % 10


def steps_for(seconds, action_freq=24):
    return math.ceil(seconds * FRAMES_PER_SECOND / action_freq)


def run(env, steps, mutate=None):
    results = []
    for index in range(steps):
        if mutate is not None:
            mutate(env.pyboy.memory, index)
        results.append(env.step(0))
        if results[-1][2] or results[-1][3]:
            break
    return results


def total(results, component=None):
    if component is None:
        return sum(result[1] for result in results)
    return sum(result[4]["reward_components"][component] for result in results)


class AcquisitionAndVisitTests(unittest.TestCase):
    def test_equipment_ids_are_credited_once_across_all_slots(self):
        env, memory = make_env(lambda m: m.__setitem__(0xDB00, 0x04))
        memory[0xDB01] = 0x01
        memory[0xDB05] = 0x01
        result = env.step(0)
        self.assertEqual(result[4]["reward_components"]["unique_equipment"], 20)
        self.assertIsNotNone(result[4]["milestones"]["sword_seconds"])

        memory[0xDB01] = 0x00
        memory[0xDB0B] = 0x04
        memory[ram.ADDR_POTION] = 1
        self.assertEqual(total(run(env, 2)), 0)

        memory[ram.ADDR_FLIPPERS] = 1
        self.assertEqual(total(run(env, 2), "unique_equipment"), 20)

    def test_last_inventory_slot_and_upgrades(self):
        env, memory = make_env()
        memory[0xDB0B] = 0x0A
        memory[ram.ADDR_SWORD_LEVEL] = 1
        result = env.step(0)
        self.assertEqual(result[4]["reward_components"]["unique_equipment"], 20)
        self.assertEqual(result[4]["reward_components"]["equipment_upgrade"], 0)
        memory[ram.ADDR_SWORD_LEVEL] = 2
        self.assertEqual(env.step(0)[4]["reward_components"]["equipment_upgrade"], 20)

    def test_dungeon_records_end_before_equipment_fields(self):
        addresses = [a for record in ram.ADDR_DUNGEON_ITEM_FLAGS for a in record]
        self.assertEqual((min(addresses), max(addresses)), (0xDB16, 0xDB3D))
        self.assertNotIn(ram.ADDR_POWER_BRACELET_LEVEL, addresses)

    def test_visited_bit_is_tested_and_baseline_is_seeded(self):
        env, memory = make_env(lambda m: m.__setitem__(0xD8A2, 0x80))
        self.assertEqual(env.step(0)[1], 0)
        memory[0xD8C2] = 0x10
        memory[0xD8B2] = 0x90
        self.assertEqual(env.step(0)[4]["reward_components"]["new_room"], 2)
        self.assertEqual(total(run(env, 3)), 0)

    def test_room_identity_separates_map_contexts(self):
        env, memory = make_env()
        memory[ram.ADDR_IS_INDOOR] = 0
        memory[ram.ADDR_MAP_ID] = 0
        result = env.step(0)
        components = result[4]["reward_components"]
        self.assertEqual((components["new_room"], components["house_exit"]), (2, 10))

        memory[ram.ADDR_IS_INDOOR] = 1
        memory[ram.ADDR_MAP_ID] = 0x11
        self.assertEqual(env.step(0)[4]["reward_components"]["new_room"], 2)
        memory[ram.ADDR_MAP_ID] = 0x10
        self.assertEqual(env.step(0)[1], 0)
        memory[ram.ADDR_IS_INDOOR] = 0
        memory[ram.ADDR_MAP_ID] = 0
        self.assertEqual(env.step(0)[1], 0)

    def test_forced_room_change_is_not_a_completed_entry(self):
        env, memory = make_env()
        memory[ram.ADDR_IS_INDOOR] = 0
        memory[ram.ADDR_LINK_MOTION_STATE] = 0x04
        self.assertEqual(env.step(0)[1], 0)
        memory[ram.ADDR_LINK_MOTION_STATE] = 0
        components = env.step(0)[4]["reward_components"]
        self.assertEqual(components["new_room"] + components["house_exit"], 12)

    def test_flags_set_during_a_transition_do_not_hide_the_first_entry(self):
        env, memory = make_env()
        memory[ram.ADDR_IS_INDOOR] = 0
        memory[0xD8A3] = 0x80
        memory[ram.ADDR_ROOM_TRANSITION_STATE] = 2
        self.assertEqual(env.step(0)[4]["reward_components"]["new_room"], 2)
        memory[ram.ADDR_ROOM_TRANSITION_STATE] = 0
        components = env.step(0)[4]["reward_components"]
        self.assertEqual((components["new_room"], components["house_exit"]), (0, 10))
        self.assertTrue(env._ep.stagnation.route[-1][2])

    def test_new_tiles_are_credited_once_per_room_within_a_budget(self):
        env, memory = make_env(rewards__tile_budget=0.05)
        self.assertEqual(env.step(0)[1], 0)
        memory[ram.ADDR_LINK_X] = 0x48
        self.assertAlmostEqual(env.step(0)[4]["reward_components"]["new_tile"], 0.02)
        memory[ram.ADDR_LINK_X] = 0x38
        self.assertEqual(env.step(0)[1], 0)
        memory[ram.ADDR_IS_INDOOR] = 0
        components = env.step(0)[4]["reward_components"]
        self.assertAlmostEqual(components["new_tile"], 0.02)
        memory[ram.ADDR_LINK_X] = 0x48
        result = env.step(0)[4]
        self.assertAlmostEqual(result["reward_components"]["new_tile"], 0.01)
        self.assertEqual(result["budgets_remaining"]["new_tile"], 0)
        self.assertGreater(result["progress_age_seconds"], 24 / FRAMES_PER_SECOND)

    def test_instruments_use_the_ownership_bit(self):
        env, memory = make_env()
        memory[0xDB65] = 0x01
        self.assertEqual(env.step(0)[1], 0)
        memory[0xDB65] = 0x03
        memory[0xDB6C] = 0x02
        result = env.step(0)
        self.assertEqual(result[4]["reward_components"]["instrument"], 200)
        self.assertEqual(result[4]["milestones"]["instruments"], 2)


class ResourceTests(unittest.TestCase):
    def test_rupees_decode_packed_bcd_with_strict_peaks(self):
        env, memory = make_env(lambda m: m.__setitem__(0xDB5E, 0x98))
        memory[0xDB5D], memory[0xDB5E] = 0x01, 0x02
        self.assertAlmostEqual(env.step(0)[1], 0.04)
        memory[0xDB5D], memory[0xDB5E] = 0x00, 0x50
        self.assertEqual(env.step(0)[1], 0)
        memory[0xDB5D], memory[0xDB5E] = 0x01, 0x02
        self.assertEqual(env.step(0)[1], 0)
        memory[0xDB5E] = 0x0A
        with self.assertRaisesRegex(ValueError, "BCD"):
            env.step(0)

    def test_consumables_have_separate_peaks(self):
        env, memory = make_env(lambda m: m.__setitem__(ram.ADDR_BOMBS, 0x09))
        memory[ram.ADDR_BOMBS] = 0x10
        memory[ram.ADDR_ARROWS] = 0x02
        self.assertAlmostEqual(env.step(0)[1], 0.02 * 3)
        memory[ram.ADDR_BOMBS] = 0x05
        memory[ram.ADDR_MAGIC_POWDER] = 0x01
        self.assertAlmostEqual(env.step(0)[1], 0.02)

    def test_health_uses_a_strict_high_water_mark(self):
        env, memory = make_env()
        memory[ram.ADDR_CURRENT_HEALTH] = 0x10
        self.assertEqual(env.step(0)[1], 0)
        memory[ram.ADDR_CURRENT_HEALTH] = 0x18
        self.assertEqual(env.step(0)[1], 0)
        memory[ram.ADDR_MAX_HEALTH] = 4
        memory[ram.ADDR_CURRENT_HEALTH] = 0x20
        self.assertAlmostEqual(env.step(0)[4]["reward_components"]["health_peak"], 0.25)

    def test_small_rewards_share_a_budget_but_peaks_still_move(self):
        env, memory = make_env()
        memory[0xDB5D], memory[0xDB5E] = 0x09, 0x99
        result = env.step(0)
        self.assertAlmostEqual(result[1], 2.0)
        self.assertEqual(result[4]["budgets_remaining"]["small_rewards"], 0)
        memory[ram.ADDR_BOMBS] = 0x10
        self.assertEqual(env.step(0)[1], 0)
        self.assertEqual(env._ep.peaks["bombs"], 10)


class HealthPenaltyTests(unittest.TestCase):
    def test_critical_health_boundary_recurrence_and_budget(self):
        env, memory = make_env(rewards__recurring_penalty_budget=0.05)
        memory[ram.ADDR_CURRENT_HEALTH] = 9
        self.assertEqual(env.step(0)[1], 0)
        memory[ram.ADDR_CURRENT_HEALTH] = 8
        per_step = 0.02 * 24 / FRAMES_PER_SECOND
        first, second = run(env, 2)
        self.assertAlmostEqual(first[1], -per_step)
        self.assertAlmostEqual(second[1], -per_step)
        self.assertTrue(first[4]["critical_health"])
        memory[ram.ADDR_CURRENT_HEALTH] = 4
        run(env, 40)
        self.assertAlmostEqual(env._ep.ledger.debits["critical_health"], 0.05)
        self.assertEqual(env.step(0)[4]["budgets_remaining"]["critical_health"], 0)
        memory[ram.ADDR_CURRENT_HEALTH] = 9
        healed = env.step(0)[4]
        self.assertEqual(healed["reward_components"]["critical_health"], 0)
        self.assertFalse(healed["critical_health"])
        self.assertAlmostEqual(env._ep.ledger.debits["critical_health"], 0.05)

    def test_death_is_penalized_once_and_ends_charges(self):
        env, memory = make_env()
        memory[ram.ADDR_CURRENT_HEALTH] = 0
        observation, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["termination_reason"], "death")
        self.assertEqual(reward, -20)
        self.assertEqual(env.step(0)[1], 0)

    def test_critical_charges_do_not_depend_on_action_frequency(self):
        debits = []
        for action_freq in (12, 48):
            env, memory = make_env(action_freq=action_freq)
            memory[ram.ADDR_CURRENT_HEALTH] = 8
            run(env, 960 // action_freq)
            debits.append(env._ep.ledger.debits["critical_health"])
        self.assertAlmostEqual(debits[0], debits[1])
        self.assertAlmostEqual(debits[0], 0.02 * 960 / FRAMES_PER_SECOND)


class StagnationTests(unittest.TestCase):
    def stagnation_steps(self, results):
        return [r[4]["reward_components"]["stagnation"] for r in results]

    def test_stationary_grace_escalation_and_budget(self):
        env, memory = make_env(episode__no_progress_seconds=1000)
        self.assertEqual(total(run(env, WINDOW_FRAMES // 24)), 0)
        rates = [r[4]["stagnation_rate"] for r in run(env, steps_for(62))]
        self.assertAlmostEqual(rates[1], 0.01)
        self.assertAlmostEqual(rates[steps_for(16)], 0.02)
        self.assertAlmostEqual(rates[steps_for(31)], 0.03)
        self.assertAlmostEqual(rates[-1], 0.04)
        run(env, steps_for(200))
        self.assertAlmostEqual(env._ep.ledger.debits["stagnation"], 5.0)

    def test_stagnation_charges_do_not_depend_on_action_frequency(self):
        debits = []
        for action_freq in (12, 48, 600):
            env, memory = make_env(action_freq=action_freq)
            run(env, 3600 // action_freq)
            debits.append(env._ep.ledger.debits["stagnation"])
        # 15 s at 0.01, 15 s at 0.02, then the remainder at 0.03 per second.
        third = 3600 - 3 * WINDOW_FRAMES
        expected = (0.01 + 0.02) * WINDOW_FRAMES + 0.03 * third
        for debit in debits:
            self.assertAlmostEqual(debit * FRAMES_PER_SECOND, expected, delta=0.03 * 36)
        self.assertAlmostEqual(min(debits), max(debits))

    def test_jitter_and_a_single_twitch_do_not_break_stationary(self):
        env, memory = make_env()

        def jitter(m, index):
            m[ram.ADDR_LINK_X] = 0x38 + (3 if index % 2 else 0)
            if index == steps_for(20):
                m[ram.ADDR_LINK_X] = 0x50

        results = run(env, steps_for(30), jitter)
        categories = [r[4]["stagnation_categories"] for r in results]
        first = next(i for i, c in enumerate(categories) if c)
        self.assertTrue(all(c == ["stationary"] for c in categories[first:]))
        self.assertGreater(results[-1][4]["stagnation_rate"], 0)

    def test_pacing_on_a_short_known_route(self):
        env, memory = make_env()
        route = [0x30, 0x38, 0x40, 0x48, 0x50, 0x48, 0x40, 0x38]
        results = run(
            env,
            steps_for(20),
            lambda m, i: m.__setitem__(ram.ADDR_LINK_X, route[i % len(route)]),
        )
        self.assertEqual(results[-1][4]["stagnation_categories"], ["pacing"])
        self.assertLess(total(results, "stagnation"), 0)

    def test_walking_a_long_route_has_no_generic_room_timer_penalty(self):
        env, memory = make_env()
        route = list(range(0x08, 0x98, 8)) + list(range(0x90, 0x08, -8))
        results = run(
            env,
            steps_for(70),
            lambda m, i: m.__setitem__(ram.ADDR_LINK_X, route[i % len(route)]),
        )
        self.assertEqual(total(results, "stagnation"), 0)
        self.assertFalse(results[-1][3])

    def test_alternating_between_visited_rooms(self):
        def seed(m):
            m[ram.ADDR_IS_INDOOR] = 0
            m[ram.ADDR_MAP_ID] = 0
            m[ram.ADDR_MAP_ROOM] = 0xA2
            m[0xD8A2] = m[0xD8B2] = 0x80

        env, memory = make_env(seed)
        results = run(
            env,
            steps_for(25),
            lambda m, i: m.__setitem__(
                ram.ADDR_MAP_ROOM, 0xB2 if (i // 4) % 2 else 0xA2
            ),
        )
        self.assertIn("room_loop", results[-1][4]["stagnation_categories"])
        self.assertLess(total(results), 0)

    def test_repeated_menu_toggling_while_moving(self):
        env, memory = make_env()

        def toggle(m, i):
            m[ram.ADDR_WINDOW_Y] = 0x00 if i % 2 else 0x80
            if i % 2 == 0:
                m[ram.ADDR_LINK_X] = 0x08 + 8 * ((i // 2) % 18)
                m[ram.ADDR_LINK_Y] = 0x10 + 8 * ((i // 36) % 12)

        results = run(env, steps_for(20), toggle)
        self.assertEqual(results[-1][4]["stagnation_categories"], ["menu_loop"])

    def test_dialogue_progress_clears_local_evidence_but_reopening_does_not(self):
        for progressing in (True, False):
            with self.subTest(progressing=progressing):
                env, memory = make_env()

                def talk(m, i):
                    m[ram.ADDR_DIALOG_STATE] = 0x89 if i % 4 else 0x00
                    m[ram.ADDR_DIALOG_INDEX] = i // 4 if progressing else 1

                results = run(env, steps_for(30), talk)
                self.assertEqual(total(results) < 0, not progressing)

    def test_forced_time_is_excluded_from_controllable_clocks(self):
        env, memory = make_env(episode__no_progress_seconds=10)
        memory[ram.ADDR_CURRENT_HEALTH] = 8
        memory[ram.ADDR_ROOM_TRANSITION_STATE] = 1
        results = run(env, steps_for(20))
        info = results[-1][4]
        self.assertEqual(total(results), 0)
        self.assertEqual(info["controllable_frames"], 0)
        self.assertEqual(info["progress_age_seconds"], 0)
        self.assertFalse(results[-1][3])
        self.assertEqual(info["elapsed_frames"], 24 * len(results))


class EpisodeBoundaryTests(unittest.TestCase):
    def test_no_progress_truncation_resets_only_on_durable_progress(self):
        env, memory = make_env()

        def events(m, i):
            m[0xDB5E] = bcd(min(99, i // 16))
            if i == steps_for(100):
                m[0xD800] = 0x80

        results = run(env, steps_for(400), events)
        self.assertTrue(results[-1][3])
        self.assertEqual(results[-1][4]["truncation_reason"], "no_progress")
        self.assertGreater(total(results, "rupee_peak"), 0)
        self.assertAlmostEqual(len(results), steps_for(280), delta=2)

    def test_max_duration_includes_forced_time(self):
        env, memory = make_env(episode__max_seconds=60)
        memory[ram.ADDR_LINK_MOTION_STATE] = 0x03
        results = run(env, steps_for(120))
        self.assertEqual(len(results), steps_for(60))
        self.assertEqual(results[-1][4]["truncation_reason"], "max_duration")
        self.assertEqual(total(results), 0)

    def test_death_takes_precedence_over_timeouts(self):
        env, memory = make_env(episode__max_seconds=1)
        results = run(
            env,
            5,
            lambda m, i: m.__setitem__(ram.ADDR_CURRENT_HEALTH, 0 if i == 2 else 0x18),
        )
        self.assertEqual(len(results), 3)
        _, _, terminated, truncated, info = results[-1]
        self.assertGreaterEqual(info["elapsed_frames"], FRAMES_PER_SECOND)
        self.assertEqual((terminated, truncated), (True, False))
        self.assertIsNone(info["truncation_reason"])

    def test_reset_clears_and_reseeds_everything(self):
        env, memory = make_env()
        memory[ram.ADDR_IS_INDOOR] = 0
        memory[0xD8A2] = 0x80
        memory[0xDB00] = 0x04
        memory[ram.ADDR_CURRENT_HEALTH] = 8
        run(env, steps_for(30))
        self.assertNotEqual(env._fitness, 0)

        _, info = env.reset()
        self.assertEqual(env._fitness, 0)
        self.assertEqual(sum(info["credits"].values()), 0)
        self.assertEqual(sum(info["debits"].values()), 0)
        self.assertEqual(info["unique_rooms"], 1)
        self.assertEqual(env.step(0)[1], 0)


class AccountingAndInterfaceTests(unittest.TestCase):
    def test_component_deltas_reconcile_with_reward(self):
        env, memory = make_env(reward_scale=0.5)

        def chaos(m, i):
            m[ram.ADDR_CURRENT_HEALTH] = 8 if i % 50 < 25 else 0x18
            m[0xD800 + i % 40] |= 0x80 if i % 7 == 0 else 0
            m[0xDB5E] = (i % 10) * 0x10
            m[ram.ADDR_LINK_X] = 0x38 + (i % 3)

        results = run(env, steps_for(60), chaos)
        for result in results:
            self.assertAlmostEqual(
                sum(result[4]["reward_components"].values()), result[1]
            )
        self.assertAlmostEqual(sum(r[1] for r in results), env._fitness)
        self.assertAlmostEqual(
            env._fitness,
            sum(results[-1][4]["credits"].values())
            - sum(results[-1][4]["debits"].values()),
        )

    def test_observations_match_their_space(self):
        env, memory = make_env()
        memory[ram.ADDR_CURRENT_HEALTH] = 0x14
        memory[0xDB5D], memory[0xDB5E] = 0x09, 0x99
        memory[0xDB0B] = 0x0D
        observation = env.step(0)[0]
        self.assertTrue(env.observation_space.contains(observation))
        self.assertAlmostEqual(observation["health"][0], 2.5 / 14)
        self.assertEqual(observation["rupees"][0], 1)
        self.assertEqual(observation["items_in_inventory"][-1], 1)

    def test_unsupported_encodings_fail_explicitly(self):
        for address, value in ((0xDB03, 0x0E), (0xDB66, 0x04), (0xDB4E, 3)):
            with self.subTest(address=hex(address)):
                env, memory = make_env()
                memory[address] = value
                with self.assertRaises(ValueError):
                    env.step(0)

    def test_invalid_configuration_fails_explicitly(self):
        invalid = [
            {"action_freq": 8},
            {"reward_scale": 0},
            {"rewards__death": -1},
            {"stagnation__window_seconds": math.nan},
            {"episode": {"max_seconds": 10}},
            {"optional_signals__enemy_defeat": True},
            {"optional_signals__story_milestones": ["sword"]},
        ]
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    make_env(**overrides)

    def test_disabled_signals_are_reported(self):
        with self.assertWarnsRegex(UserWarning, "enemy_defeat"):
            with patch("config.gym.PyBoy", return_value=FakePyBoy()):
                ZeldaGymEnv(make_config())
        env, _ = make_env()
        availability = env.step(0)[4]["availability"]
        self.assertFalse(availability["enemy_defeat"])
        self.assertTrue(availability["house_exit"])


if __name__ == "__main__":
    unittest.main()
