import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, Dict
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from .memory_addresses import *


class ZeldaGymEnv(gym.Env):

    def __init__(self, config: dict, debug=False):
        super().__init__()
        self.rom_path = config["rom_path"]

        assert self.rom_path is not None, "ROM path is required"

        self.pyboy = PyBoy(self.rom_path, sound=config["game_with_sound"])

        self._fitness = 0
        self._previous_fitness = 0
        self.debug = debug

        self.action_freq = config["action_freq"]

        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

        self.observation_space = Dict({
            'screen': Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8),
            'current_room_layout': Box(low=0, high=255, shape=(156,), dtype=np.uint8),
            'items_in_hand': Box(low=0, high=255, shape=(2,), dtype=np.uint8),
            'health': Box(low=0, high=16, shape=(1,), dtype=np.uint8),
            'rupees': Box(low=0, high=999, shape=(1,), dtype=np.uint8),
        })

        self.action_space = Discrete(len(self.valid_actions))

        self.items = {
            "01": False,  # Sword
            "02": False,  # Bombs
            "03": False,  # Power bracelet
            "04": False,  # Shield
            "05": False,  # Bow
            "06": False,  # Hookshot
            "07": False,  # Fire rod
            "08": False,  # Pegasus boots
            "09": False,  # Ocarina
            "0A": False,  # Feather
            "0B": False,  # Shovel
            "0C": False,  # Magic powder
            "0D": False,  # Boomrang
        }

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        if action == 0:
            pass
        else:
            self.pyboy.button(self.valid_actions[action])

    def step(self, action):
        self.run_action(action)

        done = self.__game_over()

        self._calculate_fitness()
        reward = self._fitness - self._previous_fitness

        observation = self._get_observation()

        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def run_action(self, action):
        self.pyboy.send_input(self.valid_actions[action])
        press_step = 8

        self.action_freq = 24

        self.pyboy.tick(press_step)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.action_freq - press_step - 1)
        self.pyboy.tick(1, True)

    def __game_over(self):
        if self.pyboy.memory[ADDR_CURRENT_HEALTH] == 0:
            return True
        return False

    def _calculate_fitness(self):
        self._previous_fitness = self._fitness

        self._fitness = 0

        self._fitness += self._check_new_items()

        # TODO: Sword and shield level

    def start_sequence(self):
        self.pyboy.button("start")
        self.pyboy.tick()
        self.pyboy.button("start")
        self.pyboy.tick()
        self.pyboy.button("start")
        self.pyboy.tick()
        self.pyboy.button("a")
        self.pyboy.tick()
        self.pyboy.button("start")
        self.pyboy.tick()
        self.pyboy.button("start")
        self.pyboy.tick()

    def reset(self, **kwargs):
        self.start_sequence()

        self._fitness = 0
        self._previous_fitness = 0

        observation = self._get_observation()

        info = {}
        return observation, info

    def render(self, mode="human"):
        pass

    def close(self):
        self.pyboy.stop()

    def _check_new_items(self):
        items_in_inventory_count = 0
        for inventory_address in ADDR_INVENTORY:
            item_in_inventory = self.pyboy.memory[inventory_address]
            if item_in_inventory in self.items:
                self.items[item_in_inventory] = True
                items_in_inventory_count += 1

        for held_address in ADDR_HELD_ITEMS:
            item_in_hand = self.pyboy.memory[held_address]

            if item_in_hand in self.items:
                self.items[item_in_hand] = True
                items_in_inventory_count += 1

        return items_in_inventory_count

    def _check_rupees(self):
        rupees = 0
        for addr in ADDR_RUPEES:
            rupees += self.pyboy.memory[addr]

        return rupees

    def _get_observation(self):
        # Image observation
        screen = self._get_screen()

        room_type = self.pyboy.memory[ADDR_DESTINATION_BYTE_1]

        room_number = self.pyboy.memory[ADDR_DESTINATION_BYTE_3]

        current_room_layout = [
            self.pyboy.memory[addr] for addr in ADDR_CURRENTLY_LOADED_MAP
        ]

        health = [self.pyboy.memory[ADDR_CURRENT_HEALTH] / 8]

        rupees = [self._check_rupees()]

        items_in_inventory = sum(
            [1 for item in self.items if self.items[item]])

        items_in_hand = [self.pyboy.memory[addr]
                         for addr in ADDR_HELD_ITEMS]

        obs = {
            'screen': screen,
            'current_room_layout': current_room_layout,
            'items_in_hand': items_in_hand,
            'health': health,
            'rupees': rupees
        }

        return obs

    def _get_screen(self):
        return self.pyboy.screen.ndarray
