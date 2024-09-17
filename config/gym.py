import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from .memory_addresses import *


class ZeldaGymEnv(gym.Env):

    def __init__(self, config: dict, debug=False):
        super().__init__()
        self.rom_path = config["rom_path"]

        assert self.rom_path is not None, "ROM path is required"

        self.pyboy = PyBoy(self.rom_path)

        self._fitness = 0
        self._previous_fitness = 0
        self.debug = debug

        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.valid_actions = [
            "",
            "a",
            "b",
            "left",
            "right",
            "up",
            "down",
            "start",
            "select",
        ]

        self.observation_space = Box(
            low=0, high=255, shape=(144, 160, 3), dtype=np.uint8
        )

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

        self.pyboy.tick()

        done = self.__game_over()

        self._calculate_fitness()
        reward = self._fitness - self._previous_fitness

        observation = self.pyboy.screen.ndarray

        info = {}
        truncated = False

        return observation, reward, done, truncated, info

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

        observation = self.pyboy.screen.ndarray

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

        return items_in_inventory_count
