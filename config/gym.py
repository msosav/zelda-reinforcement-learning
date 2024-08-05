import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from .memory_addresses import *


class ZeldaGymEnv(gym.Env):

    def __init__(self, config: dict, debug=False):
        super().__init__()
        self.rom_path = config['rom_path']
        self.state_path = config['state_path']

        assert self.rom_path is not None, "ROM path is required"

        self.pyboy = PyBoy(self.rom_path)

        self._fitness = 0
        self._previous_fitness = 0
        self.debug = debug

        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.valid_actions = ['', 'a', 'b', 'left', 'right',
                              'up', 'down', 'start', 'select']

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(16, 20), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self.valid_actions))

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        if action == 0:
            pass
        else:
            self.pyboy.button(self.valid_actions[action])

        self.pyboy.tick()

        done = self.__game_over()

        self._calculate_fitness()
        reward = self._fitness-self._previous_fitness

        observation = self.pyboy.game_area()
        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def __game_over(self):
        if self.pyboy.memory[ADDR_CURRENT_HEALTH] == 0:
            return True
        return False

    def _calculate_fitness(self):
        self._previous_fitness = self._fitness

        # TODO: Implement reward logic
        self._fitness = 0

    def reset(self, **kwargs):
        try:
            with open(self.state_path, 'rb') as state_file:
                self.pyboy.load_state(state_file)
        except (FileNotFoundError):
            pass

        self._fitness = 0
        self._previous_fitness = 0

        observation = self.pyboy.game_area()
        info = {}
        return observation, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()
