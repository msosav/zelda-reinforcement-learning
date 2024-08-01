import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent


class ZeldaGymEnv(gym.Env):

    def __init__(self, config: dict, debug=False):
        super().__init__()
        self.rom_path = config['rom_path']
        self.state_path = config['state_path']

        try:
            self.pyboy = PyBoy(self.rom_path)
        except (FileNotFoundError):
            raise SystemExit("You should have your ROM in the roms/ folder")

        self._fitness = 0
        self._previous_fitness = 0
        self.debug = debug

        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(16, 20), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self.valid_actions))

        # TODO: Implement game initialization

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        if action == 0:
            pass
        else:
            self.pyboy.send_input(self.valid_actions[action])

        self.pyboy.tick(1)

        # TODO: Implement game over logic
        done = self.pyboy.game_wrapper.game_over

        self._calculate_fitness()
        reward = self._fitness-self._previous_fitness

        observation = self.pyboy.game_area()
        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def _calculate_fitness(self):
        self._previous_fitness = self._fitness

        # TODO: Implement reward logic
        self._fitness = 0

    # TODO: Implement reset game
    def reset(self, **kwargs):
        self.pyboy.game_wrapper.reset_game()
        self._fitness = 0
        self._previous_fitness = 0

        observation = self.pyboy.game_area()
        info = {}
        return observation, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.pyboy.stop()
