# Copilot Instructions

Reinforcement learning agent (Stable-Baselines3 PPO) that plays *The Legend of Zelda: Link's Awakening* (Game Boy) through the PyBoy emulator. Requires Python 3.11.

## Commands

Always use `uv` for environment setup, installing packages, and running anything. Do not call `pip`, `python`, or `venv` directly. There is no `pyproject.toml`, so `uv run` uses the project-local `.venv`.

```bash
uv venv --python 3.11
uv pip install -r requirements.txt

uv run main.py train                                   # trains PPO; checkpoints -> checkpoints/, TensorBoard -> logs/
uv run main.py test ./checkpoints/best_model_5000.zip  # runs a saved checkpoint forever
uv run tensorboard --logdir logs/

uv run black . && uv run isort .                       # formatting (previously enforced by a CI workflow)
```

When adding a dependency, run `uv pip install <pkg>` and pin it in `requirements.txt`.

There is no test suite or CI. The ROM must exist at `roms/ZeldaLinksAwakening.gb` (gitignored, along with `checkpoints/` and `logs/`); any run will fail without it.

## Architecture

The environment stack is assembled across three files:

1. `config/gym.py` – `ZeldaGymEnv(gym.Env)` (Gymnasium API) wraps a `PyBoy` instance. It owns the action mapping, observation building, reward ("fitness") computation, and game-over detection, all by reading emulator RAM via `self.pyboy.memory[addr]`.
2. `config/memory_addresses.py` – `ADDR_*` constants for the game's RAM map (sourced from datacrystal.romhacking.net). Single addresses are ints; multi-byte regions are lists of ints. Imported into `gym.py` with `from .memory_addresses import *`.
3. `utils.py` – `PreprocessEnv(config)` wraps the env: `ZeldaGymEnv` → `DictGrayScaleObservation` (converts only the `"screen"` key to grayscale) → `DummyVecEnv` → `VecFrameStack(4)`. Also defines `CheckpointAndLoggingCallback`, which saves `best_model_{n_calls}.zip` every `check_freq` steps and logs `game/*` metrics to TensorBoard (it reads RAM via `training_env.get_attr("pyboy")`).

`main.py` holds the single `config` dict (ROM path, dirs, `action_freq`, `exploration_reward`, `reward_scale`, `game_with_sound`) and PPO hyperparameters, and dispatches on `sys.argv[1]` (`train`/`test`).

## Key conventions and gotchas

- **Observations are a `Dict` space** (`screen`, `current_room_layout`, `items_in_hand`, `items_in_inventory`, `health`, `rupees`), so PPO uses `"MultiInputPolicy"`. When adding an observation key, update both `observation_space` in `ZeldaGymEnv.__init__` and the dict returned by `_get_observation()`.
- **Reward is a fitness delta.** `_calculate_fitness()` recomputes total fitness from scratch from current RAM state each step, and `step()` returns `self._fitness - self._previous_fitness`. New reward terms should be absolute state-derived quantities added to `_fitness`, scaled by `self.reward_scale`. Update the "Reward System" section of `README.md` when changing rewards.
- `_check_new_items()` also sets `self.moving_things_in_inventory` as a side effect, which `_calculate_fitness()` reads afterward; keep that call order.
- `self.items` uses string keys (`"01"`…`"0D"`) while `pyboy.memory[...]` returns ints, so the `in self.items` checks currently never match.
- `run_action()` overwrites `self.action_freq = 24`, ignoring the config value. Each action is held for 8 frames, then released, and only the last tick renders.
- `ZeldaGymEnv` is created with `debug=True` in `PreprocessEnv`, which keeps real-time emulation speed; `set_emulation_speed(0)` (unthrottled) only applies when `debug=False`.
- `ZeldaGymEnv.step()` returns the Gymnasium 5-tuple, but `main.py` test mode talks to the `VecEnv`, so it uses the SB3 VecEnv API (`reset()` → obs, `step()` → 4-tuple).
- Add new RAM addresses to `memory_addresses.py` as `ADDR_*` constants with a comment explaining value encoding, rather than hardcoding hex in logic (the callback's `0xDB5A` is an existing exception).

## Aliases

- **`cap`** – commit and push: stage the relevant changes, commit with a concise message, and `git push` to the current branch. Never add a `Co-authored-by` trailer or any other Copilot co-author attribution to the commit.
