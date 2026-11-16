# Copilot Instructions

Reinforcement learning agent (Stable-Baselines3 PPO) that plays *The Legend of Zelda: Link's Awakening* (Game Boy) through the PyBoy emulator. Requires Python 3.11.

## Commands

Always use `uv` for environment setup, installing packages, and running anything. Do not call `pip`, `python`, or `venv` directly. There is no `pyproject.toml`, so `uv run` uses the project-local `.venv`.

```bash
uv venv --python 3.11
uv pip install -r requirements.txt

uv run main.py train                                   # trains PPO; checkpoints -> checkpoints/<run>/, TensorBoard -> logs/
uv run main.py test ./checkpoints/PPO_10/best_model_50000.zip  # runs a saved checkpoint forever
uv run tensorboard --logdir logs/

uv run black . && uv run isort .                       # formatting (previously enforced by a CI workflow)

uv run python -m unittest discover -s tests            # mocked-RAM tests; no ROM needed
ZELDA_REAL_ROM_TESTS=1 uv run python -m unittest tests/test_real_rom.py  # scripted trace on the real ROM
```

When adding a dependency, run `uv pip install <pkg>` and pin it in `requirements.txt`.

There is no CI. Tests use `unittest` with a fake `PyBoy` (`tests/test_rewards.py`) that tests mutate between steps. The ROM must exist at `roms/ZeldaLinksAwakening.gb` (gitignored, along with `checkpoints/` and `logs/`); any run will fail without it.

## Architecture

The environment stack is assembled across three files:

1. `config/gym.py` – `ZeldaGymEnv(gym.Env)` (Gymnasium API) wraps a `PyBoy` instance. It owns the action mapping, observation building, reward ledger (`RewardLedger`), stagnation detection (`StagnationTracker`), episode limits, and game-over detection, all by reading emulator RAM via `self.pyboy.memory[addr]`. `validate_config()` checks every reward, stagnation, and episode setting.
2. `config/memory_addresses.py` – `ADDR_*` constants for the game's RAM map (sourced from datacrystal.romhacking.net). Single addresses are ints; multi-byte regions are lists of ints. Imported into `gym.py` with `from .memory_addresses import *`.
3. `utils.py` – `PreprocessEnv(config, debug=True, n_envs=1, show_window=True)` wraps the env: `ZeldaGymEnv` → `DictGrayScaleObservation(keep_dim=True)` (converts only the `"screen"` key to grayscale, keeping a channel axis so stacked frames are a 144×160×4 image for SB3's CNN) → `DummyVecEnv` (or `SubprocVecEnv` when `n_envs > 1`) → `VecFrameStack(4)`. The window and sound require `n_envs = 1`. Also defines `CheckpointAndLoggingCallback`, which saves `checkpoints/<run>/best_model_{num_timesteps}.zip` (`<run>` matches the TensorBoard folder, e.g. `PPO_10`) every `check_freq` environment steps (summed across parallel envs) and logs per-episode `game/*` metrics from the vectorized `dones` and terminal `infos`.

`main.py` holds the module-level `config` dict (ROM path, dirs, `action_freq`, `reward_scale`, `game_with_sound` (test mode), and the `training`, `rewards`, `stagnation`, `episode`, and `optional_signals` sections; tests import it) and PPO hyperparameters, and dispatches on `sys.argv[1]` (`train`/`test`).

## Key conventions and gotchas

- Never manually wrap prose paragraphs with hard line breaks. Keep each paragraph on a single source line and let the editor or renderer handle visual wrapping.
- **Observations are a `Dict` space** (`screen`, `current_room_layout`, `items_in_hand`, `items_in_inventory`, `health`, `rupees`), so PPO uses `"MultiInputPolicy"`. When adding an observation key, update both `observation_space` in `ZeldaGymEnv.__init__` and the dict returned by `_get_observation()`. Non-screen values are float32 scaled to [0, 1]; keep new keys normalized the same way.
- **Reward is a fitness delta over a ledger.** Fitness is `reward_scale * (credits - debits)` from the episode's `RewardLedger`, and `step()` returns `self._fitness - self._previous_fitness`. Add reward terms as ledger `credit()`/`debit()` calls for verified events or elapsed controllable frames, add the component to `REWARD_COMPONENTS`, and keep per-step `info["reward_components"]` summing to the reward. Update the "Reward System" section of `README.md` when changing rewards.
- All episode state lives in `EpisodeState`, which `reset()` replaces and reseeds from the restored snapshot; never keep reward bookkeeping on the env itself.
- Actions are tuples of buttons pressed together (`valid_actions`/`release_actions`): 4 directions, A, B, START, and 4 direction+B combos (needed to push sea urchins with the raised shield). `run_action()` holds them for `PRESS_FRAMES` (8), releases it, and ticks one frame at a time up to `action_freq`, calling `_process_frame()` on every frame (rooms, dialogue, critical health, death, stagnation samples). Durable state (items, flags, peaks) is read once per step in `_process_durable_state()`. Timing uses emulated frames (`FRAMES_PER_SECOND`), never steps or wall-clock time.
- `ZeldaGymEnv.archive` (Go-Explore savestates keyed by room, 32-px region, equipment, and instruments) is the one piece of state that deliberately survives `reset()`; `reset()` loads an archived cell with `go_explore.start_probability` and reports `info["start"]` (`initial` or `archive`). Tests default it to 0 via `make_config()`.
- Unverified signals (enemy defeats, story flags) are listed in `UNAVAILABLE_SIGNALS`; keep them out of totals until verified against the ROM.
- `debug=True` keeps real-time emulation speed; `debug=False` calls `set_emulation_speed(0)` (unthrottled). `main.py` uses `debug=False` for `train` (parallel, headless by default per `config["training"]`, wrapped in `VecNormalize(norm_obs=False, norm_reward=True)`; the callback logs raw rewards via `unwrap_vec_normalize`) and `debug=True` for `test`. `ZeldaGymEnv.close()` stops PyBoy with `save=False` so cartridge RAM (save slots) is never written next to the ROM.
- `ZeldaGymEnv.step()` returns the Gymnasium 5-tuple, but `main.py` test mode talks to the `VecEnv`, so it uses the SB3 VecEnv API (`reset()` → obs, `step()` → 4-tuple).
- Add new RAM addresses to `memory_addresses.py` as `ADDR_*` constants with a comment explaining value encoding, rather than hardcoding hex in logic. Mark addresses checked against the ROM as "Runtime-verified"; unsupported RAM values should raise `ValueError` rather than be guessed.

## Aliases

- **`cap`** – commit and push: stage the relevant changes, commit with a concise message, and `git push` to the current branch. Never add a `Co-authored-by` trailer or any other Copilot co-author attribution to the commit.
