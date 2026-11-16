<div align="center">
  <img src="./res/banner.png" alt="Banner" width="100%">
</div>

# The Legend of Zelda: Reinforcement Learning

This repository contains the code for the project "The Legend of Zelda: Reinforcement Learning", it's purpose is to train an agent to play the game "The Legend of Zelda: Link's Awakening" for GameBoy using Reinforcement Learning.

## Installation

Clone the repository

```bash
git clone https://github.com/msosav/zelda-reinforcement-learning
cd zelda-reinforcement-learning
```

Create a virtual environment

> [!NOTE]
> You need to have [uv](https://docs.astral.sh/uv/) installed in your machine. It will fetch Python 3.11 if needed.

```bash
uv venv --python 3.11
```

Install the requirements

```bash
uv pip install -r requirements.txt
```

## Usage

The program is divided into two main parts: the training and the testing.

> [!NOTE]
> You need to create a folder called `roms` in the root of the project and put the rom of the game "The Legend of Zelda: Link's Awakening" for GameBoy in it and name it `ZeldaLinksAwakening.gb`.

Startup waits for each menu to be ready, creates a new game in the first empty save slot with the name `A`, and stops at Marin's first dialogue after Link wakes up. The agent advances the dialogue and gets Link out of bed. Each episode resets to this same in-memory emulator state. Existing saves are not overwritten; startup fails explicitly if all three slots are occupied or it cannot reach the wake-up dialogue within 7,200 frames (about two minutes of game time).

### Training

To train the agent, run the following command:

```bash
uv run main.py train
```

Training settings live in `config["training"]` in `main.py`. By default it runs 4 emulators in parallel without a window or sound and trains on the Apple GPU (`"device": "mps"`; use `"cpu"` elsewhere). On an Apple M5 Pro that is about 440 steps per second, so the default 10 million steps take roughly 6.5 hours. Training wraps the environments in `VecNormalize` to scale rewards for the value function; TensorBoard still logs raw episode returns. To watch training instead, set `"n_envs": 1` and `"show_window": True`, plus `"sound": True` if you want audio; the window and sound require a single emulator. Training runs unthrottled, so a visible window plays much faster than real time (press Space in the window to toggle 1x speed). `rollout_steps` is split across the emulators and must be divisible by `n_envs`.

Checkpoints (about 48 MB each) are saved to `checkpoints/` every `checkpoint_freq` (50,000) environment steps; monitor progress with `uv run tensorboard --logdir logs/`. Testing always runs a single emulator at real-time speed with the window, using `game_with_sound` for audio.

### Testing

To test the agent, run the following command:

```bash
uv run main.py test ./checkpoints/<checkpoint>.zip
```

Example:

```bash
uv run main.py test ./checkpoints/best_model_5000.zip
```

## Reward System

Rewards are computed from emulator RAM only (no computer vision). Each step returns the change in total fitness, where fitness is `reward_scale * (credited gains - debited penalties)` accumulated in episode-local ledgers. Weights, budgets, stagnation thresholds, and episode limits are configured in `main.py` and validated at startup; invalid values fail explicitly. The values below are starting points for tuning, not proven optima, and are all multiplied by `reward_scale`.

| Component | Value | Conditions |
| --- | --- | --- |
| Unique equipment | +20 | Once per item ID (equipped and inventory slots are deduplicated) or flippers; potions and initial equipment earn nothing |
| Equipment upgrade | +20 per level | Sword, shield, or bracelet level above 1 (level 1 comes with the base item) |
| Leaving Marin's house | +10 | Once, on the first controllable frame in the overworld |
| First visit to a room | +2 | Once per room identity (overworld, or indoor map plus room); overworld visits also come from the visited bit |
| New tile | +0.02 | Once per 8-pixel tile within each room per episode, up to +10; gives a steady exploration signal (it does not count as durable progress) |
| Dungeon instrument | +100 | Once per instrument |
| New health peak | +0.25 per heart | Only above the episode's previous peak, so healing back to full earns nothing |
| New rupee peak | +0.01 per rupee | Only above the previous peak, so spending and recollecting earns nothing |
| New consumable peak | +0.02 per unit | Separate peaks for arrows, bombs, and magic powder |
| Critical health | -0.02 per second | Health at one heart or less, during controllable time |
| Stagnation | -0.01 to -0.04 per second | See below; categories do not stack |
| Death | -20 | Once, when health reaches zero (ends the episode) |

Health, rupee, and consumable credits share a total budget of +2 per episode. The tile reward is an addition beyond the original reward spec: the first training runs showed that rewards for leaving the house and new rooms alone were too sparse for the agent to learn anything. Critical-health and stagnation debits each stop at 5 per episode; death is outside those budgets. Penalties are never refunded when their condition ends. Every reset restores the wake-up snapshot and reseeds all baselines from it, so initial possessions, health, and rooms earn nothing.

Time is measured in emulated frames (about 59.73 per second), not agent steps, so penalty rates do not depend on `action_freq`. Each action holds the button for 8 frames, releases it, and runs until `action_freq` frames (9 to 600) have passed, sampling RAM on every frame so intermediate rooms, deaths, and control changes are not missed. Forced time (scripted motion, map loading, room scrolling, and the inventory sliding open or closed) is excluded from controllable clocks; dialogue and the open inventory count as controllable.

### Stagnation

There is no blanket penalty for staying in one room. Over a rolling 15-second window of controllable time, the environment detects:

- **Stationary**: at least 90% of positions within 4 pixels of the median (walking into walls, idling, or sitting in a menu).
- **Pacing**: repeating a short known route (2 to 16 eight-pixel buckets, at most one new bucket) at least three times.
- **Room loop**: alternating at least three times through the same sequence of previously visited rooms.
- **Menu loop**: opening the inventory at least three times in the window with the same repeating equipment changes.

After a complete window the rate starts at 0.01 per second (0.02 for room loops, the dominant pattern in early training runs) and rises by 0.01 for each further 15 seconds of the same pattern, up to 0.04. Leaving a room flushes the local position history. Reaching a dialogue page not seen this episode, or any durable progress, clears local evidence; reopening the same dialogue does not.

### Episode limits

Episodes are truncated after 3 minutes of controllable time without durable progress (a new room, new equipment or upgrade, leaving the house, or an instrument) or after 30 minutes of emulated time. Resource peaks, healing, movement, menus, and revisited rooms do not reset the progress clock. Death takes precedence over a simultaneous timeout. `info` reports the termination or truncation reason, per-component step rewards (which sum to the step reward), cumulative credits and debits, remaining budgets, stagnation state, and milestone times.

### Signal verification and disabled components

RAM addresses come from the Data Crystal RAM map and were checked against the local ROM with controlled PyBoy traces (see `config/memory_addresses.py`). The following stay disabled, excluded from totals, and are listed at startup and in TensorBoard under `game/availability/*`, because no reliable RAM signal has been verified:

- Enemy defeats, and recognizing useful combat to suppress stagnation.
- Story milestones other than leaving Marin's house.
- Puzzle-state progress.

Enabling an unverified signal in `config["optional_signals"]` raises an error.

### Observations and checkpoints

The inventory observation now covers all ten slots and `rupees` is the decoded count. All non-screen observations are floats scaled to [0, 1]: room layout bytes and item IDs are divided by their maxima, `health` is the fraction of 14 hearts (so half hearts are representable), and `rupees` is the fraction of 999. `screen` drops PyBoy's constant alpha channel. The grayscale screen keeps a channel axis, so the four stacked frames form a 144×160×4 image that the policy processes with a CNN. Because the observation space changed, checkpoints trained before this reward system cannot be loaded (`main.py test` reports the mismatch); train a fresh model. Policies trained under earlier rewards are not comparable baselines for this system.

### Metrics and tests

The training callback logs episode return and length, house-exit and sword rates and times, instruments, unique rooms, critical-health and stagnation durations, budget saturation, death and timeout rates, and per-component totals under `game/*` in TensorBoard.

```bash
uv run python -m unittest discover -s tests
ZELDA_REAL_ROM_TESTS=1 uv run python -m unittest tests/test_real_rom.py  # scripted trace on the real ROM
```

Improved gameplay has not been measured yet: comparing the previous rewards, repaired item/exploration rewards, and this shaping across several seeds of one-million-step training remains to be done.
