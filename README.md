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

### Testing

To test the agent, run the following command:

```bash
uv run main.py test ./checkpoints/<checkpoint>.zip
```

Example:

```bash
uv run main.py test ./checkpoints/best_model_5000.zip
```

## Reward System (WIP)

The reward system is based on the following rules:

- `+1` for each item in the inventory
