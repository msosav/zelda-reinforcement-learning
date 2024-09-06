# The Legend of Zelda: Reinforcement Learning

This repository contains the code for the project "The Legend of Zelda: Reinforcement Learning", it's purpose is to train an agent to play the game "The Legend of Zelda: Link's Awakening" for GameBoy using Reinforcement Learning.

## Dependencies

- `Python 3.11`
- `pyboy`
- `numpy`
- `pytorch`
- `gymnasium`
- `stable-baselines3`
- `opencv-python`

## Installation

1. Clone the repository

```bash
git clone https://github.com/msosav/zelda-reinforcement-learning
cd zelda-reinforcement-learning
```

1. Create a virtual environment and activate it

```bash
python -m venv venv
source venv/bin/activate
```

1. Install the requirements

```bash
pip install -r requirements.txt
```

## Usage

The program is divided into two main parts: the training and the testing.

### Training

To train the agent, run the following command:

```bash
python main.py train
```

### Testing

To test the agent, run the following command:

```bash
python main.py test
```

And then change the main.py file to use the trained model.

```python
model = PPO.load("./checkpoints/{model_name}")
```

## Reward System (WIP)

The reward system is based on the following rules:

- `+1` for each item in the inventory

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

