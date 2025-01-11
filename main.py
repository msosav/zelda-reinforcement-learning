import sys

from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common import env_checker

from utils import CheckpointAndLoggingCallback, PreprocessEnv

if __name__ == "__main__":
    config = {
        "rom_path": "roms/ZeldaLinksAwakening.gb",
        "checkpoint_dir": "checkpoints/",
        "log_dir": "logs/",
        "action_freq": 24,
        "exploration_reward": 0.25,
        "reward_scale": 1,
        "game_with_sound": True,
    }

    callback = CheckpointAndLoggingCallback(
        check_freq=5000, save_path=config["checkpoint_dir"]
    )

    env = PreprocessEnv(config)

    mode = sys.argv[1]

    if mode == "train":
        model = PPO("MultiInputPolicy", env, verbose=1, n_steps=2048,
                    batch_size=512, n_epochs=1, gamma=0.997, ent_coef=0.01, tensorboard_log=config["log_dir"])

        model.learn(total_timesteps=1000000, callback=callback)
    elif mode == "test":
        checkpoint = sys.argv[2]
        try:
            model = PPO.load(checkpoint)
        except:
            print("Checkpoint not found")
            sys.exit(1)

        observation = env.reset()

        while True:
            action, _state = model.predict(observation)
            observation, reward, done, info = env.step(action)
