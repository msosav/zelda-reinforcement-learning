import sys

from stable_baselines3 import PPO

from utils import CheckpointAndLoggingCallback, PreprocessEnv

if __name__ == "__main__":
    config = {
        "rom_path": "roms/ZeldaLinksAwakening.gb",
        "checkpoint_dir": "checkpoints/",
        "log_dir": "logs/",
    }

    callback = CheckpointAndLoggingCallback(
        check_freq=1000, save_path=config["checkpoint_dir"]
    )

    env = PreprocessEnv(config)

    mode = sys.argv[1]

    if mode == "train":
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=config["log_dir"],
            learning_rate=0.000001,
            n_steps=512,
        )

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
