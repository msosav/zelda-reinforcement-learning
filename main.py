import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from utils import CheckpointAndLoggingCallback, PreprocessEnv

# Reward magnitudes are in unscaled units; penalties are positive debits. Every value is
# multiplied by reward_scale. These are starting values to tune, not proven optima.
config = {
    "rom_path": "roms/ZeldaLinksAwakening.gb",
    "checkpoint_dir": "checkpoints/",
    "log_dir": "logs/",
    "action_freq": 24,
    "reward_scale": 1,
    # Sound while watching a checkpoint with `test`.
    "game_with_sound": True,
    # Defaults suit an Apple M5 Pro: parallel headless emulators, CNN on the GPU. To watch
    # training, set n_envs to 1 and enable show_window (and sound if wanted).
    "training": {
        "n_envs": 8,
        "show_window": False,  # requires n_envs = 1
        "sound": False,  # requires show_window
        "device": "mps",  # Apple GPU; ~2.5x faster than "cpu" for the CNN policy
        "rollout_steps": 2048,  # PPO steps per update, split across environments
        "total_timesteps": 10000000,
        "checkpoint_freq": 50000,  # environment steps between checkpoints
    },
    "rewards": {
        "unique_equipment": 20.0,
        "equipment_upgrade": 20.0,
        "house_exit": 10.0,
        "new_room": 2.0,
        "new_tile": 0.02,  # each new 8-px tile in a room, once per episode
        "instrument": 100.0,
        "story_milestone": 20.0,
        "health_peak_per_heart": 0.25,
        "rupee_peak_per_rupee": 0.01,
        "consumable_peak_per_unit": 0.02,
        "enemy_defeat": 0.1,
        "critical_health_per_second": 0.02,
        "death": 20.0,
        "small_reward_budget": 2.0,
        "recurring_penalty_budget": 5.0,
        "tile_budget": 10.0,
    },
    "stagnation": {
        "window_seconds": 15.0,
        "base_rate": 0.01,
        "room_loop_base_rate": 0.02,  # bouncing between known rooms costs more
        "rate_step": 0.01,
        "max_rate": 0.04,
        "jitter_pixels": 4,
        "bucket_pixels": 8,
        "stationary_fraction": 0.9,
        "min_repeats": 3,
        "max_route_buckets": 16,
        "new_bucket_tolerance": 1,
        "max_room_loop_period": 4,
    },
    "episode": {
        "no_progress_seconds": 180.0,
        "max_seconds": 1800.0,
    },
    # Unverified signals stay disabled; enabling them fails explicitly.
    "optional_signals": {
        "enemy_defeat": False,
        "story_milestones": [],
    },
}

if __name__ == "__main__":
    usage = "Usage: uv run main.py train | uv run main.py test <checkpoint.zip>"
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    if mode not in ("train", "test") or (mode == "test" and len(sys.argv) < 3):
        sys.exit(usage)

    if mode == "train":
        training = config["training"]
        n_envs = training["n_envs"]
        if training["rollout_steps"] % n_envs:
            raise ValueError("training['rollout_steps'] must be divisible by n_envs")

        # Train as fast as the emulator allows (press Space in the window for 1x).
        env = PreprocessEnv(
            {**config, "game_with_sound": training["sound"]},
            debug=False,
            n_envs=n_envs,
            show_window=training["show_window"],
        )
        # Scale rewards for the value function; observations are already in [0, 1].
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.997)
        callback = CheckpointAndLoggingCallback(
            check_freq=training["checkpoint_freq"], save_path=config["checkpoint_dir"]
        )
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=training["rollout_steps"] // n_envs,
            batch_size=512,
            n_epochs=4,
            gamma=0.997,
            learning_rate=1e-4,
            ent_coef=0.02,  # keep exploring beyond the first few rooms
            target_kl=0.03,  # stop an update early once the policy has moved enough
            device=training["device"],
            tensorboard_log=config["log_dir"],
        )

        try:
            model.learn(total_timesteps=training["total_timesteps"], callback=callback)
        finally:
            env.close()
    elif mode == "test":
        # Watch a checkpoint at real-time speed in a single window.
        env = PreprocessEnv(config, debug=True)
        checkpoint = sys.argv[2]
        try:
            model = PPO.load(checkpoint, env=env)
        except FileNotFoundError:
            print("Checkpoint not found")
            sys.exit(1)
        except ValueError as error:
            print(f"Checkpoint is incompatible with the current environment: {error}")
            sys.exit(1)

        observation = env.reset()

        while True:
            action, _state = model.predict(observation)
            observation, reward, done, info = env.step(action)
