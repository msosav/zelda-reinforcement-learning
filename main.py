from config.preprocess import PreprocessEnv

if __name__ == "__main__":
    config = {
        'rom_path': 'roms/ZeldaLinksAwakening.gb',
        'state_path': 'roms/ZeldaLinksAwakening.gb.state'
    }

    env = PreprocessEnv(config)
    done = True
    for step in range(100000):
        if done:
            env.reset()
        observation, reward, done, info = env.step(
            [env.action_space.sample()])
        env.render()
    env.close()
