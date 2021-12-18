import retro

def main():
    env = retro.make(game='SonicTheHedgehog-Genesis', state='MarbleZone.Act2')
    obs = env.reset()
    env.render()
    i = 0
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
