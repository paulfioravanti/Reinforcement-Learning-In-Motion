import gym

ENV = gym.make("FrozenLake-v0")
EPISODES = 5
STEPS = 20

for episode in range(EPISODES):
    observation = ENV.reset()
    done = False
    for step in range(STEPS):
        # Prints lake configuration to the terminal
        ENV.render()
        action = 1 # 0 = left, 1 = down, 2 = right, 3 = up
        print(observation, action)
        observation, _reward, done, _info = ENV.step(action)
        if done:
            # Reset episode
            print("Episode ", episode, "finished after ", step, "timesteps")
            break
