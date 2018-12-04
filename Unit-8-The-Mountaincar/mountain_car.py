import numpy as np
import gym
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 0 - Push Left
    # 1 - No Push
    # 2 - Push Right
    ENV = gym.make("MountainCar-v0")
    NUM_EPISODES = 1000
    REWARDS = np.zeros(NUM_EPISODES)

    for episode in range(NUM_EPISODES):
        observation = ENV.reset()
        done = False
        episode_rewards = 0
        while not done:
            action = ENV.action_space.sample()
            observation, reward, done, info = ENV.step(action)
            episode_rewards += reward
            # ENV.render()
        REWARDS[episode] = episode_rewards

    plt.plot(REWARDS)
    plt.show()
