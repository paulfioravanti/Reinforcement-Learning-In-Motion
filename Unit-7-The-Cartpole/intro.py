import numpy as np
import matplotlib.pyplot as plt
import gym

def simple_policy(state):
    action = 0 if state < 0 else 1
    return action

if __name__ == "__main__":
    NUM_EPISODES = 1000
    ENV = gym.make("CartPole-v0")

    total_rewards = []
    for i in range(NUM_EPISODES):
        # cart x position, cart velocity, pole angle (theta), pole velocity
        observation = ENV.reset()
        done = False
        episode_rewards = 0
        while not done:
            action = simple_policy(observation[2])
            observation_, reward, done, info = ENV.step(action)
            episode_rewards += reward
            observation = observation_
            ENV.render()
        total_rewards.append(episode_rewards)

    plt.plot(total_rewards)
    plt.show()
