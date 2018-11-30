import matplotlib.pyplot as plt
import gym

# returns action
def simple_policy(state):
    return 0 if state < 0 else 1

if __name__ == "__main__":
    NUM_EPISODES = 1000
    ENV = gym.make("CartPole-v0")

    TOTAL_REWARDS = []
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
        TOTAL_REWARDS.append(episode_rewards)

    plt.plot(TOTAL_REWARDS)
    plt.show()
