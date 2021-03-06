from collections import defaultdict
import matplotlib.pyplot as plt
import gym
from agent import Agent

class RandomPolicy(defaultdict):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def __missing__(self, key):
        return self.env.action_space.sample()

def run_simulation(robot):
    episode_rewards = []
    rewards = 0
    for _episode in range(EPISODES):
        done = False
        observation = ENV.reset()
        while not done:
            action = robot.choose_action(observation)
            observation, reward, done, _info = ENV.step(action)
            robot.update_memory(observation, reward)
            rewards += reward
        robot.update_value_function()
        episode_rewards.append(rewards)
    robot.print_value_function()
    plt.plot(episode_rewards)
    plt.show()

if __name__ == "__main__":
    ENV = gym.make("FrozenLake-v0")
    NUM_STATES = 16
    STATES = list(range(NUM_STATES))
    EPISODES = 1000
    DISCOUNT = 0.9
    RANDOM_POLICY = RandomPolicy(ENV)
    # an attempt at a reasonable policy
    # 0 = left, 1 = down, 2 = right, 3 = up
    DIRECTED_POLICY = {
        0: 1,
        1: 2,
        2: 1,
        3: 0,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        8: 2,
        9: 1,
        10: 1,
        11: 1,
        12: 2,
        13: 2,
        14: 2
    }
    ROBOT1 = Agent(discount=DISCOUNT, states=STATES, policy=RANDOM_POLICY)
    ROBOT2 = Agent(discount=DISCOUNT, states=STATES, policy=DIRECTED_POLICY)

    run_simulation(ROBOT1)
    print("\n------------------------\n")
    run_simulation(ROBOT2)
