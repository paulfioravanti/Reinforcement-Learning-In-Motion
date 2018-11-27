from collections import defaultdict
import numpy as np

class Agent:
    def __init__(self, discount, states, policy):
        self.discount = discount
        # mapping of states to actions
        self.policy = policy
        # state reward pairs
        self.memory = []
        # states and the discounted rewards that followed.
        # aka the expected cumulative future discounted rewards.
        self.state_returns = defaultdict(list)
        # Value function is a reward for following a policy
        self.value_function = {state: 0 for state in states}

    def update_memory(self, state, reward):
        self.memory.append((state, reward))

    def update_value_function(self):
        discounted_reward = 0
        # assemble discounted future rewards from the agent's memory
        for state, reward in reversed(self.memory):
            self.state_returns[state].append(discounted_reward)
            # Calculate discounted reward recursively
            discounted_reward = reward + self.discount * discounted_reward

        # use discounted future rewards to calculate averages for each state
        for state in self.state_returns:
            self.value_function[state] = np.mean(self.state_returns[state])

        # Zero out agent's memory to make room for the new episode.
        self.memory = []

    def choose_action(self, state):
        return self.policy[state]

    def print_value_function(self):
        for state, value in self.value_function.items():
            print(state, "%.5f" % value)
