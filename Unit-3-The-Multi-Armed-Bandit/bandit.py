import numpy as np
from matplotlib import pyplot as plt

class Bandit:
    def __init__(self, num_arms, true_rewards, epsilon):
        self.num_arms = num_arms
        self.reward_estimates = np.zeros(num_arms)
        self.action_tally = np.zeros(num_arms, dtype=int)
        self.epsilon = epsilon
        self.true_rewards = true_rewards
        self.last_arm_pulled = None

    def pull(self):
        rand = np.random.random()
        if rand <= self.epsilon:
            which_arm = np.random.choice(self.num_arms)
        elif rand > self.epsilon:
            arr = np.array([approx for approx in self.reward_estimates])
            which_arm = np.random.choice(np.where(arr == arr.max())[0])
        self.last_arm_pulled = which_arm

        return np.random.randn() + self.true_rewards[which_arm]

    def update_mean(self, sample):
        which_arm = self.last_arm_pulled
        self.action_tally[which_arm] += 1
        self.reward_estimates[which_arm] = (
            self.reward_estimates[which_arm] + 1.0 /
            self.action_tally[which_arm] *
            (sample - self.reward_estimates[which_arm])
        )

def simulate(num_arms, epsilon, num_pulls):
    reward_history = np.zeros(num_pulls)
    for j in range(2000):
        rewards = [np.random.randn() for _ in range(__NUM_ACTIONS)]
        bandit = Bandit(num_arms, rewards, epsilon)
        if j % 200 == 0:
            print(j)
        for i in range(num_pulls):
            reward = bandit.pull()
            bandit.update_mean(reward)

            reward_history[i] += reward
    average = reward_history / 2000

    return average

if __name__ == "__main__":
    __NUM_ACTIONS = 5
    __NUM_PULLS = 1000
    __EPSILON1 = 0.1
    __EPSILON2 = 0.01
    __EPSILON3 = 0.0
    __RUN1 = simulate(__NUM_ACTIONS, epsilon=__EPSILON1, num_pulls=__NUM_PULLS)
    __RUN2 = simulate(__NUM_ACTIONS, epsilon=__EPSILON2, num_pulls=__NUM_PULLS)
    __RUN3 = simulate(__NUM_ACTIONS, epsilon=__EPSILON3, num_pulls=__NUM_PULLS)
    plt.plot(__RUN1, "b--", __RUN2, "r--", __RUN3, "g--")
    plt.legend([
        f"epsilon={__EPSILON1}",
        f"epsilon={__EPSILON2}",
        f"epsilon={__EPSILON3}, Pure greedy"
    ])
    plt.show()
