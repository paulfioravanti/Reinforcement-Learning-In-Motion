import numpy as np
from matplotlib import pyplot as plt

class Bandit:
    __INCREMENT = 1

    def __init__(self, num_arms, true_rewards, epsilon):
        self.num_arms = num_arms
        self.reward_estimates = np.zeros(num_arms)
        self.pulled_arms_tally = np.zeros(num_arms, dtype=int)
        self.epsilon = epsilon
        self.true_rewards = true_rewards
        self.last_arm_pulled = None

    def pull(self):
        self.__pull_arm()
        return self.__reward()

    def update_mean(self, reward):
        self.__increment_arm_tally()
        self.__update_reward_estimates(reward)

    def __pull_arm(self):
        if self.__should_exploit():
            self.last_arm_pulled = self.__random_best_arm()
        else:
            self.last_arm_pulled = self.__random_arm()

    def __should_exploit(self):
        return np.random.random() > self.epsilon

    def __random_best_arm(self):
        estimates = self.reward_estimates
        # np.where returns an array of row idxs and an array of col idxs but
        # since columns are of length 1, the latter is an empty array.
        # Regardless, the first value in the array needs to be specifically
        # retrieved.
        # REF: https://stackoverflow.com/q/34667282/567863
        best_arms = np.where(estimates == estimates.max())[0]
        return np.random.choice(best_arms)

    def __random_arm(self):
        return np.random.choice(self.num_arms)

    def __reward(self):
        # Normally distributed random reward centered around the true reward
        # for the last pulled arm
        return np.random.randn() + self.true_rewards[self.last_arm_pulled]

    def __increment_arm_tally(self):
        self.pulled_arms_tally[self.last_arm_pulled] += self.__INCREMENT

    # Q(A) <- Q(A) + 1/N(A)[R-Q(A)]
    def __update_reward_estimates(self, reward):
        self.reward_estimates[self.last_arm_pulled] = (
            self.__old_estimate() + self.__alpha() * self.__error(reward)
        )

    # Q(A)
    def __old_estimate(self):
        return self.reward_estimates[self.last_arm_pulled]

    # 1/N(A)
    def __alpha(self):
        # alpha decreases over time as more actions performed to help converge
        # on the expected reward more efficiently.
        return 1.0 / self.pulled_arms_tally[self.last_arm_pulled]

    # R-Q(A) Error in the estimate
    def __error(self, convergence_target):
        return convergence_target - self.__old_estimate()

def simulate(num_arms, epsilon, num_pulls):
    reward_history = np.zeros(num_pulls)
    print(f"Beginning {__NUM_SIMULATIONS} simulations with:")
    print(f"Bandit <num_arms: {num_arms}, epsilon: {epsilon}, ", end="")
    print(f"num_pulls {num_pulls}>")
    print(f"Number of simulations completed:")
    for simulation_num in range(__NUM_SIMULATIONS):
        rewards = np.random.randn(num_arms)
        bandit = Bandit(num_arms, rewards, epsilon)
        if __should_report_simulation_number(simulation_num):
            print(simulation_num, end="...", flush=True)
        for arm_pull in range(num_pulls):
            reward = bandit.pull()
            bandit.update_mean(reward)
            reward_history[arm_pull] += reward
    print(f"{__NUM_SIMULATIONS}.")
    # Average
    return reward_history / __NUM_SIMULATIONS

def __should_report_simulation_number(simulation):
    return simulation % __SIMULATION_STATUS_INTERVAL == 0


if __name__ == "__main__":
    __NUM_SIMULATIONS = 2000
    __NUM_ARMS = 5
    __NUM_PULLS = 1000
    __SIMULATION_STATUS_INTERVAL = 200
    __EPSILON1 = 0.1
    __EPSILON2 = 0.01
    __EPSILON3 = 0.0
    __RUN1 = simulate(__NUM_ARMS, epsilon=__EPSILON1, num_pulls=__NUM_PULLS)
    __RUN2 = simulate(__NUM_ARMS, epsilon=__EPSILON2, num_pulls=__NUM_PULLS)
    __RUN3 = simulate(__NUM_ARMS, epsilon=__EPSILON3, num_pulls=__NUM_PULLS)
    plt.plot(__RUN1, "b--", __RUN2, "r--", __RUN3, "g--")
    plt.legend([
        f"epsilon={__EPSILON1}",
        f"epsilon={__EPSILON2}",
        f"epsilon={__EPSILON3}, Pure greedy"
    ])
    print("Simulations complete.")
    print("----------")
    print("A python matplotlib window has opened.")
    print("Switch over to it, and quit there to terminate this script.")
    plt.show()
