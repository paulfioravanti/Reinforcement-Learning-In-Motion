from enum import Enum
import numpy as np
from matplotlib import pyplot as plt

class Mode(Enum):
    SAMPLE_AVERAGE = "Sample Average"
    CONSTANT = "Constant Alpha"

class Bandit:
    __INCREMENT = 1

    # pylint: disable-msg=too-many-arguments
    def __init__(self, num_arms, initial_estimate, true_rewards, epsilon, mode):
        self.num_arms = num_arms
        self.reward_estimates = np.full(num_arms, initial_estimate)
        self.pulled_arms_tally = np.zeros(num_arms, dtype=int)
        self.epsilon = epsilon
        self.true_rewards = true_rewards
        self.last_arm_pulled = None
        self.mode = mode
    # pylint: enable-msg=too-many-arguments

    def pull(self):
        self.__pull_arm()
        # Rewards are non-stationary.
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

    def __update_reward_estimates(self, reward):
        if self.mode is Mode.SAMPLE_AVERAGE:
            self.__sample_average_update(reward)
        else:
            # Weights most recent rewards more heavily than long past rewards.
            self.__constant_update(reward)

    # Q(A) <- Q(A) + 1/N(A)[R-Q(A)]
    def __sample_average_update(self, reward):
        self.reward_estimates[self.last_arm_pulled] = (
            self.__old_estimate()
            + self.__sample_average_alpha()
            * self.__error(reward)
        )

    # Q(A)
    def __old_estimate(self):
        return self.reward_estimates[self.last_arm_pulled]

    # 1/N(A)
    def __sample_average_alpha(self):
        # alpha decreases over time as more actions performed to help converge
        # on the expected reward more efficiently.
        return 1.0 / self.pulled_arms_tally[self.last_arm_pulled]

    # R-Q(A) Error in the estimate
    def __error(self, convergence_target):
        return convergence_target - self.__old_estimate()

    # Q(A) <- Q(A) + 0.1*[R-Q(A)]
    def __constant_update(self, reward):
        self.reward_estimates[self.last_arm_pulled] = (
            self.__old_estimate()
            + self.__constant_alpha()
            * self.__error(reward)
        )

    @staticmethod
    def __constant_alpha():
        return 0.1

# pylint: disable-msg=too-many-arguments
def simulate(num_simulations,
             num_arms,
             initial_estimate,
             epsilon,
             num_pulls,
             mode):
    reward_history = np.zeros(num_pulls)
    print(f"Beginning {num_simulations} simulations with:")
    print(f"Bandit <num_arms: {num_arms}, epsilon: {epsilon}, ", end="")
    print(f"num_pulls {num_pulls}, mode: {mode.value}>")
    print(f"Number of simulations completed:")
    for simulation_num in range(num_simulations):
        rewards = np.random.randn(num_arms)
        bandit = Bandit(num_arms, initial_estimate, rewards, epsilon, mode)
        if __should_report_simulation_number(simulation_num):
            print(simulation_num, end="...", flush=True)
        for arm_pull in range(num_pulls):
            reward = bandit.pull()
            bandit.update_mean(reward)
            reward_history[arm_pull] += reward
    print(f"{num_simulations}.")
    # Average
    return reward_history / __NUM_SIMULATIONS
# pylint: enable-msg=too-many-arguments

def __should_report_simulation_number(simulation):
    return simulation % __SIMULATION_STATUS_INTERVAL == 0

if __name__ == "__main__":
    __NUM_SIMULATIONS = 2000
    __NUM_ARMS = 5
    __NUM_PULLS = 1000
    __SIMULATION_STATUS_INTERVAL = 200
    __PESSIMISTIC_INITIAL_ESTIMATE = 0.0
    __OPTIMISTIC_INITIAL_ESTIMATE = 10.0
    __EPSILON1 = 0.1
    __EPSILON2 = 0.0

    # Realistic Epsilon Greedy
    __RUN1 = simulate(
        __NUM_SIMULATIONS,
        __NUM_ARMS,
        initial_estimate=__PESSIMISTIC_INITIAL_ESTIMATE,
        epsilon=__EPSILON1,
        num_pulls=__NUM_PULLS,
        mode=Mode.CONSTANT
    )
    # Optimistic Pure Greedy
    __RUN2 = simulate(
        __NUM_SIMULATIONS,
        __NUM_ARMS,
        initial_estimate=__OPTIMISTIC_INITIAL_ESTIMATE,
        epsilon=__EPSILON2,
        num_pulls=__NUM_PULLS,
        mode=Mode.CONSTANT
    )

    plt.plot(__RUN1, "b--", __RUN2, "r--")
    plt.legend(["Realistic Epsilon Greedy", "Optimistic Pure Greedy"])
    print("Simulations complete.")
    print("----------")
    print("A python matplotlib window has opened.")
    print("Switch over to it, and quit there to terminate this script.")
    plt.show()
