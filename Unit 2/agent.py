import numpy as np

class Agent:
    # Represents actions mapped to robot translations on a board
    ACTION_SPACE = { "U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1) }

    # 80% exploit, 20% explore
    def __init__(self, states, step_size=0.15, explore_upper_bound=0.2):
        # list of states and reward pairs
        self.episode_state_rewards = [((0, 0), 0)]
        # Constant value for Controling rate/speed of learning. aka alpha
        # Usually between 0 and 1
        # - if this is 0, then only the current state is considered and no
        # estimates get updated (therefore agent is not really learning
        # anything).
        # - If this is 1, the estimation of the expected future reward
        # becomes the running total itself ie the agent considers every step it
        # has taken
        self.step_size = step_size
        # Value to determine the chance of picking a different action at random
        # versus continuing to exploit the reward generated from a different
        # action. aka epsilon in epsilon-greedy
        self.explore_upper_bound = explore_upper_bound
        self.__init_reward_estimates(states)
        self.__construct_allowed_states()

    def choose_action(self, state, valid_actions):
        if should_explore():
            return self.__explore(valid_actions)
        else:
            return self.__exploit(state, valid_actions)

    def learn(self):
        # we only learn when we beat the maze and get the reward of 0
        target = 0

        for previous_state, reward in reversed(self.episode_state_rewards):
            __update_reward_estimate_for_state(previous_state, target)
            target += reward

        # zero-out Agent's memory to make room for the next episode
        self.episode_state_rewards = []
        # marginally increase chance of exploitation
        self.explore_upper_bound -= 10e-5

    def update_reward_estimates(self, state, reward):
        self.reward_estimates.append((state, reward))


    def __init_reward_estimates(self, states):
        # keys are states, and values are estimates of future rewards
        # starting from that state. a.k.a: G
        self.reward_estimates = {}
        for state in states:
            self.reward_estimates[state] = __random_estimate()

    def __random_estimate(self):
        # high is rounded to 0.1 so that no square looks better than the exit
        return np.random.uniform(low = -1.0, high = -0.1)

    def __should_explore(self):
        return np.random.random() < self.explore_upper_bound

    def __explore(self, valid_actions):
        return np.random.choice(valid_actions)

    def __exploit(self, state, valid_actions):
        max_estimate = -10e15
        next_action = None

        for action in valid_actions:
            new_state = (
                tuple([sum(x) for x in zip(state, self.ACTION_SPACE[action])])
            )
            if self.reward_estimates[new_state] >= max_estimate:
                next_action = action
                max_estimate = self.reward_estimates[new_state]
        return next_action

    def __update_reward_estimate_for_state(self, state, target):
        reward_estimate = self.reward_estimates[state]
        self.reward_estimates[state] = (
            reward_estimate + self.step_size * (target - reward_estimate)
        )
