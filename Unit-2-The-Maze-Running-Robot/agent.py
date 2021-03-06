import numpy as np

class Agent:
    __EXPLORATION_STEP = 10e-5
    __HIGH_NEGATIVE_ESTIMATE = -10e15

    # 80% exploit, 20% explore
    def __init__(self, action_space, allowed_states, alpha=0.15, epsilon=0.2):
        # Represents actions mapped to robot translations on a board
        self.action_space = action_space
        # list of states and reward pairs
        self.state_rewards = [((0, 0), 0)]
        # Constant value for controlling rate/speed of learning.
        # The learning rate or step size determines to what extent newly
        # acquired information overrides old information.
        # Usually between 0 and 1
        # - if this is 0, then only the current state is considered and no
        # estimates get updated (therefore agent is not really learning
        # anything).
        # - If this is 1, the estimation of the expected future reward
        # becomes the running total itself ie the agent considers every step it
        # has taken
        self.alpha = alpha
        # Value to determine the chance of picking a different action at random
        # versus continuing to exploit the reward generated from a different
        # action. aka the epsilon in epsilon-greedy
        self.epsilon = epsilon
        self.__init_reward_estimates(allowed_states)

    def choose_action(self, state, allowed_moves):
        if self.__should_explore():
            return self.__explore(allowed_moves)
        return self.__exploit(state, allowed_moves)

    def update_state_rewards(self, state, reward):
        self.state_rewards.append((state, reward))

    def learn(self):
        # we only learn when we beat the maze and get the reward of 0
        target = 0

        for previous_state, reward in reversed(self.state_rewards):
            self.__update_reward_estimate_for_state(previous_state, target)
            target += reward

        # zero-out Agent's memory to make room for the next episode
        self.state_rewards = []
        # marginally decrease chance of exploration in the next episode
        self.epsilon -= self.__EXPLORATION_STEP

    def __init_reward_estimates(self, states):
        # keys are states, and values are estimates of future rewards
        # starting from that state. a.k.a: G
        self.reward_estimates = {}
        for state in states:
            self.reward_estimates[state] = self.__random_estimate()

    def __should_explore(self):
        return np.random.random() < self.epsilon

    def __exploit(self, state, valid_actions):
        max_estimate = self.__HIGH_NEGATIVE_ESTIMATE
        next_action = None

        for action in valid_actions:
            potential_new_state = self.__transition_state(state, action)
            if self.reward_estimates[potential_new_state] >= max_estimate:
                next_action = action
                max_estimate = self.reward_estimates[potential_new_state]
        return next_action

    def __update_reward_estimate_for_state(self, state, target):
        old_estimate = self.reward_estimates[state]
        estimate_error = target - old_estimate
        self.reward_estimates[state] = (
            old_estimate + self.alpha * estimate_error
        )

    def __transition_state(self, state, action):
        return tuple(
            sum(translation)
            for translation in zip(state, self.action_space[action])
        )

    @staticmethod
    def __random_estimate():
        # high is rounded to 0.1 so that no square looks better than the exit
        return np.random.uniform(low=-1.0, high=-0.1)

    @staticmethod
    def __explore(valid_actions):
        return np.random.choice(valid_actions)
