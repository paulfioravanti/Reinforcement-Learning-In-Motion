import numpy as np

class Agent:
    def __init__(
            self, action_space, allowed_states, step_size = 0.15,
            explore_probability = 0.2):
        self.action_space = action_space
        self.state_history = [((0,0), 0)]
        self.reward_estimates = {}  # present value of expected future rewards
        self.explore_probability = explore_probability
        self.step_size = step_size
        self.__init_reward(allowed_states)

    def choose_action(self, state, allowed_moves):
        max_estimate = -10e15
        next_move = None
        random_n = np.random.random()
        if random_n < self.explore_probability:
            next_move = np.random.choice(allowed_moves)
        else:
            for action in allowed_moves:
                new_state = tuple([sum(x) for x in zip(state, self.action_space[action])])
                if self.reward_estimates[new_state] >= max_estimate:
                    max_estimate = self.reward_estimates[new_state]
                    next_move = action
        return next_move

    def update_state_history(self, state, reward):
        self.state_history.append((state, reward))

    def learn(self):
        target = 0 # we only learn when we beat the maze

        for prev, reward in reversed(self.state_history):
            self.reward_estimates[prev] = self.reward_estimates[prev] + self.step_size * (target - self.reward_estimates[prev])
            target += reward

        self.state_history = []
        self.explore_probability -= 10e-5

    def __init_reward(self, allowed_states):
        for state in allowed_states:
            self.reward_estimates[state] = np.random.uniform(low=-1.0, high=-0.1)
