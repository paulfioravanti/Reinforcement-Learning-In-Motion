import numpy as np

class Agent:
    def __init__(
            self, action_space, allowed_states, step_size = 0.15,
            explore_probability = 0.2):
        self.action_space = action_space
        self.state_history = [((0,0), 0)]
        self.G = {}  # present value of expected future rewards
        self.explore_probability = explore_probability
        self.step_size = step_size
        self.__init_reward(allowed_states)

    def choose_action(self, state, allowedMoves):
        maxG = -10e15
        nextMove = None
        randomN = np.random.random()
        if randomN < self.explore_probability:
            nextMove = np.random.choice(allowedMoves)
        else:
            for action in allowedMoves:
                newState = tuple([sum(x) for x in zip(state, self.action_space[action])])
                if self.G[newState] >= maxG:
                    maxG = self.G[newState]
                    nextMove = action
        return nextMove

    def update_state_history(self, state, reward):
        self.state_history.append((state, reward))

    def learn(self):
        target = 0 # we only learn when we beat the maze

        for prev, reward in reversed(self.state_history):
            self.G[prev] = self.G[prev] + self.step_size * (target - self.G[prev])
            target += reward

        self.state_history = []
        self.explore_probability -= 10e-5

    def __init_reward(self, allowed_states):
        for state in allowed_states:
            self.G[state] = np.random.uniform(low=-1.0, high=-0.1)
