import numpy as np
from windy_grid import WindyGrid
from utils import print_policy, print_estimates

if __name__ == "__main__":
    GRID = WindyGrid(6, 6, wind=[0, 0, 1, 2, 1, 0])
    DISCOUNT = 0.9
    EPSILON = 0.4

    ESTIMATES = {}
    RETURNS = {}
    PAIRS_VISITED = {}
    for state in GRID.total_state_space:
        for action in GRID.action_space:
            ESTIMATES[(state, action)] = 0
            RETURNS[(state, action)] = 0
            PAIRS_VISITED[(state, action)] = 0

    POLICY = {}
    for state in GRID.state_space:
        POLICY[state] = GRID.possible_actions

    NUM_EPISODES = 1000000
    EPISODE_INTERVAL = 100000

    for i in range(NUM_EPISODES):
        states_actions_returns = []
        if i % EPISODE_INTERVAL == 0:
            print("starting episode", i)
        observation, done = GRID.reset()
        memory = []
        steps = 0
        while not done:
            if len(POLICY[observation]) > 1:
                action = np.random.choice(POLICY[observation])
            else:
                action = POLICY[observation]
            observation_, reward, done, info = GRID.step(action)
            steps += 1
            if steps > 25 and not done:
                done = True
                reward = -steps
            memory.append((observation, action, reward))
            observation = observation_

        #append the terminal state
        memory.append((observation, action, reward))

        returns = 0
        last = True # start at t = T - 1
        for state, action, reward in reversed(memory):
            if last:
                last = False
            else:
                states_actions_returns.append((state, action, returns))
            returns = DISCOUNT * returns + reward
        states_actions_returns.reverse()

        states_and_actions = []
        for state, action, returns in states_actions_returns:
            if (state, action) not in states_and_actions:
                PAIRS_VISITED[(state, action)] += 1
                RETURNS[(state, action)] += (
                    (1 / PAIRS_VISITED[(state, action)])
                    * (returns - RETURNS[(state, action)])
                )
                ESTIMATES[(state, action)] = RETURNS[(state, action)]
                states_and_actions.append((state, action))
                values = np.array([
                    ESTIMATES[(state, a)] for a in GRID.possible_actions
                ])
                best = np.random.choice(np.where(values == values.max())[0])
                rand = np.random.random()
                if rand < 1 - EPSILON:
                    POLICY[state] = GRID.possible_actions[best]
                else:
                    POLICY[state] = np.random.choice(GRID.possible_actions)

    print_estimates(ESTIMATES, GRID)
    print_policy(POLICY, GRID)
