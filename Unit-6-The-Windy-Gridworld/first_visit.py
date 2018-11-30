import numpy as np
from windy_grid import WindyGrid
from utils import print_value_function

if __name__ == "__main__":
    GRID = WindyGrid(6, 6, wind=[0, 0, 1, 2, 1, 0])
    DISCOUNT = 0.9
    NUM_GAMES = 500
    STATUS_INTERVAL = 50

    POLICY = {}
    for state in GRID.state_space:
        POLICY[state] = GRID.possible_actions

    VALUE_FUNCTION_ESTIMATE = {}
    for state in GRID.total_state_space:
        VALUE_FUNCTION_ESTIMATE[state] = 0

    RETURNS = {}
    for state in GRID.state_space:
        RETURNS[state] = []

    for i in range(NUM_GAMES):
        observation, done = GRID.reset()
        memory = []
        states_returns = []
        if i % STATUS_INTERVAL == 0:
            print("starting episode", i)
        while not done:
            # attempt to follow the policy
            action = np.random.choice(POLICY[observation])
            observation_, reward, done, info = GRID.step(action)
            memory.append((observation, action, reward))
            observation = observation_
        # append terminal state
        memory.append((observation, action, reward))

        returns = 0
        last = True
        for state, action, reward in reversed(memory):
            if last:
                last = False
            else:
                states_returns.append((state, returns))
            returns = DISCOUNT * returns + reward

        states_returns.reverse()
        states_visited = []
        for state, returns in states_returns:
            if state not in states_visited:
                RETURNS[state].append(returns)
                VALUE_FUNCTION_ESTIMATE[state] = np.mean(RETURNS[state])
                states_visited.append(state)

    print_value_function(VALUE_FUNCTION_ESTIMATE, GRID)
