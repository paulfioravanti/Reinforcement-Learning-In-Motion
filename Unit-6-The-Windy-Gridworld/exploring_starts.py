import numpy as np
from windy_grid import WindyGrid
from utils import print_policy, print_estimates

if __name__ == "__main__":
    GRID = WindyGrid(6, 6, wind=[0, 0, 1, 2, 1, 0])
    DISCOUNT = 0.9

    ESTIMATES = {}
    RETURNS = {}
    PAIRS_VISITED = {}
    for state in GRID.total_state_space:
        for action in GRID.possible_actions:
            ESTIMATES[(state, action)] = 0
            RETURNS[(state, action)] = 0
            PAIRS_VISITED[(state, action)] = 0


    POLICY = {}
    for state in GRID.state_space:
        POLICY[state] = np.random.choice(GRID.possible_actions)

    NUM_EPISODES = 1000000
    EPISODE_INTERVAL = 50000

    for i in range(NUM_EPISODES):
        if i % EPISODE_INTERVAL == 0:
            print("starting episode", i)
        states_actions_returns = []
        observation = np.random.choice(GRID.state_space)
        action = np.random.choice(GRID.possible_actions)
        GRID.set_state(observation)
        observation_, reward, done, info = GRID.step(action)
        memory = [(observation, action, reward)]
        steps = 1
        while not done:
            action = POLICY[observation_]
            steps += 1
            observation, reward, done, info = GRID.step(action)
            if steps > 15 and not done:
                done = True
                reward = -steps
            memory.append((observation_, action, reward))
            observation_ = observation

        # append the terminal state
        memory.append((observation_, action, reward))

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
                values = np.array([ESTIMATES[(state, a)] for a in GRID.possible_actions])
                best = np.argmax(values)
                POLICY[state] = GRID.possible_actions[best]

    print_estimates(ESTIMATES, GRID)
    print_policy(POLICY, GRID)
