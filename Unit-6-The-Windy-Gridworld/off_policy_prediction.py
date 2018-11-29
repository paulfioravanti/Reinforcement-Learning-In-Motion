import numpy as np
from windy_grid import WindyGrid
from utils import print_estimates

if __name__ == "__main__":
    GRID = WindyGrid(6, 6, wind=[0, 0, 1, 2, 1, 0])
    DISCOUNT = 0.9

    ESTIMATES = {}
    C_ESTIMATES = {}
    for state in GRID.total_state_space:
        for action in GRID.possible_actions:
            ESTIMATES[(state, action)] = 0
            C_ESTIMATES[(state, action)] = 0

    TARGET_POLICY = {}
    for state in GRID.state_space:
        TARGET_POLICY[state] = np.random.choice(GRID.possible_actions)

    NUM_EPISODES = 1000
    EPISODE_INTERVAL = 100

    for i in range(NUM_EPISODES):
        if i % EPISODE_INTERVAL == 0:
            print(i)
        behavior_policy = {}
        for state in GRID.state_space:
            behavior_policy[state] = GRID.possible_actions
        memory = []
        observation, done = GRID.reset()
        steps = 0
        while not done:
            action = np.random.choice(behavior_policy[observation])
            observation_, reward, done, info = GRID.step(action)
            steps += 1
            if steps > 25:
                done = True
                reward = -steps
            memory.append((observation, action, reward))
            observation = observation_
        memory.append((observation, action, reward))

        returns = 0
        relative_probability = 1
        last = True
        for (state, action, reward) in reversed(memory):
            if last:
                last = False
            else:
                C_ESTIMATES[state, action] += relative_probability
                ESTIMATES[state, action] += (
                    (relative_probability / C_ESTIMATES[state, action])
                    * (returns - ESTIMATES[state, action])
                )
                prob = 1 if action in TARGET_POLICY[state] else 0
                relative_probability *= prob / (1 / len(behavior_policy[state]))
                if relative_probability == 0:
                    break
            returns = DISCOUNT * returns + reward
    print_estimates(ESTIMATES, GRID)
