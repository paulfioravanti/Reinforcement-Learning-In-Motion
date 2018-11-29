import numpy as np
from windy_grid import WindyGrid
from utils import print_policy

if __name__ == "__main__":
    GRID = WindyGrid(6, 6, wind=[0, 0, 1, 2, 1, 0])
    DISCOUNT = 0.9
    EPSILON = 0.4

    ESTIMATES = {}
    C_ESTIMATES = {}
    for state in GRID.total_state_space:
        for action in GRID.possible_actions:
            ESTIMATES[(state, action)] = 0
            C_ESTIMATES[(state, action)] = 0

    TARGET_POLICY = {}
    for state in GRID.state_space:
        vals = np.array([ESTIMATES[state, a] for a in GRID.possible_actions])
        argmax = np.argmax(vals)
        TARGET_POLICY[state] = GRID.possible_actions[argmax]

    NUM_EPISODES = 10000000
    EPISODE_INTERVAL = 100000

    for i in range(NUM_EPISODES):
        if i % EPISODE_INTERVAL == 0:
            print("starting episode", i)
        behavior_policy = {}
        for state in GRID.state_space:
            rand = np.random.random()
            if rand < 1 - EPSILON:
                behavior_policy[state] = [TARGET_POLICY[state]]
            else:
                behavior_policy[state] = GRID.possible_actions
        memory = []
        observation, done = GRID.reset()
        steps = 0
        while not done:
            action = np.random.choice(behavior_policy[state])
            observation_, reward, done, info = GRID.step(action)
            steps += 1
            if steps > 25 and not done:
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
                C_ESTIMATES[(state, action)] += relative_probability
                ESTIMATES[(state, action)] += (
                    (relative_probability / C_ESTIMATES[(state, action)])
                    * (returns-ESTIMATES[(state, action)])
                )
                vals = np.array([
                    ESTIMATES[(state, a)] for a in GRID.possible_actions
                ])
                argmax = np.argmax(vals)
                TARGET_POLICY[state] = GRID.possible_actions[argmax]
                if action != TARGET_POLICY[state]:
                    break
                if len(behavior_policy[state]) == 1:
                    prob = 1 - EPSILON
                else:
                    prob = EPSILON / len(behavior_policy[state])
                relative_probability *= 1 / prob
            returns = DISCOUNT * returns + reward

    print_policy(TARGET_POLICY, GRID)
