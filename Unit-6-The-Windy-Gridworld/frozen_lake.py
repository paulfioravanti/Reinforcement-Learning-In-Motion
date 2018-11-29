import numpy as np
import gym
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ENV = gym.make("FrozenLake-v0")
    DISCOUNT = 0.9
    EPSILON = 0.1
    POSSIBLE_ACTIONS = [0, 1, 2, 3]
    STATE_SPACE = list(range(16))
    ESTIMATES = {}
    RETURNS = {}
    PAIRS_VISITED = {}

    for state in STATE_SPACE:
        for action in POSSIBLE_ACTIONS:
            ESTIMATES[(state, action)] = 0
            RETURNS[(state, action)] = 0
            PAIRS_VISITED[(state, action)] = 0

    POLICY = {}
    for state in STATE_SPACE:
        POLICY[state] = np.random.choice(POSSIBLE_ACTIONS)

    NUM_EPISODES = 100000
    STATUS_INTERVAL = 5000

    for i in range(NUM_EPISODES):
        states_actions_returns = []
        if i % STATUS_INTERVAL == 0:
            print("starting episode", i)
        observation = ENV.reset()
        memory = []
        done = False
        while not done:
            action = POLICY[observation]
            observation_, reward, done, info = ENV.step(action)
            memory.append((observation, action, reward))
            observation = observation_

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
                values = np.array(
                    [ESTIMATES[(state, a)] for a in POSSIBLE_ACTIONS]
                )
                best = np.random.choice(np.where(values == values.max())[0])
                rand = np.random.random()
                if rand < 1 - EPSILON:
                    POLICY[state] = POSSIBLE_ACTIONS[best]
                else:
                    POLICY[state] = np.random.choice(POSSIBLE_ACTIONS)
    NUM_GAMES = 1000
    REWARDS = np.zeros(NUM_GAMES)
    EPISODE_REWARDS = 0
    for i in range(NUM_GAMES):
        observation = ENV.reset()
        done = False
        while not done:
            action = POLICY[observation]
            observation_, reward, done, info = ENV.step(action)
            observation = observation_
            EPISODE_REWARDS += reward
        REWARDS[i] = EPISODE_REWARDS
    print(EPISODE_REWARDS / NUM_GAMES)
    plt.plot(REWARDS)
    plt.show()
