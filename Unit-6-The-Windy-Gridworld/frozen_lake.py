import numpy as np
import gym
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    GAMMA = 0.9
    EPS = 0.1
    possibleActions = [0, 1, 2, 3]
    stateSpace = [i for i in range(16)]
    Q = {}
    returns = {}
    pairsVisited = {}
    for state in stateSpace:
        for action in possibleActions:
            Q[(state, action)] = 0
            returns[(state,action)] = 0
            pairsVisited[(state,action)] = 0

    policy = {}
    for state in stateSpace:
        policy[state] = np.random.choice(possibleActions)

    for i in range(100000):
        statesActionsReturns = []
        if i % 5000 == 0:
            print('starting episode', i)
        observation = env.reset()
        memory = []
        done = False
        while not done:
            action = policy[observation]
            observation_, reward, done, info = env.step(action)
            memory.append((observation, action, reward))
            observation = observation_

        memory.append((observation, action, reward))
        G = 0
        last = True # start at t = T - 1
        for state, action, reward in reversed(memory):
            if last:
                last = False
            else:
                statesActionsReturns.append((state,action,G))
            G = GAMMA*G + reward

        statesActionsReturns.reverse()
        statesAndActions = []
        for state, action, G in statesActionsReturns:
            if (state, action) not in statesAndActions:
                pairsVisited[(state,action)] += 1
                returns[(state,action)] += (1 / pairsVisited[(state,action)])*(G-returns[(state,action)])
                Q[(state,action)] = returns[(state,action)]
                statesAndActions.append((state,action))
                values = np.array([Q[(state,a)] for a in possibleActions])
                best = np.random.choice(np.where(values==values.max())[0])
                rand = np.random.random()
                if rand < 1 - EPS:
                    policy[state] = possibleActions[best]
                else:
                    policy[state] = np.random.choice(possibleActions)
    numGames = 1000
    rewards = np.zeros(numGames)
    epRewards = 0
    for i in range(numGames):
        observation = env.reset()
        done = False
        while not done:
            action = policy[observation]
            observation_, reward, done, info = env.step(action)
            observation = observation_
            epRewards += reward
        rewards[i] = epRewards
    print (epRewards/numGames)
    plt.plot(rewards)
    plt.show()
