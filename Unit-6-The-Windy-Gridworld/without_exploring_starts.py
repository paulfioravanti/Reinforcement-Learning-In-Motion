from windygrid import WindyGrid
from utils import printPolicy, printQ, sampleReducedActionSpace
import numpy as np

if __name__ == "__main__":
    grid = WindyGrid(6,6, wind=[0, 0, 1, 2, 1, 0])
    GAMMA = 0.9
    EPS = 0.4

    Q = {}
    returns = {}
    pairsVisited = {}
    for state in grid.stateSpacePlus:
        for action in grid.actionSpace.keys():
            Q[(state, action)] = 0
            returns[(state,action)] = 0
            pairsVisited[(state,action)] = 0

    policy = {}
    for state in grid.stateSpace:
        policy[state] = grid.possibleActions

    for i in range(1000000):
        statesActionsReturns = []
        if i % 100000 == 0:
            print("starting episode", i)
        observation, done = grid.reset()
        memory = []
        steps = 0
        while not done:
            if len(policy[observation]) > 1:
                action = np.random.choice(policy[observation])
            else:
                action = policy[observation]
            observation_, reward, done, info = grid.step(action)
            steps += 1
            if steps > 25 and not done:
                done = True
                reward = -steps
            memory.append((observation, action, reward))
            observation = observation_

        #append the terminal state
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
                values = np.array([Q[(state,a)] for a in grid.possibleActions])
                best = np.random.choice(np.where(values==values.max())[0])
                rand = np.random.random()
                if rand < 1 - EPS:
                    policy[state] = grid.possibleActions[best]
                else:
                    policy[state] = np.random.choice(grid.possibleActions)

    printQ(Q, grid)
    printPolicy(policy,grid)
