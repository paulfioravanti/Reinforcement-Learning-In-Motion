import numpy as np

def print_value_function(value_function, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _idx in enumerate(row):
            state = grid.rows * idx + idy
            print("%.2f" % value_function[state], end="\t")
        print("\n")
    print("--------------------")

def printPolicy(policy, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _ in enumerate(row):
            state = grid.m * idx + idy
            if state in grid.stateSpace:
                string = ''.join(policy[state])
                print(string, end='\t')
            else:
                print('', end='\t')
        print('\n')
    print('--------------------')

def printQ(Q, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _ in enumerate(row):
            state = grid.m * idx + idy
            if state != grid.m * grid.n - 1:
                vals = [np.round(Q[state,action], 5) for action in grid.possibleActions]
                print(vals, end='\t')
        print('\n')
    print('--------------------')

def sampleReducedActionSpace(grid, action):
    actions = grid.possibleActions[:]
    actions.remove(action)
    sample = np.random.choice(actions)
    return sample
