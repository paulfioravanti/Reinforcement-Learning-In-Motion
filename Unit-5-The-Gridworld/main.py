from grid import GridWorld
from policy_evaluation import evaluate_policy
# from policyImprovement import improvePolicy
# from valueIteration import iterateValues
from utils import print_value_function_estimates

if __name__ == "__main__":
    grid = GridWorld(4, 4)
    THETA = 10e-6
    GAMMA = 1.0

    # initialize V(s)
    value_function = {state: 0 for state in grid.all_spaces}

    # Initialize policy
    policy = {}
    for state in grid.non_terminal_spaces:
        policy[state] = list(grid.action_space.keys())

    # Initial policy evaluation
    value_function = evaluate_policy(grid, value_function, policy, GAMMA, THETA)
    print_value_function_estimates(value_function, grid)

    # Iterative policy evaluation
    # stable = False
    # while not stable:
    #     V = evaluatePolicy(grid, V, policy, GAMMA, THETA)
    #     printV(V, grid)
    #     stable, policy = improvePolicy(grid, V, policy, GAMMA)

    # printV(V, grid)
    # for state in policy:
    #     print(state, policy[state])
    # print('\n---------------\n')

    # # initialize V(s)
    # V = {}
    # for state in grid.stateSpacePlus:
    #     V[state] = 0
    # # Reinitialize policy
    # policy = {}
    # for state in grid.stateSpace:
    #     policy[state] = [key for key in grid.actionSpace.keys()]

    # # 2 round of value iteration ftw
    # for i in range(2):
    #     V, policy = iterateValues(grid, V, policy, GAMMA, THETA)

    # printV(V, grid)

    # for state in policy:
    #     print(state, policy[state])
    # print()

