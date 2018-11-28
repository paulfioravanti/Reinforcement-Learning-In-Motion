from grid import GridWorld
from policy_evaluation import evaluate_policy
from policy_improvement import improve_policy
from value_iteration import iterate_values
from utils import print_value_function_estimates

if __name__ == "__main__":
    grid = GridWorld(4, 4)
    THETA = 10e-6
    DISCOUNT = 1.0

    # initialize V(s)
    value_function_estimates = {state: 0 for state in grid.all_spaces}

    # Initialize policy
    policy = {}
    for state in grid.non_terminal_spaces:
        policy[state] = list(grid.action_space.keys())

    # Initial policy evaluation
    # value_function_estimates = evaluate_policy(
    #     grid, value_function_estimates, policy, DISCOUNT, THETA
    # )
    # print_value_function_estimates(value_function_estimates, grid)

    # Iterative policy evaluation
    # stable = False
    # while not stable:
    #     value_function_estimates = evaluate_policy(
    #         grid, value_function_estimates, policy, DISCOUNT, THETA
    #     )
    #     print_value_function_estimates(value_function_estimates, grid)
    #     stable, policy = improve_policy(
    #         grid, value_function_estimates, policy, DISCOUNT
    #     )

    # print_value_function_estimates(value_function_estimates, grid)
    # for state in policy:
    #     print(state, policy[state])
    # print("\n---------------\n")

    # # initialize V(s)
    # V = {}
    # for state in grid.stateSpacePlus:
    #     V[state] = 0
    # # Reinitialize policy
    # policy = {}
    # for state in grid.stateSpace:
    #     policy[state] = [key for key in grid.actionSpace.keys()]

    # 2 round of value iteration ftw
    for i in range(2):
        value_function_estimates, policy = (
            iterate_values(
                grid, value_function_estimates, policy, DISCOUNT, THETA
            )
        )

    print_value_function_estimates(value_function_estimates, grid)

    for state in policy:
        print(state, policy[state])
    print()
