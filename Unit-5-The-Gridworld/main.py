from grid import GridWorld
# from policy_evaluation import evaluate_policy
# from policy_improvement import improve_policy
from value_iteration import iterate_values
from utils import print_value_function_estimates

if __name__ == "__main__":
    GRID = GridWorld(4, 4)
    THETA = 10e-6
    DISCOUNT = 1.0

    # initialize V(s)
    VALUE_FUNCTION_ESTIMATES = {state: 0 for state in GRID.all_spaces}

    # Initialize policy
    POLICY = {}
    for state in GRID.non_terminal_spaces:
        POLICY[state] = list(GRID.action_space.keys())

    # Initial policy evaluation
    # VALUE_FUNCTION_ESTIMATES = evaluate_policy(
    #     GRID, VALUE_FUNCTION_ESTIMATES, POLICY, DISCOUNT, THETA
    # )
    # print_value_function_estimates(VALUE_FUNCTION_ESTIMATES, GRID)

    # Iterative policy evaluation
    # stable = False
    # while not stable:
    #     VALUE_FUNCTION_ESTIMATES = evaluate_policy(
    #         GRID, VALUE_FUNCTION_ESTIMATES, POLICY, DISCOUNT, THETA
    #     )
    #     print_value_function_estimates(VALUE_FUNCTION_ESTIMATES, GRID)
    #     stable, POLICY = improve_policy(
    #         GRID, VALUE_FUNCTION_ESTIMATES, POLICY, DISCOUNT
    #     )

    # print_value_function_estimates(VALUE_FUNCTION_ESTIMATES, GRID)
    # for state in POLICY:
    #     print(state, POLICY[state])
    # print("\n---------------\n")

    # # initialize V(s)
    # V = {}
    # for state in GRID.stateSpacePlus:
    #     V[state] = 0
    # # Reinitialize policy
    # POLICY = {}
    # for state in GRID.stateSpace:
    #     POLICY[state] = [key for key in GRID.actionSpace.keys()]

    # 2 round of value iteration ftw
    for i in range(2):
        VALUE_FUNCTION_ESTIMATES, POLICY = (
            iterate_values(
                GRID, VALUE_FUNCTION_ESTIMATES, POLICY, DISCOUNT, THETA
            )
        )

    print_value_function_estimates(VALUE_FUNCTION_ESTIMATES, GRID)

    for state in POLICY:
        print(state, POLICY[state])
    print()
