import numpy as np

# pylint: disable-msg=too-many-locals
def iterate_values(grid, value_function_estimates, policy, discount, theta):
    converged = False
    while not converged:
        delta = 0
        for state in grid.non_terminal_spaces:
            old_value_function_estimates = value_function_estimates[state]
            new_value_function_estimates = []
            for action in grid.action_space:
                grid.set_state(state)
                new_state, reward, _done, _info = grid.step(action)
                key = (new_state, reward, state, action)
                new_value_function_estimates.append(
                    grid.probability_functions[key]
                    * (
                        reward
                        + discount
                        * value_function_estimates[new_state]
                    )
                )
            new_value_function_estimates = (
                np.array(new_value_function_estimates)
            )
            best_value_function_estimates = (
                np.where(new_value_function_estimates ==
                         new_value_function_estimates.max())[0]
            )
            best_state = np.random.choice(best_value_function_estimates)
            value_function_estimates[state] = (
                new_value_function_estimates[best_state]
            )
            delta = max(
                delta,
                np.abs(
                    old_value_function_estimates
                    - value_function_estimates[state]
                )
            )
            converged = delta < theta

    for state in grid.non_terminal_spaces:
        new_values = []
        actions = []
        for action in grid.action_space:
            grid.set_state(state)
            new_state, reward, _done, _info = grid.step(action)
            key = (new_state, reward, state, action)
            new_values.append(
                grid.probability_functions[key]
                * (
                    reward
                    + discount
                    * value_function_estimates[new_state]
                )
            )
            actions.append(action)
        new_values = np.array(new_values)
        best_action_idx = np.where(new_values == new_values.max())[0]
        best_actions = actions[best_action_idx[0]]
        policy[state] = best_actions

    return value_function_estimates, policy
