import numpy as np

def iterate_values(grid, value_function_estimates, policy, discount, theta):
    converged = False
    while not converged:
        delta = 0
        for state in grid.non_terminal_spaces:
            old_value_function_estimates = value_function_estimates[state]
            new_value_function_estimates = []
            for action in grid.action_space:
                for key in grid.probability_functions:
                    (new_state, reward, old_state, act) = key
                    if state == old_state and action == act:
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
            converged = True if delta < theta else False

    for state in grid.non_terminal_spaces:
        new_values = []
        actions = []
        for action in grid.action_space:
            for key in grid.probability_functions:
                (new_state, reward, old_state, act) = key
                if state == old_state and action == act:
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
