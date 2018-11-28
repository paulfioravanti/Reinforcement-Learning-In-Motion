import numpy as np

def evaluate_policy(grid, value_function, policy, discount, theta):
    # policy evaluation for the random choice in gridworld
    converged = False
    while not converged:
        delta = 0
        for state in grid.non_terminal_spaces:
            old_state_value_function = value_function[state]
            total = 0
            weight = 1 / len(policy[state])
            for action in policy[state]:
                for key in grid.probability_functions:
                    (new_state, reward, old_state, action_performed) = key
                    # We're given the state and action, want new state and reward
                    if old_state == state and action == action_performed:
                        total += weight * grid.probability_functions[key] * (reward + discount * value_function[new_state])
            value_function[state] = total
            delta = max(delta, np.abs(old_state_value_function - value_function[state]))
            converged = True if delta < theta else False
    return value_function
