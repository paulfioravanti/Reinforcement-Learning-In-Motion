import numpy as np

# pylint: disable-msg=too-many-locals
def improve_policy(grid, value_function_estimate, policy, discount):
    stable = True
    new_policy = {}
    for state in grid.non_terminal_spaces:
        old_actions = policy[state]
        value = []
        new_action = []
        for action in policy[state]:
            grid.set_state(state)
            weight = 1 / len(policy[state])
            new_state, reward, _, _ = grid.step(action)
            key = (new_state, reward, state, action)
            value.append(
                np.round(
                    weight
                    * grid.probability_functions[key]
                    * (reward + discount * value_function_estimate[new_state]),
                    2
                )
            )
            new_action.append(action)
        value = np.array(value)
        best = np.where(value == value.max())[0]
        best_actions = [new_action[item] for item in best]
        new_policy[state] = best_actions

        if old_actions != best_actions:
            stable = False

    return stable, new_policy
