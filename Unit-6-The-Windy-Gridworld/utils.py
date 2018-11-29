import numpy as np

def print_value_function(value_function, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _idx in enumerate(row):
            state = grid.rows * idx + idy
            print("%.2f" % value_function[state], end="\t")
        print("\n")
    print("--------------------")

def print_policy(policy, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _ in enumerate(row):
            state = grid.rows * idx + idy
            if state in grid.state_space:
                string = "".join(policy[state])
                print(string, end="\t")
            else:
                print("", end="\t")
        print("\n")
    print("--------------------")

def print_estimates(estimates, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _column in enumerate(row):
            state = grid.rows * idx + idy
            if state != grid.rows * grid.columns - 1:
                vals = list(map(
                    lambda action, state=state: np.round(
                        estimates[state, action], 5
                    ),
                    grid.possible_actions
                ))
                print(vals, end="\t")
        print("\n")
    print("--------------------")

def sample_reduced_action_space(grid, action):
    actions = grid.possible_actions[:]
    actions.remove(action)
    sample = np.random.choice(actions)
    return sample
