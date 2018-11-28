def print_value_function_estimates(value_function, grid):
    for idx, row in enumerate(grid.grid):
        for idy, _ in enumerate(row):
            state = grid.rows * idx + idy
            print("%.2f" % value_function[state], end="\t")
        print("\n")
    print("--------------------")
