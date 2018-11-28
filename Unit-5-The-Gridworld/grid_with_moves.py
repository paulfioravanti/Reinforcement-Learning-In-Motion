import numpy as np

class GridWorld:
    """
    Gridworld defined by m x n matrix with
    terminal states at top left corner and bottom right corner.
    State transitions are deterministic; attempting to move
    off the grid leaves the state unchanged, and rewards are -1 on
    each step.
    In this implementation we model the environment as a system of
    equations to be solved, rather than as a game to be played.
    """

    __NUM_TERMINAL_SPACES = 2
    __COLUMN_DISPLAYS = {0: " -", 1: " X"}

    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.grid = np.zeros((rows, columns))
        self.non_terminal_spaces = self.__init_non_terminal_spaces()
        self.all_spaces = list(range(self.rows * self.columns))
        self.action_space = {
            "up": -self.rows,
            "down": self.rows,
            "left": -1,
            "right": 1
        }
        self.probability_functions = self.__init_probability_functions()
        self.agent_position = np.random.choice(self.non_terminal_spaces)
        x, y = self.get_agent_row_and_column()
        self.grid[x][y] = 1

    def set_state(self, state):
        x, y = self.__get_agent_row_and_column()
        self.grid[x][y] = 0
        self.agent_position = state
        x, y = self.__get_agent_row_and_column()
        self.grid[x][y] = 1

    def __init_non_terminal_spaces(self):
        return list(map(
            lambda i: i + 1,
            range(
                self.rows
                * self.columns
                - self.__NUM_TERMINAL_SPACES
            )
        ))

    def __init_probability_functions(self):
        """
        construct state transition probabilities for
        use in value function. P(s', r|s, a) is a dictionary
        with keys corresponding to the functional arguments.
        values are either 1 or 0.
        Translations that take agent off grid leave the state unchanged.
        (s', r|s, a)
        (1, -1|1, "up") = 1
        (1, -1|2, "left") = 1
        (1, -1|3, "left") = 0
        """
        probability_functions = {}
        for state in self.non_terminal_spaces:
            for action in self.action_space:
                resulting_state = state + self.action_space[action]
                # key = (state, -1, state, action) if self.__off_grid_move(resulting_state, state) \
                #                                  else (resulting_state, -1, state, action)
                if self.__off_grid_move(resulting_state, state):
                    key = (state, -1, state, action)
                else:
                    key = (resulting_state, -1, state, action)
                probability_functions[key] = 1
        return probability_functions

    def step(self, action):
        resulting_state = self.agent_position + self.action_space[action]
        if not self.off_grid_move(resulting_state, self.agent_position):
            self.set_state(resulting_state)
            return (resulting_state, -1, self.is_terminal_state(resulting_state), None)
        else:
            return (self.agent_position, -1, self.is_terminal_state(self.agent_position), None)

    def is_terminal_state(self, state):
        return (
            state in self.all_spaces and
            state not in self.non_terminal_spaces
        )

    def render(self):
        print("-------------------------")
        for row in self.grid:
            print("|", end="")
            for column in row:
                print(self.__COLUMN_DISPLAYS[column], end=" |")
            print("\n-------------------------")

    def __off_grid_move(self, new_state, old_state):
        # if we move into a row not in the grid
        if new_state not in self.all_spaces:
            return True
        # if we're trying to wrap around to next row
        elif old_state % self.rows == 0 and new_state  % self.rows == self.rows - 1:
            return True
        elif old_state % self.rows == self.rows - 1 and new_state % self.rows == 0:
            return True
        else:
            return False

    def __get_agent_row_and_column(self):
        x = self.agent_position // self.rows
        y = self.agent_position % self.columns
        return x, y
