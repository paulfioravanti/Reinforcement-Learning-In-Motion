import numpy as np

# pylint: disable-msg=too-few-public-methods
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
                if self.__off_grid_move(resulting_state, state):
                    key = (state, -1, state, action)
                else:
                    key = (resulting_state, -1, state, action)
                probability_functions[key] = 1
        return probability_functions

    def __off_grid_move(self, new_state, old_state):
        # if we move into a row not in the grid
        if (
                new_state not in self.all_spaces or
                (
                    old_state % self.rows == 0 and
                    new_state % self.rows == self.rows - 1
                ) or
                old_state % self.rows == self.rows - 1 and
                new_state % self.rows == 0
        ):
            return True
        return False
