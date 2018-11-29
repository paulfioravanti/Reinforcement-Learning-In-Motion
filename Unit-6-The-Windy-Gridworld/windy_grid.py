import numpy as np

# pylint: disable-msg=too-many-instance-attributes
class WindyGrid:
    __REWARD = -1

    def __init__(self, rows, columns, wind):
        self.grid = np.zeros((rows, columns))
        self.rows = rows
        self.columns = columns
        # S+ - all states in state space including terminal states
        self.total_state_space = list(range(self.rows * self.columns))
        self.state_space = self.total_state_space.copy()
        self.state_space.remove(28)
        self.action_space = {
            "U": -self.rows,
            "D": self.rows,
            "L": -1,
            "R": 1
        }
        self.possible_actions = ["U", "D", "L", "R"]
        # Top left corner of the grid
        self.agent_position = 0
        self.wind = wind

    def get_agent_row_and_column(self):
        x = self.agent_position // self.rows
        y = self.agent_position % self.columns
        return x, y

    def set_state(self, state):
        x, y = self.get_agent_row_and_column()
        self.grid[x][y] = 0
        self.agent_position = state
        x, y = self.get_agent_row_and_column()
        self.grid[x][y] = 1

    def step(self, action):
        agent_x, agent_y = self.get_agent_row_and_column()
        if self.__moving_horizontally_across_top_row(agent_x, action):
            # Horizontal actions successfully performed
            proposed_state = (
                self.agent_position + self.action_space[action]
            )
        else:
            # Agent is not on the top row of the grid
            proposed_state = (
                self.agent_position + self.action_space[action]
                + (self.wind[agent_y] * self.action_space["U"])
            )
            # if the wind is trying to push agent off grid
            if proposed_state < 0:
                proposed_state += self.rows

        if not self.__off_grid_move(proposed_state):
            self.set_state(proposed_state)

        return (
            self.agent_position,
            self.__REWARD,
            self.__is_terminal_state(self.agent_position),
            None
        )

    def reset(self):
        self.agent_position = 0
        self.grid = np.zeros((self.rows, self.columns))
        return self.agent_position, False

    def render(self):
        print("------------------------------------------")
        for row in self.grid:
            for column in row:
                if column == 0:
                    print("-", end="\t")
                elif column == 1:
                    print("X", end="\t")
            print("\n")
        print("------------------------------------------")

    @staticmethod
    def __moving_horizontally_across_top_row(agent_x, action):
        return agent_x == 0 and action in ("L", "R")

    def __off_grid_move(self, proposed_state):
        # if we move into a row not in the grid
        if (
                proposed_state not in self.total_state_space or
                (
                    self.agent_position % self.rows == 0 and
                    proposed_state % self.rows == self.rows - 1
                ) or
                self.agent_position % self.rows == self.rows - 1 and
                proposed_state % self.rows == 0
        ):
            return True
        return False

    def __is_terminal_state(self, state):
        return (
            state not in self.state_space and
            state in self.total_state_space
        )
