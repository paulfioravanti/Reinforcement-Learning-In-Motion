import numpy as np

class Maze:
    # Represents actions mapped to robot translations on a board
    ACTION_SPACE = { "U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1) }
    COLUMN_DISPLAYS = { 0: "", 1: "X", 2: "R" }
    MIN_BOARD_VALUE = 5
    MAX_BOARD_VALUE = 5
    EMPTY_SPACE = 0
    WALL = 1
    ROBOT = 2
    END_OF_MAZE = (5, 5)

    def __init__(self):
        # 6x6 maze - exit at 5,5
        # `0`s are empty spaces
        self.maze = np.zeros((6,6))
        self.num_steps = 0
        self.__init_walls()
        self.__init_allowed_states()
        self.__init_robot()

    def print_maze(self):
        print("------------------------------------------")
        for row in self.maze:
            for column in row:
                print(self.COLUMN_DISPLAYS[column], end = "\t")
            print("\n")
        print("------------------------------------------")

    def is_valid_move(self, state, action):
        y, x = state
        y += self.ACTION_SPACE[action][0]
        x += self.ACTION_SPACE[action][1]
        if self.__is_out_of_bounds(y, x):
            return False

        if self.__empty_space(x, y) or self.__initial_robot_position(x, y):
            return True
        else:
            return False

    def update_maze(self, action):
        y, x = self.robot_position
        self.maze[y,x] = self.EMPTY_SPACE
        y += self.ACTION_SPACE[action][0]
        x += self.ACTION_SPACE[action][1]
        self.robot_position = (y, x)
        self.maze[y, x] = self.ROBOT
        self.num_steps += 1

    def is_game_over(self):
        if self.robot_position == END_OF_MAZE:
            return True
        else:
            return False

    def get_state_and_reward(self):
        return (self.robot_position, self.__give_reward())

    def __init_walls(self):
        # `1`s are walls
        self.maze[5, :5] = self.WALL
        self.maze[:4, 5] = self.WALL
        self.maze[2, 2:] = self.WALL
        self.maze[3, 2] = self.WALL

    def __init_allowed_states(self):
        allowed_states= {}
        for y, row in enumerate(self.maze):
            for x, col in enumerate(row):
                if self.maze[(y, x)] != self.WALL:
                    allowed_states[(y, x)] = []
                    for action in self.ACTION_SPACE:
                        if self.is_valid_move((y, x), action):
                            allowed_states[(y, x)].append(action)
        self.allowed_states = allowed_states

    def __init_robot(self):
        self.maze[0, 0] = self.ROBOT
        self.robot_position = (0, 0)

    def __is_out_of_bounds(self, y, x):
        return not (
            self.MIN_BOARD_VALUE < y < self.MAX_BOARD_VALUE and
            self.MIN_BOARD_VALUE < x < self.MAX_BOARD_VALUE
        )

    def __is_empty_space(self, y, x):
        return self.maze[y, x] == self.EMPTY_SPACE

    def __is_initial_robot_position(self, y, x):
        return self.maze[y, x] == self.ROBOT

    def __give_reward(self):
        if self.robot_position == self.END_OF_MAZE:
            return 0
        else:
            return -1

