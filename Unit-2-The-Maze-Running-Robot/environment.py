import numpy as np

class Maze:
    MAZE_SIZE = (6, 6)
    START_OF_MAZE = (0, 0)
    END_OF_MAZE = (5, 5)
    X_BOUNDARY = 5
    Y_BOUNDARY = 5
    EMPTY_SPACE = 0
    WALL = 1
    ROBOT = 2
    COLUMN_DISPLAYS = { 0: "", 1: "X", 2: "R" }

    def __init__(self, action_space):
        self.action_space = action_space
        # 6x6 maze - exit at 5,5
        self.maze = np.zeros(self.MAZE_SIZE)
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

    def update_maze(self, action):
        y, x = self.robot_position
        self.maze[y, x] = self.EMPTY_SPACE
        y += self.action_space[action][0]
        x += self.action_space[action][1]
        self.robot_position = (y, x)
        self.maze[y, x] = self.ROBOT
        self.num_steps += 1

    def is_game_over(self):
        if self.robot_position == self.END_OF_MAZE:
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

    def __give_reward(self):
        if self.robot_position == self.END_OF_MAZE:
            return 0
        else:
            return -1

    def __init_allowed_states(self):
        allowed_states= {}
        for y, row in enumerate(self.maze):
            for x, _column in enumerate(row):
                if self.maze[(y, x)] != self.WALL:
                    allowed_states[(y, x)] = []
                    for action in self.action_space:
                        if self.__is_valid_move((y, x), action):
                            allowed_states[(y, x)].append(action)
        self.allowed_states = allowed_states

    def __is_valid_move(self, state, action):
        y, x = state
        y += self.action_space[action][0]
        x += self.action_space[action][1]
        if self.__is_out_of_bounds(y, x):
            return False

        if self.__is_empty_space(y, x) or self.__is_initial_robot_position(y, x):
            return True
        else:
            return False

    def __is_out_of_bounds(self, y, x):
        return y < 0 or x < 0 or y > self.Y_BOUNDARY or x > self.X_BOUNDARY

    def __is_empty_space(self, y, x):
        return self.maze[y, x] == self.EMPTY_SPACE

    def __is_initial_robot_position(self, y, x):
        return self.maze[y, x] == self.ROBOT

    def __init_robot(self):
        self.maze[0, 0] = self.ROBOT
        self.robot_position = self.START_OF_MAZE
