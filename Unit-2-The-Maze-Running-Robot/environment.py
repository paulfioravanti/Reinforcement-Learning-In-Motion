import numpy as np

class Maze:
    __START = (0, 0)
    __FINISH = (5, 5)
    __X_BOUNDARY = 6
    __Y_BOUNDARY = 6
    __EMPTY_SPACE = 0
    __WALL = 1
    __ROBOT = 2
    __COLUMN_DISPLAYS = {0: "  ", 1: " X", 2: " R"}

    def __init__(self, env_actions):
        self.env_actions = env_actions
        # 6x6 maze - exit at 5,5
        self.maze = np.zeros((self.__Y_BOUNDARY, self.__X_BOUNDARY))
        self.num_steps = 0
        self.__init_walls()
        self.__init_allowed_states()
        self.maze[0, 0] = self.__ROBOT
        self.robot_position = self.__START

    def print_maze(self):
        print("----------------------")
        for row in self.maze:
            for column in row:
                print(self.__COLUMN_DISPLAYS[column], end="  ")
            print("\n")
        print("----------------------")

    def update_maze(self, action):
        y, x = self.robot_position
        self.maze[y, x] = self.__EMPTY_SPACE
        y += self.env_actions[action][0]
        x += self.env_actions[action][1]
        self.robot_position = (y, x)
        self.maze[y, x] = self.__ROBOT
        self.num_steps += 1

    def is_game_over(self):
        return self.robot_position == self.__FINISH

    def get_state_and_reward(self):
        return (self.robot_position, self.__give_reward())

    def __init_walls(self):
        # `1`s are walls
        self.maze[5, :5] = self.__WALL
        self.maze[:4, 5] = self.__WALL
        self.maze[2, 2:] = self.__WALL
        self.maze[3, 2] = self.__WALL

    def __give_reward(self):
        if self.robot_position == self.__FINISH:
            return 0
        return -1

    def __init_allowed_states(self):
        allowed_states = {}
        for y, row in enumerate(self.maze):
            for x, _column in enumerate(row):
                if self.maze[(y, x)] != self.__WALL:
                    allowed_states[(y, x)] = []
                    for action in self.env_actions:
                        if self.__is_valid_move((y, x), action):
                            allowed_states[(y, x)].append(action)
        self.allowed_states = allowed_states

    def __is_valid_move(self, state, action):
        y, x = state
        y += self.env_actions[action][0]
        x += self.env_actions[action][1]
        if self.__is_out_of_bounds(y, x):
            return False

        if self.__is_empty_space(y, x) or self.__is_initial_robot_position(y, x):
            return True
        return False

    def __is_out_of_bounds(self, y, x):
        return (
            y not in range(0, self.__Y_BOUNDARY) or
            x not in range(0, self.__X_BOUNDARY)
        )


    def __is_empty_space(self, y, x):
        return self.maze[y, x] == self.__EMPTY_SPACE

    def __is_initial_robot_position(self, y, x):
        return self.maze[y, x] == self.__ROBOT
