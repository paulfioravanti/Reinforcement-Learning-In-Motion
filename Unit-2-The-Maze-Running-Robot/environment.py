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

    def __init__(self, action_space):
        self.action_space = action_space
        # 6x6 maze - exit at 5,5
        self.maze = np.zeros((self.__Y_BOUNDARY, self.__X_BOUNDARY))
        self.num_steps = 0
        self.__init_walls()
        self.__init_allowed_states()
        self.maze[0, 0] = self.__ROBOT
        self.robot_position = self.__START

    def print_maze(self):
        print("-------------------------")
        for row in self.maze:
            print("|", end="")
            for column in row:
                print(self.__COLUMN_DISPLAYS[column], end=" |")
            print("\n-------------------------")

    def update_maze(self, action):
        current_position = self.robot_position
        new_position = self.__next_state(current_position, action)
        self.maze[current_position] = self.__EMPTY_SPACE
        self.robot_position = new_position
        self.maze[new_position] = self.__ROBOT
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
                    for action in self.action_space:
                        if self.__is_valid_move((y, x), action):
                            allowed_states[(y, x)].append(action)
        self.allowed_states = allowed_states

    def __is_valid_move(self, state, action):
        proposed_state = self.__next_state(state, action)

        if (self.__is_out_of_bounds(proposed_state) or
                self.__is_wall(proposed_state)):
            return False
        return True

    def __next_state(self, state, action):
        return tuple(
            position + transition for position, transition in
            zip(state, self.action_space[action])
        )

    def __is_out_of_bounds(self, state):
        y, x = state
        return (
            y not in range(0, self.__Y_BOUNDARY) or
            x not in range(0, self.__X_BOUNDARY)
        )

    def __is_wall(self, state):
        return self.maze[state] == self.__WALL
