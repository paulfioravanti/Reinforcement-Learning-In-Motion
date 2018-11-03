import numpy as np

class Maze:
    END_OF_MAZE = (5, 5)
    EMPTY_SPACE = 0
    WALL = 1
    ROBOT = 2

    def __init__(self, action_space):
        self.maze = np.zeros((6,6)) # 6x6 maze - exit at 5,5
        self.maze[5, :5] = 1
        self.maze[:4, 5] = 1
        self.maze[2, 2:] = 1
        self.maze[3,2] = 1
        self.maze[0,0] = 2
        self.robot_position = (0,0)
        self.num_steps = 0
        self.action_space = action_space
        self.__construct_allowed_states()

    def print_maze(self):
        print("------------------------------------------")
        for row in self.maze:
            for col in row:
                if col == 0:
                    print("", end="\t")
                elif col == 1:
                    print("X", end="\t")
                elif col == 2:
                    print("R", end="\t")
            print("\n")
        print("------------------------------------------")

    def update_maze(self, action):
        y,x = self.robot_position
        self.maze[y,x] = self.EMPTY_SPACE
        y += self.action_space[action][0]
        x += self.action_space[action][1]
        self.robot_position = (y,x)
        self.maze[y,x] = self.ROBOT
        self.num_steps += 1

    def is_game_over(self):
        if self.robot_position == self.END_OF_MAZE:
            return True
        else:
            return False

    def get_state_and_reward(self):
        reward = self.__give_reward()
        return self.robot_position, reward

    def __give_reward(self):
        if self.robot_position == self.END_OF_MAZE:
            return 0
        else:
            return -1

    def __construct_allowed_states(self):
        allowed_states= {}
        for y, row in enumerate(self.maze):
            for x, col in enumerate(row):
                if self.maze[(y,x)] != self.WALL:
                    allowed_states[(y,x)] = []
                    for action in self.action_space:
                        if self.__is_allowed_move((y,x), action):
                            allowed_states[(y,x)].append(action)
        self.allowed_states = allowed_states

    def __is_allowed_move(self, state, action):
        y, x = state
        y += self.action_space[action][0]
        x += self.action_space[action][1]
        if y < 0 or x < 0 or y > 5 or x > 5:
            return False

        if self.maze[y,x] == self.EMPTY_SPACE or self.maze[y,x] == self.ROBOT:
            return True
        else:
            return False
