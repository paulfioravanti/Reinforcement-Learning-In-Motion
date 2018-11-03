import numpy as np
from environment import Maze
from agent import Agent
import matplotlib.pyplot as plt

if __name__ == "__main__":
    maze = Maze()
    robot = Agent(maze, step_size = 0.1, explore_upper_bound = 0.25)
    step_totals = []
    for i in range(5000):
        if i % 1000 == 0:
            print(i)
        while not maze.is_game_over():
            state, _reward = maze.get_state_and_reward()
            action = robot.choose_action(state, maze.allowed_states[state])
            maze.update_maze(action)
            state, reward = maze.get_state_and_reward()
            robot.update_reward_estimates(state, reward)
        robot.learn()
        step_totals.append(maze.num_steps)
        maze = Maze()

    maze = Maze()
    robot = Agent(maze, step_size = 0.99, explore_upper_boundk = 0.25)
    step_totals_2 = []
    for i in range(5000):
        if i % 1000 == 0:
            print(i)
        while not maze.is_game_over():
            state, _reward = maze.get_state_and_reward()
            action = robot.choose_action(state, maze.allowed_states[state])
            maze.update_maze(action)
            state, reward = maze.get_state_and_reward()
            robot.update_reward_estimates(state, reward)
        robot.learn()
        step_totals.append(maze.num_steps)
        maze = Maze()
    plt.semilogy(step_totals, "b--", step_totals_2, "r--")
    plt.legend(["step_size = 0.1","step_size = 0.99"])
    plt.show()
