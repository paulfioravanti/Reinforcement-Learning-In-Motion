import os
import time
import matplotlib.pyplot as plt
from environment import Maze
from agent import Agent

__NUM_EPISODES = 5000
__EPSILON = 0.25
__ACTION_SPACE = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

def run_simulation(alpha, watch=False):
    maze = Maze(__ACTION_SPACE)
    robot = (
        Agent(
            __ACTION_SPACE,
            maze.allowed_states,
            alpha=alpha,
            epsilon=__EPSILON
        )
    )
    step_totals = []
    if watch:
        print("Starting simulation...")
        time.sleep(0.1)
    else:
        print("Beginning simulation with:")
        print(f"Robot <alpha: {robot.alpha}, epsilon: {robot.epsilon}>")
        print("Maze:")
        maze.print_maze()
        print("Starting episodes...")
    for i in range(__NUM_EPISODES):
        if i % 1000 == 0 and i > 0:
            print(f"{i} episodes completed...")
        while not maze.is_game_over():
            run_episode(maze, robot)
            if watch:
                print(f"Robot <alpha: {robot.alpha}, epsilon: {robot.epsilon}>")
                print(f"Episode {i}")
                maze.print_maze()
                print(f"Number of steps: {maze.num_steps}")
                time.sleep(0.020)
                os.system("clear")
        robot.learn()
        step_totals.append(maze.num_steps)
        # reset maze for next episode
        maze = Maze(__ACTION_SPACE)

    print("Simulation complete.")
    print("----------")
    return step_totals

def run_episode(maze, robot):
    state, _reward = maze.get_state_and_reward()
    action = robot.choose_action(state, maze.allowed_states[state])
    maze.update_maze(action)
    state, reward = maze.get_state_and_reward()
    robot.update_state_rewards(state, reward)

if __name__ == "__main__":
    ALPHA_1 = 0.1
    STEP_TOTALS_1 = run_simulation(alpha=ALPHA_1, watch=False)
    ALPHA_2 = 0.99
    STEP_TOTALS_2 = run_simulation(alpha=ALPHA_2, watch=False)

    plt.semilogy(STEP_TOTALS_1, "b--", STEP_TOTALS_2, "r--")
    plt.legend([f"alpha = {ALPHA_1}", f"alpha = {ALPHA_2}"])
    print("A python matplotlib window has opened.")
    print("Switch over to it, and quit there to terminate this script.")
    plt.show()
