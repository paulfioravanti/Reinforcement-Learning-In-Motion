import numpy as np
from environment import Maze
from agent import Agent
import matplotlib.pyplot as plt

# NOTE: Re-enable after refactor
# NUM_EPISODES = 5000
NUM_EPISODES = 1
EXPLORATION_CHANCE = 0.25

def run_simulation(step_size):
    maze = Maze()
    robot = (
        Agent(
            maze.allowed_states,
            step_size = step_size,
            randomFactor = EXPLORATION_CHANCE
        )
    )
    step_totals = []
    print("Beginning simulation with:")
    print(f"Robot <step_size: {robot.step_size}, randomFactor: {robot.randomFactor}>")
    print("Maze:")
    maze.printMaze()
    print("Starting episodes...")
    for i in range(NUM_EPISODES):
        if i % 1000 == 0 and i > 0:
            print(f"{i} episodes completed...")
        while not maze.isGameOver():
            state, _reward = maze.getStateAndReward()
            action = robot.chooseAction(state, maze.allowed_states[state])
            maze.updateMaze(action)
            state, reward = maze.getStateAndReward()
            robot.updateStateHistory(state, reward)
        robot.learn()
        step_totals.append(maze.steps)
        maze = Maze()
    print("Simulation complete.")
    print("----------")
    return step_totals


if __name__ == "__main__":
    step_size = 0.1
    step_totals = run_simulation(step_size = step_size)
    step_size_2 = 0.99
    step_totals_2 = run_simulation(step_size = step_size_2)

    plt.semilogy(step_totals, "b--", step_totals_2, "r--")
    plt.legend([f"step_size = {step_size}", f"step_size = {step_size_2}"])
    print("A python window has opened.")
    print("Switch over to it, and quit there to terminate this script.")
    # NOTE: Re-enable this after refactor
    # plt.show()
