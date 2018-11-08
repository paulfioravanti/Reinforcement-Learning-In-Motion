import matplotlib.pyplot as plt
from environment import Maze
from agent import Agent

__NUM_EPISODES = 5000
__EXPLORE_PROBABILITY = 0.25
__ACTION_SPACE = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

def run_simulation(step_size):
    maze = Maze(__ACTION_SPACE)
    robot = (
        Agent(
            __ACTION_SPACE,
            maze.allowed_states,
            step_size=step_size,
            explore_probability=__EXPLORE_PROBABILITY
        )
    )
    step_totals = []
    print("Beginning simulation with:")
    print(
        f"Robot <step_size: {robot.step_size},",
        f"explore_probability: {robot.explore_probability}>"
    )
    print("Maze:")
    maze.print_maze()
    print("Starting episodes...")
    for i in range(__NUM_EPISODES):
        if i % 1000 == 0 and i > 0:
            print(f"{i} episodes completed...")
        while not maze.is_game_over():
            run_episode(maze, robot)
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
    STEP_SIZE_1 = 0.1
    STEP_TOTALS_1 = run_simulation(step_size=STEP_SIZE_1)
    STEP_SIZE_2 = 0.99
    STEP_TOTALS_2 = run_simulation(step_size=STEP_SIZE_2)

    plt.semilogy(STEP_TOTALS_1, "b--", STEP_TOTALS_2, "r--")
    plt.legend([f"step_size = {STEP_SIZE_1}", f"step_size = {STEP_SIZE_2}"])
    print("A python matplotlib window has opened.")
    print("Switch over to it, and quit there to terminate this script.")
    plt.show()
