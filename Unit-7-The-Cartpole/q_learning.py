import gym
import numpy as np
from util import plot_running_average

# pylint: disable-msg=redefined-outer-name
def max_action(estimates, state):
    values = np.array([estimates[state, i] for i in range(2)])
    action = np.argmax(values)
    return action

def get_state(observation):
    cart_x, cart_x_dot, cart_theta, cart_theta_dot = observation
    cart_x = int(np.digitize(cart_x, CART_POS_SPACE))
    cart_x_dot = int(np.digitize(cart_x_dot, CART_VEL_SPACE))
    cart_theta = int(np.digitize(cart_theta, POLE_THETA_SPACE))
    cart_theta_dot = int(np.digitize(cart_theta_dot, POLE_THETA_VEL_SPACE))

    return (cart_x, cart_x_dot, cart_theta, cart_theta_dot)

# discretize the spaces
POLE_THETA_SPACE = np.linspace(-0.20943951, 0.20943951, 10)
POLE_THETA_VEL_SPACE = np.linspace(-4, 4, 10)
CART_POS_SPACE = np.linspace(-2.4, 2.4, 10)
CART_VEL_SPACE = np.linspace(-4, 4, 10)

if __name__ == "__main__":
    ENV = gym.make("CartPole-v0")
    # model hyperparameters
    STEP_SIZE = 0.1
    DISCOUNT = 1.0
    EPSILON = 1.0

    # construct state space
    STATES = []
    for i in range(len(CART_POS_SPACE) + 1):
        for j in range(len(CART_VEL_SPACE) + 1):
            for k in range(len(POLE_THETA_SPACE) + 1):
                for l in range(len(POLE_THETA_VEL_SPACE) + 1):
                    STATES.append((i, j, k, l))

    ESTIMATES = {}
    for state in STATES:
        for action in range(2):
            ESTIMATES[state, action] = 0

    NUM_EPISODES = 50000
    REPORT_INTERVAL = 5000
    TOTAL_REWARDS = np.zeros(NUM_EPISODES)
    for i in range(NUM_EPISODES):
        if i % REPORT_INTERVAL == 0:
            print("starting game ", i)
        done = False
        episode_rewards = 0
        observation = ENV.reset()
        while not done:
            state = get_state(observation)
            rand = np.random.random()
            if rand < (1 - EPSILON):
                action = max_action(ESTIMATES, state)
            else:
                action = ENV.action_space.sample()
            observation_, reward, done, info = ENV.step(action)
            episode_rewards += reward
            state_ = get_state(observation_)
            action_ = max_action(ESTIMATES, state_)
            ESTIMATES[state, action] = (
                ESTIMATES[state, action] + STEP_SIZE
                * (
                    reward + DISCOUNT
                    * ESTIMATES[state_, action_] - ESTIMATES[state, action]
                )
            )
            observation = observation_
        if EPSILON - 2 / NUM_EPISODES > 0:
            EPSILON -= 2 / NUM_EPISODES
        else:
            EPSILON = 0
        TOTAL_REWARDS[i] = episode_rewards

    plot_running_average(TOTAL_REWARDS)
