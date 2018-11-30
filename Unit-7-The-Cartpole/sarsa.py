import numpy as np
import gym
from util import plot_running_average
from gym import wrappers

def max_action(estimates, state):
    values = np.array([estimates[state , i] for i in range(2)])
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
    states = []
    for i in range(len(CART_POS_SPACE) + 1):
        for j in range(len(CART_VEL_SPACE) + 1):
            for k in range(len(POLE_THETA_SPACE) + 1):
                for l in range(len(POLE_THETA_VEL_SPACE) + 1):
                    states.append((i, j, k, l))

    ESTIMATES = {}
    for state in states:
        for action in range(2):
            ESTIMATES[state, action] = 0

    NUM_EPISODES = 50000
    REPORT_INTERVAL = 5000
    TOTAL_REWARDS = np.zeros(NUM_EPISODES)

    for i in range(NUM_EPISODES):
        if i % REPORT_INTERVAL == 0:
            print("starting game", i)
        # cart x position, cart velocity, pole angle (theta), pole velocity
        observation = ENV.reset()
        state = get_state(observation)
        rand = np.random.random()
        action = max_action(ESTIMATES, state) if rand < (1 - EPSILON) else ENV.action_space.sample()
        done = False
        episode_rewards = 0
        while not done:
            observation_, reward, done, info = ENV.step(action)
            episode_rewards += reward
            state_ = get_state(observation_)
            rand = np.random.random()
            action_ = max_action(ESTIMATES, state_) if rand < (1 - EPSILON) else ENV.action_space.sample()
            ESTIMATES[state,action] = (
                ESTIMATES[state,action] + STEP_SIZE
                * (reward + DISCOUNT * ESTIMATES[state_, action_] - ESTIMATES[state, action])
            )
            state, action = state_, action_
        if EPSILON - 2 / NUM_EPISODES > 0:
            EPSILON -= 2 / NUM_EPISODES
        else:
            EPSILON = 0
        TOTAL_REWARDS[i] = episode_rewards
    plot_running_average(TOTAL_REWARDS)
