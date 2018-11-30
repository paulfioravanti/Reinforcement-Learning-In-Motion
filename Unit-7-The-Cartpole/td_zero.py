import numpy as np
import gym

def simple_policy(state):
    action = 0 if state < 5 else 1
    return action

if __name__ == "__main__":
    ENV = gym.make("CartPole-v0")
    STEP_SIZE = 0.1
    DISCOUNT = 1.0
    SEQUENCE_START = -0.20943951
    SEQUENCE_END = 0.20943951
    NUM_SAMPLES = 10
    # discretize the space
    STATES = np.linspace(SEQUENCE_START, SEQUENCE_END, NUM_SAMPLES)

    ESTIMATES = {}
    for state in range(len(STATES) + 1):
        ESTIMATES[state] = 0

    for i in range(1000):
        # cart x position, cart velocity, pole angle (theta), pole velocity
        observation = ENV.reset()
        done = False
        while not done:
            state = int(np.digitize(observation[2], STATES))
            action = simple_policy(state)
            observation_, reward, done, info = ENV.step(action)
            state_ = int(np.digitize(observation_[2], STATES))
            ESTIMATES[state] = (
                ESTIMATES[state]
                + STEP_SIZE
                * (reward + DISCOUNT * ESTIMATES[state_] - ESTIMATES[state])
            )
            observation = observation_

    for state in ESTIMATES:
        print(state, "%.3f" % ESTIMATES[state])
