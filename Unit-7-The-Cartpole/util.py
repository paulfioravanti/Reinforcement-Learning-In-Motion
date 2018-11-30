import numpy as np
import matplotlib.pyplot as plt

def plot_running_average(total_rewards):
    num_rewards = len(total_rewards)
    running_avg = np.empty(num_rewards)
    for i in range(num_rewards):
        running_avg[i] = np.mean(total_rewards[max(0, i - 100):(i + 1)])
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

def plot_running_average_comparison(algo1, algo2, labels=None):
    n_1 = len(algo1)
    n_2 = len(algo2)
    running_avg_algo1 = np.empty(n_1)
    running_avg_algo2 = np.empty(n_2)
    for i in range(n_1):
        running_avg_algo1[i] = np.mean(algo1[max(0, i - 100):(i + 1)])
        running_avg_algo2[i] = np.mean(algo2[max(0, i - 100):(i + 1)])

    plt.plot(running_avg_algo1, 'r--')
    plt.plot(running_avg_algo2, 'b--')
    plt.title("Running Average")
    if labels:
        plt.legend((labels[0], labels[1]))
    plt.show()
