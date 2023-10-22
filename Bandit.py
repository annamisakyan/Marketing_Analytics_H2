"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from logs import *

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)

import numpy as np
import pandas as pd
import math
from statsmodels.stats.power import TTestIndPower
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import csv

BANDIT_REWARD=[1,2,3,4]
NUMBER_OF_TRIALS= 20000
EPS = 0.1

class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        self.p = p
        self.N = 0
        self.mean = 0

    @abstractmethod
    def __repr__(self):
        return f"{self.__class__.__name__}({self.p})"

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        with open('rewards.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.__class__.__name__, self.mean])

        # print the average reward
        logger.info(f"Average Reward for {self.__class__.__name__}: {self.mean:.4f}")

#--------------------------------------#


class Visualization():
    def plot1(self, rewards_list):
        # Visualize the performance of each bandit: linear and log
        plt.figure(figsize=(12, 6))

        # Linear plot
        for i, rewards in enumerate(rewards_list):
            plt.plot(rewards, label=f'Bandit {i} (Linear)')
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Rewards')
        plt.legend()

        plt.figure(figsize=(12, 6))

        # Log plot with handling for zero or negative values
        for i, rewards in enumerate(rewards_list):
            # Add a small constant to rewards to avoid log(0) or log(negative)
            logged_rewards = [np.log(reward + 1e-10) for reward in rewards]
            plt.plot(rewards, label=f'Bandit {i}')

        plt.yscale('log')  # Set the y-axis to logarithmic scale
        plt.xlabel('Time Steps')
        plt.ylabel('Logarithmic Cumulative Rewards')
        plt.legend()
        plt.title('Performance of Bandits (Y-axis: Logarithmic)')
        plt.show()

    def plot2(self):
        # Create instances of EpsilonGreedy and ThompsonSampling
        epsilon_greedy_bandit = EpsilonGreedy(p=0.5, epsilon=0.1)
        thompson_sampling_bandit = ThompsonSampling(p=0.6, alpha=1, beta=1)

        # Run experiments and collect data
        e1 = epsilon_greedy_bandit.experiment(NUMBER_OF_TRIALS)
        e2 = thompson_sampling_bandit.experiment(NUMBER_OF_TRIALS)

        # Calculate win rates
        NUM_TRIALS1 = epsilon_greedy_bandit.N
        NUM_TRIALS2 = thompson_sampling_bandit.N
        win_rates1 = np.array(e1) / (np.arange(NUM_TRIALS1) + 1)
        win_rates2 = np.array(e2) / (np.arange(NUM_TRIALS2) + 1)

        plt.figure(figsize=(12, 6))

        # Plot Epsilon-Greedy data
        plt.plot(win_rates1, label="Epsilon-Greedy")

        # Plot Thompson Sampling data
        plt.plot(win_rates2, label="Thompson Sampling")

        # Add labels and legends
        plt.ylim([0, 1])
        plt.xlabel("Number of Trials")
        plt.ylabel("Average Reward")
        plt.legend()

        plt.show()

#--------------------------------------#

class EpsilonGreedy(Bandit):
    def __init__(self, p, epsilon):
        super().__init__(p)
        self.epsilon = epsilon

    def pull(self):
        if np.random.random() < self.epsilon:
            return np.random.random() < self.p
        else:
            return self.mean < np.random.random()

    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x
        self.epsilon /= (self.N + 1)

    def experiment(self, N):
        rewards = []
        cumulative_rewards = []
        for i in range(N):
            x = self.pull()
            rewards.append(x)
            cumulative_rewards.append(sum(rewards))
            self.update(x)
        return cumulative_rewards

    def __repr__(self):
        return f"EpsilonGreedy({self.p}, {self.epsilon})"

    def report(self):
        # Add your reporting logic here
        pass

#--------------------------------------#

class ThompsonSampling(Bandit):
    def __init__(self, p, alpha, beta):
        super().__init__(p)
        self.alpha = alpha
        self.beta = beta

    def pull(self):
        return np.random.beta(self.alpha, self.beta) < self.p

    def update(self, x):
        self.N += 1
        self.alpha += x
        self.beta += 1 - x
        self.mean = self.alpha / (self.alpha + self.beta)

    def experiment(self, NUMBER_OF_TRIALS):
        rewards = []
        cumulative_rewards = []
        for i in range(NUMBER_OF_TRIALS):
            x = self.pull()
            rewards.append(x)
            cumulative_rewards.append(sum(rewards))
            self.update(x)
        return cumulative_rewards

    def __repr__(self):
        return f"ThompsonSampling({self.p}, {self.alpha}, {self.beta})"

    def report(self):
        # Add your reporting logic here
        pass




def comparison(N, p1, p2, epsilon, alpha, beta):
    bandits = [EpsilonGreedy(p1, epsilon), ThompsonSampling(p1, alpha, beta)]
    rewards = []
    for bandit in bandits:
        cumulative_rewards = bandit.experiment(N)
        rewards.append(cumulative_rewards)
    return rewards

if __name__ == '__main':
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename='mylog.log')

    p1 = 0.5
    p2 = 0.6
    epsilon = 0.1
    alpha = 1
    beta = 1

    bandit1 = EpsilonGreedy(p1, epsilon)
    bandit2 = ThompsonSampling(p2, alpha, beta)

    # Run experiments
    cumulative_rewards1 = bandit1.experiment(NUMBER_OF_TRIALS)
    cumulative_rewards2 = bandit2.experiment(NUMBER_OF_TRIALS)

    # Generate visualizations
    visualization = Visualization()
    rewards_list = [cumulative_rewards1, cumulative_rewards2]
    visualization.plot1(rewards_list)
    visualization.plot2()

    # Comparison of bandit algorithms
    rewards = comparison(NUMBER_OF_TRIALS, p1, p2, epsilon, alpha, beta)

    # Log and report results
    logger.info("Experiment results:")
    logger.info(f"Bandit 1 Cumulative Reward: {cumulative_rewards1[-1]}")
    logger.info(f"Bandit 2 Cumulative Reward: {cumulative_rewards2[-1]}")