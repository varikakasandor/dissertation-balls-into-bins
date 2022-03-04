from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class StrategyBase(metaclass=ABCMeta):

    def __init__(self, n, m):
        self.loads = [0] * n
        self.n = n
        self.m = m
        self.curr_thresholds = []
        self.max_loads = []
        self.chosen_loads = []
        self.thresholds = []

    @abstractmethod
    def decide(self, bin):
        pass

    @abstractmethod
    def note(self, bin):
        pass

    @abstractmethod
    def reset(self):
        pass

    def create_analyses(self, save_path):  # Does not necessarily need to be overridden
        pass

    def create_summary(self, save_path):  # Does not necessarily need to be overridden
        pass

    def note_(self, bin):
        self.loads[bin] += 1
        self.max_loads.append(max(self.loads))
        self.chosen_loads.append(self.loads[bin])
        self.note(bin)

    def reset_(self):
        self.loads = [0] * self.n
        self.thresholds.append(self.curr_thresholds)
        self.curr_thresholds = []
        self.max_loads = []
        self.chosen_loads = []
        self.reset()

    def decide_(self, bin):
        return self.decide(bin)

    def create_analyses_(self, save_path):
        self.create_analyses(save_path)
        plt.clf()

    def create_summary_(self, save_path):
        self.create_summary(save_path)
        plt.clf()

    def create_plot(self, save_path):  # Helper function for those strategies which decide based on a
        # threshold
        x = np.arange(self.m)
        plt.plot(x, np.array(self.curr_thresholds), label="threshold")
        plt.plot(x, np.array(self.max_loads), label="max load")
        plt.plot(x, np.array(self.chosen_loads), label="chosen load")
        plt.title("Threshold progression")
        plt.xlabel("Ball")
        plt.ylabel("Chosen threshold")
        plt.legend()
        plt.savefig(save_path)

    def create_summary_plot(self, save_path):  # Helper function for those strategies which decide based on a
        # threshold
        plt.plot(np.array(self.thresholds).T)
        plt.title("Threshold progression (multiple runs)")
        plt.xlabel("Ball")
        plt.ylabel("Chosen threshold")
        plt.savefig(save_path)
