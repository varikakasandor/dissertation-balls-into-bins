from math import log, sqrt
import numpy as np
import heapq

from two_thinning.strategy_base import StrategyBase


class DriftStrategy(StrategyBase):
    def __init__(self, n, m, theta):
        super(DriftStrategy, self).__init__(n, m)
        if (theta <= 0) or (theta > sqrt(5) - 2):
            raise Exception("Theta not in valid range")
        self.theta = theta
        self.points = [(np.random.default_rng().exponential(1 / (1 + theta)), i) for i in
                       range(self.n)]  # TODO: check if 1/(1+theta) or simply 1/theta
        heapq.heapify(self.points)
        self.cnt_points = [0] * self.n
        self.t = 0

    def decide(self, bin):
        sum_lambdas = sum(
            [(1 + self.theta) if self.cnt_points[i] < self.t else (1 - self.theta) for i in range(self.n)])
        lambda_i = (1 + self.theta) if self.cnt_points[bin] < self.t else (1 - self.theta)
        p_i = lambda_i / sum_lambdas
        c = 2 * self.theta / (
                    1 - self.theta)  # TODO: c can be anything between 2theta/(1-theta) and (1-theta)/(1+theta), it all works out for any such c
        u = np.random.default_rng().uniform()
        if self.n * p_i - c >= u:
            return True
        else:
            return False

    def note(self, bin):
        self.t, what = heapq.heappop(self.points)
        self.cnt_points[what] += 1
        if self.cnt_points[what] < self.t:
            new_event = self.t + np.random.default_rng().exponential(1 / (1 + self.theta))
        else:
            new_event_candidate = self.t + np.random.default_rng().exponential(1 / (1 - self.theta))
            if new_event_candidate <= self.cnt_points[what]:
                new_event = new_event_candidate
            else:
                new_event = self.cnt_points[what] + np.random.default_rng().exponential(1 / (1 + self.theta))

        heapq.heappush(self.points, (new_event, what))

    def reset(self):
        self.points = [(np.random.default_rng().exponential(1 / (1 + self.theta)), i) for i in
                       range(self.n)]  # TODO: check if 1/(1+theta) or simply 1/theta
        heapq.heapify(self.points)
        self.cnt_points = [0] * self.n
        self.t = 0
