from math import log, sqrt
import numpy as np
import heapq

from two_thinning.strategy_base import StrategyBase


class TheRelativeThresholdStrategy(StrategyBase):
    def __init__(self, n, m, theta):
        super(TheRelativeThresholdStrategy, self).__init__(n, m)
        if ((theta<=0) or (theta>sqrt(5)-2)):
            raise Exception("Theta not in valid range")
        self.theta = theta
        self.points = [np.random.default_rng().exponential(1/(1+theta)) for i in range(n)]  # TODO: check if 1/(1+theta) or simply 1/theta
        heapq.heapify(self.points)
        self.cnt_points = [0] * n
        self.t = 0

    def decide(self, bin):
        sum_lambdas = sum([(1+self.theta) if self.cnt_points[i]<self.t else (1-self.theta) for i in range(self.n)])
        lambda_i = (1+self.theta) if self.cnt_points[bin]<self.t else (1-self.theta)
        p_i = lambda_i/sum_lambdas
        c = 2*self.theta/(1-self.theta)  # TODO: c can be anything between 2theta/(1-theta) and (1-theta)/(1+theta), it all works out for any such c
        u = np.random.default_rng().uniform()
        if self.n*p_i-c>=u:
            return True
        else:
            return False

    def note(self, bin):
        self.loads[bin] += 1
