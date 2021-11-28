from math import log

from two_thinning.strategy_base import StrategyBase


class TheRelativeThresholdStrategy(StrategyBase):
    def __init__(self, n, m, limit):
        super(TheRelativeThresholdStrategy, self).__init__(n, m)
        self.limit = limit  # l is the constant threshold parameter of this strategy
        self.primary_count = [0] * n
        self.round = 0

    def decide(self, bin):
        few_primary = self.primary_count[bin] < (self.limit + (self.round-1)/self.n) # Note: it is a bit strange that for self.round=0 it has (-1)/self.n, but according to the formula this is correct
        small_load = (self.loads[bin]-self.round/self.n) < log(self.n)
        if few_primary or small_load:
            self.primary_count[bin] += 1
            return True
        else:
            return False

    def note(self, bin):
        self.loads[bin] += 1
