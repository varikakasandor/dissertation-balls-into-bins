import random

from k_thinning.strategies.strategy_base import StrategyBase


class RandomStrategy(StrategyBase):
    def __init__(self, n, m, k):
        super(RandomStrategy, self).__init__(n, m, k)
        self.accept_time = random.randrange(1, self.k)  # 1 means reject

    def decide(self, bin):
        return self.choices_left == self.accept_time

    def note(self, bin):
        self.accept_time = random.randrange(1, self.k)

    def reset(self):
        pass
