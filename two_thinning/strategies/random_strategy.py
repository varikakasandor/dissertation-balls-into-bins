from two_thinning.strategy_base import StrategyBase
import random


class RandomStrategy(StrategyBase):
    def __init__(self, n, m):
        super(RandomStrategy, self).__init__(n, m)

    def decide(self, bin):
        return random.choice([True, False])

    def note(self, bin):
        pass

    def reset(self):
        pass
