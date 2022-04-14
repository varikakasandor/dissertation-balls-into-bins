from math import floor

from k_thinning.strategies.strategy_base import StrategyBase


class MeanThinningStrategy(StrategyBase):
    def __init__(self, n, m, k):
        super(MeanThinningStrategy, self).__init__(n, m, k)

    def decide(self, bin):
        threshold_idx = self.n * (1 - 2 ** (-1 / self.choices_left))
        threshold = sorted(self.loads)[floor(threshold_idx)]
        return self.loads[bin] <= threshold

    def note(self, bin):
        pass

    def reset(self):
        pass
