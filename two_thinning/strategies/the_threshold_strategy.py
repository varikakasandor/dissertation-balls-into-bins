from two_thinning.strategies.strategy_base import StrategyBase
from two_thinning.constant_threshold.RL.train import train


class TheThresholdStrategy(StrategyBase):
    # Works only reasonably if m=n
    def __init__(self, n, m, limit=None):
        super(TheThresholdStrategy, self).__init__(n, m)
        self.limit = limit if limit is not None else train(n=n, m=m, primary_only=True)  # l is the constant threshold parameter of this
        # strategy
        self.primary_count = [0] * self.n

    def decide(self, bin):
        if self.primary_count[bin] < self.limit:
            self.primary_count[bin] += 1
            return True
        else:
            return False

    def note(self, bin):
        pass

    def reset(self):
        self.primary_count = [0] * self.n
