from two_thinning.strategies.strategy_base import StrategyBase
from two_thinning.constant_threshold.RL.train import train


class TheThresholdStrategy(StrategyBase):
    # Works only reasonably if m=n
    def __init__(self, n, m, threshold=None, **kwargs):
        super(TheThresholdStrategy, self).__init__(n, m)
        self.threshold = threshold if threshold is not None else train(n=n, m=m, **kwargs)  # l is the constant
        # threshold parameter of this strategy
        self.primary_count = [0] * self.n

    def decide(self, bin):
        if self.primary_count[bin] < self.threshold:
            self.primary_count[bin] += 1
            return True
        else:
            return False

    def note(self, bin):
        pass

    def reset(self):
        self.primary_count = [0] * self.n
