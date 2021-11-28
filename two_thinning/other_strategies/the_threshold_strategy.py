from two_thinning.strategy_base import StrategyBase


class TheThresholdStrategy(StrategyBase):
    # Works only reasonably if m=n
    def __init__(self, n, m, limit):
        super(TheThresholdStrategy, self).__init__(n, m)
        self.limit = limit  # l is the constant threshold parameter of this strategy
        self.primary_count = [0] * n

    def decide(self, bin):
        if self.primary_count[bin] < self.limit:
            self.primary_count[bin] += 1
            return True
        else:
            return False

    def note(self, bin):
        pass
