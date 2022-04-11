from two_thinning.strategies.strategy_base import StrategyBase


class MeanThinningStrategy(StrategyBase):
    def __init__(self, n, m):
        super(MeanThinningStrategy, self).__init__(n, m)

    def decide(self, bin):
        return self.loads[bin] <= sum(self.loads) / len(self.loads)  # TODO: consider < instead of <=

    def note(self, bin):
        pass

    def reset(self):
        pass
