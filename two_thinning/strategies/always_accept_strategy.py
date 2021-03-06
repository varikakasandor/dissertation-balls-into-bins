from two_thinning.strategies.strategy_base import StrategyBase


class AlwaysAcceptStrategy(StrategyBase):
    # Note: this is exactly the same as one-choice, and as AlwaysReject
    def __init__(self, n, m):
        super(AlwaysAcceptStrategy, self).__init__(n, m)

    def decide(self, bin):
        return True

    def note(self, bin):
        pass

    def reset(self):
        pass
