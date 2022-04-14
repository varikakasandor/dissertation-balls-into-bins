from k_thinning.strategies.strategy_base import StrategyBase


class LocalRewardOptimiserStrategy(StrategyBase):
    def __init__(self, n, m, k):
        super(LocalRewardOptimiserStrategy, self).__init__(n, m, k)
        # TODO: make it more generic, now it assumes MAX_LOAD_POTENTIAL


    def decide(self, bin):
        curr_max_load = max(self.loads)
        return self.loads[bin] < curr_max_load

    def note(self, bin):
        pass

    def reset(self):
        pass
