import random

from k_choice.graphical.two_choice.strategies.strategy_base import StrategyBase


class LocalRewardOptimiserStrategy(StrategyBase):
    def __init__(self, graph, m):
        super(LocalRewardOptimiserStrategy, self).__init__(graph, m)

    def decide(self, bin1, bin2):
        curr_max_load = max(self.loads)
        if self.loads[bin1] != curr_max_load and self.loads[bin2] == curr_max_load:
            return True
        elif self.loads[bin2] != curr_max_load and self.loads[bin1] == curr_max_load:
            return False
        return random.choice([True, False])

    def note(self, bin):
        pass

    def reset(self):
        pass
