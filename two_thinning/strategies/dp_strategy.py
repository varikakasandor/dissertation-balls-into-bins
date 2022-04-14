from two_thinning.strategies.strategy_base import StrategyBase
from two_thinning.full_knowledge.dp import find_best_strategy
from two_thinning.full_knowledge.RL.DQN.constants import *


class DPStrategy(StrategyBase):
    def __init__(self, n, m, reward_fun=REWARD_FUN):
        super(DPStrategy, self).__init__(n, m)
        self.strategy = find_best_strategy(n=n, m=m, reward=reward_fun, use_threshold_dp=True, print_behaviour=False)

    def decide(self, bin):
        sorted_loads = tuple(sorted(self.loads))
        _, threshold = self.strategy[sorted_loads]
        self.curr_thresholds.append(threshold)
        return self.loads[bin] <= threshold  # either accept is better (-1) or it doesn't matter (0)

    def note(self, bin):
        pass

    def reset(self):
        pass

    def create_analyses(self, save_path):
        self.create_plot(save_path)

    def create_summary(self, save_path):
        self.create_summary_plot(save_path)
