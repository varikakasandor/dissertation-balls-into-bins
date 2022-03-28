from two_thinning.strategy_base import StrategyBase
from two_thinning.full_knowledge.dp import find_best_strategy, rindex
from two_thinning.full_knowledge.RL.DQN.constants import *


class DPStrategy(StrategyBase):
    def __init__(self, n, m, reward_fun=REWARD_FUN):
        super(DPStrategy, self).__init__(n, m)
        self.strategy = find_best_strategy(n=n, m=m, reward=reward_fun)

    def decide(self, bin):
        sorted_loads = tuple(sorted(self.loads))

        # Generating (an artificial "threshold")
        threshold = sorted_loads[0]
        for x in sorted_loads:
            _, decision = self.strategy[(sorted_loads, x)]
            if decision < 1:
                threshold = x
            else:
                break
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
