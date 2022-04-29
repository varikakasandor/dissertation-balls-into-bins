import random

from two_thinning.strategies.strategy_base import StrategyBase
from two_thinning.full_knowledge.RL.DQN.constants import REWARD_FUN


class ConstantOffsetStrategy(StrategyBase):
    def __init__(self, n, m, offset=None, reward_fun=REWARD_FUN):
        super(ConstantOffsetStrategy, self).__init__(n, m)
        self.reward_fun = reward_fun
        self.offset = offset if offset is not None else self.find_best_offset()

    def find_best_offset(self, runs_per_offset=100):
        best_avg = - self.m - 1
        best_offset = None
        for offset in range(-5, 6):
            curr_score = 0
            for _ in range(runs_per_offset):
                loads = [0] * self.n
                for i in range(self.m):
                    first_choice = random.randrange(self.n)
                    if loads[first_choice] <= sum(loads) / self.n + offset:
                        final_choice = first_choice
                    else:
                        final_choice = random.randrange(self.n)
                    loads[final_choice] += 1

                curr_score += self.reward_fun(loads)
            curr_avg = curr_score / runs_per_offset
            if curr_avg > best_avg:
                best_avg = curr_avg
                best_offset = offset

        return best_offset

    def decide(self, bin):
        return self.loads[bin] <= sum(self.loads) / self.n + self.offset

    def note(self, bin):
        pass

    def reset(self):
        pass
