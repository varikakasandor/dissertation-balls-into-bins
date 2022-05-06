import random
from os.path import join, dirname, abspath
import numpy as np
from matplotlib import pyplot as plt

from two_thinning.strategies.strategy_base import StrategyBase
from two_thinning.full_knowledge.RL.DQN.constants import REWARD_FUN


class ConstantOffsetStrategy(StrategyBase):
    def __init__(self, n, m, offset=None, reward_fun=REWARD_FUN):
        super(ConstantOffsetStrategy, self).__init__(n, m)
        self.reward_fun = reward_fun
        self.offset = offset if offset is not None else self.find_best_offset()

    def find_best_offset(self, min_offset=-20, max_offset=20, runs_per_offset=100):
        best_avg = - self.m - 1
        best_offset = None
        vals = []
        for offset in range(min_offset, max_offset+1):
            curr_sum = 0
            curr_max_loads = []
            for _ in range(runs_per_offset):
                loads = [0] * self.n
                for i in range(self.m):
                    first_choice = random.randrange(self.n)
                    if loads[first_choice] <= sum(loads) / self.n + offset:
                        final_choice = first_choice
                    else:
                        final_choice = random.randrange(self.n)
                    loads[final_choice] += 1

                curr_sum += self.reward_fun(loads)
                curr_max_loads.append(max(loads))
            vals.append(curr_max_loads)
            curr_avg = curr_sum / runs_per_offset
            if curr_avg > best_avg:
                best_avg = curr_avg
                best_offset = offset

        xs = list(range(min_offset, max_offset+1))
        avgs = [np.mean(np.array(l)) for l in vals]
        stds = [np.std(np.array(l)) for l in vals]

        plt.rcParams['font.size'] = '14'
        plt.errorbar(xs, avgs, yerr=stds, label="standard deviation")
        plt.axhline(y=66.58, color='r', linestyle='-', label="One-Choice/Always Accept")
        plt.xlabel("offset")
        plt.ylabel("average max load on 100 runs")
        plt.xticks(np.arange(min_offset, max_offset+1, step=4))
        plt.legend()
        file_name = f"offset_analysis_{self.n}_{self.m}.pdf"
        save_path = join(dirname(dirname(dirname(abspath(__file__)))), "evaluation", "two_thinning", "data", file_name)
        plt.savefig(save_path)
        return best_offset

    def decide(self, bin):
        return self.loads[bin] <= sum(self.loads) / self.n + self.offset

    def note(self, bin):
        pass

    def reset(self):
        pass
