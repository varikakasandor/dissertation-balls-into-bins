import copy

from two_thinning.strategies.strategy_base import StrategyBase
from two_thinning.full_knowledge.RL.DQN.constants import *


class LocalRewardOptimiserStrategy(StrategyBase):
    def __init__(self, n, m, reward_fun=REWARD_FUN, potential_fun=POTENTIAL_FUN):
        super(LocalRewardOptimiserStrategy, self).__init__(n, m)
        self.reward_fun = reward_fun
        self.potential_fun = potential_fun

    def calc_local_reward(self, bin):
        next_loads = copy.deepcopy(self.loads)
        next_loads[bin] += 1
        if sum(next_loads) == self.m:
            return self.reward_fun(next_loads) - self.potential_fun(self.loads)  # Note: the potential_fun(
        # self.loads) is not even needed, since it is the same for both reject and accept, doesn't make a difference
        else:
            return self.potential_fun(next_loads) - self.potential_fun(self.loads)


    def decide(self, bin):
        # called "Away from max"
        reward_accept = self.calc_local_reward(bin)
        reward_reject = 0.0
        for i in range(self.n):
            reward_reject += self.calc_local_reward(i)
        reward_reject /= self.n

        return reward_accept >= reward_reject

    def note(self, bin):
        pass

    def reset(self):
        pass
