from k_choice.graphical.two_choice.full_knowledge.RL.DQN.constants import *
from k_choice.graphical.two_choice.full_knowledge.dp import find_best_strategy
from k_choice.graphical.two_choice.strategies.strategy_base import StrategyBase


class DPStrategy(StrategyBase):

    def __init__(self, graph: GraphBase, m, reward_fun=REWARD_FUN):
        super(DPStrategy, self).__init__(graph, m)
        self.strategy = find_best_strategy(graph=graph, m=m, reward_fun=reward_fun, print_behaviour=False)

    def decide(self, bin1, bin2):
        if (tuple(self.loads), (bin1, bin2)) not in self.strategy:
            print("HELLO")
        return self.strategy[(tuple(self.loads), (bin1, bin2))] <= 0

    def reset(self):
        pass
