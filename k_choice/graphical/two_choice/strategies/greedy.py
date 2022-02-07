from k_choice.graphical.two_choice.graph_base import GraphBase
from k_choice.graphical.two_choice.strategy_base import StrategyBase


class Greedy(StrategyBase):

    def __init__(self, graph: GraphBase, m):
        super(Greedy, self).__init__(graph, m)

    def decide(self, bin1, bin2):
        return self.loads[bin1] <= self.loads[bin2]  # TODO: we could do random instead in case of tie

    def reset(self):
        pass
