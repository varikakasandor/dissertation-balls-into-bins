import random

from k_choice.graphical.two_choice.graphs.graph_base import GraphBase
from k_choice.graphical.two_choice.strategies.strategy_base import StrategyBase


class RandomStrategy(StrategyBase):

    def __init__(self, graph: GraphBase, m):
        super(RandomStrategy, self).__init__(graph, m)

    def decide(self, bin1, bin2):
        return random.choice([True, False])

    def reset(self):
        pass
