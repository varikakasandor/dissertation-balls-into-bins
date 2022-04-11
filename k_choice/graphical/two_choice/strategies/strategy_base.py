from abc import ABCMeta, abstractmethod
from k_choice.graphical.two_choice.graphs.graph_base import GraphBase


class StrategyBase(metaclass=ABCMeta):

    def __init__(self, graph: GraphBase, m):
        self.graph = graph
        self.m = m
        self.loads = [0] * graph.n

    @abstractmethod
    def decide(self, bin1, bin2):
        pass

    @abstractmethod
    def reset(self):
        pass

    def reset_(self):
        self.loads = [0] * self.graph.n
        self.reset()

    def decide_(self, bin1, bin2):
        decision = self.decide(bin1, bin2)
        self.loads[bin1 if decision else bin2] += 1
        return decision
