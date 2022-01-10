from abc import ABCMeta, abstractmethod


class StrategyBase(metaclass=ABCMeta):

    def __init__(self, n, m, k):
        self.loads = [0] * n
        self.n = n
        self.m = m
        self.k = k

    @abstractmethod
    def decide(self, bin):  # TODO: might add a choices_left as an input
        pass

    @abstractmethod
    def note(self, bin):
        pass

    @abstractmethod
    def reset(self):
        pass

    def decide_(self, bin):
        return self.decide(bin)

    def note_(self, bin):
        self.loads[bin] += 1
        self.note(bin)

    def reset_(self):
        self.loads = [0] * self.n
        self.reset()
