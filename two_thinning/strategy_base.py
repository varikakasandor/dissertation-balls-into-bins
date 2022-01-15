from abc import ABCMeta, abstractmethod


class StrategyBase(metaclass=ABCMeta):

    def __init__(self, n, m):
        self.loads = [0] * n
        self.n = n
        self.m = m

    @abstractmethod
    def decide(self, bin):
        pass

    @abstractmethod
    def note(self, bin):
        pass

    @abstractmethod
    def reset(self):
        pass

    def note_(self, bin):
        self.loads[bin] += 1
        self.note(bin)

    def reset_(self):
        self.loads = [0] * self.n
        self.reset()

    def decide_(self, bin):
        return self.decide(bin)