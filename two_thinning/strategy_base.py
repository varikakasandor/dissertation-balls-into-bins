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
