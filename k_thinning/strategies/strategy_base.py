from abc import ABCMeta, abstractmethod


class StrategyBase(metaclass=ABCMeta):

    def __init__(self, n, m, k):
        self.loads = [0] * n
        self.n = n
        self.m = m
        self.k = k
        self.choices_left = k

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
        decision = self.decide(bin)
        if not decision:
            self.choices_left -= 1
        return decision

    def note_(self, bin):
        self.loads[bin] += 1
        self.choices_left = self.k
        self.note(bin)

    def reset_(self):
        self.loads = [0] * self.n
        self.reset()
