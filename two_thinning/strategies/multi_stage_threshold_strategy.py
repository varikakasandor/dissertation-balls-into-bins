from math import log, floor

from two_thinning.strategies.strategy_base import StrategyBase


class MultiStageThresholdStrategy(StrategyBase):
    def __init__(self, n, m, T, L_0, l):
        # TODO: In itself should be called with t=m/n, L_0=0, l=(log n)^beta_k
        super(MultiStageThresholdStrategy, self).__init__(n, m)
        self.T = T
        self.L_0 = L_0
        self.l = l
        self.k = max(floor(log(log(self.n)) / (3 * log(log(log(self.n))))),1)  # TODO: this is =1 for all reasonable values of n (n<10^150), so the whole strategy is simply the threshold strategy
        self.alpha = log(self.T) / (log(log(self.n)))
        self.eta = 0  # TODO: it can be anything between (alpha-0.5)/(4k-2), but it does not seem to be a parameter of the algorithm
        self.BETA = self.alpha + self.eta
        self.epsilon = (2 * self.BETA - 1) / (2 * (self.k + 1))
        self.beta = [(2 * self.BETA - 1 - self.epsilon) * i / (2 * self.k + 1) for i in range(self.k+1)]
        self.t = [0] + [floor(T - (log(n) ** (self.beta[i]))) for i in range(1, self.k)] + [T]

        self.stage = 1
        self.H = [[]]  # TODO: if we do not start with empty bins, then different
        self.H_was = [False] * self.n
        self.round = 0
        self.primary_accepted = [0] * self.n

    def decide(self, bin):
        load_diff = self.loads[bin] - self.round / self.n
        accepted_much = (self.primary_accepted[bin] >= self.t[self.stage] - self.t[self.stage - 1] + self.l)
        if (load_diff >= -log(self.n)) and (self.H_was[bin] or accepted_much):  # TODO: assumes H[0]=[]
            return False
        else:
            self.primary_accepted[bin] += 1
            return True

    def note(self, bin):
        self.round += 1
        if self.round >= self.n * self.t[self.stage]:  # TODO: ==, but I am afraid of rounding issues leading to
            # missing t_1
            self.primary_accepted = [0] * self.n
            became_too_high = [i for i in range(self.n) if (not self.H_was[i]) and (
                    self.loads[i] - self.round / self.n >= self.L_0 + 2 * self.stage * self.l)]
            self.H.append(became_too_high)
            for h in became_too_high:
                self.H_was[h] = True
            self.stage += 1

    def reset(self):
        self.stage = 1
        self.H = [[]]
        self.H_was = [False] * self.n
        self.round = 0
        self.primary_accepted = [0] * self.n
