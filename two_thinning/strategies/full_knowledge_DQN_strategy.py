from two_thinning.strategy_base import StrategyBase
from two_thinning.full_knowledge.RL.DQN.evaluate import load_best_model
from two_thinning.full_knowledge.RL.DQN.train import train, greedy


class FullKnowledgeDQNStrategy(StrategyBase):
    def __init__(self, n, m, use_pre_trained=True):
        super(FullKnowledgeDQNStrategy, self).__init__(n, m)
        if use_pre_trained:
            self.model = load_best_model(n=n, m=m)
            if self.model is None:
                print("There is no trained model yet with the given parameters, so a new one will be trained now.")
                self.model = train(n=n, m=m)
        else:
            self.model = train(n=n, m=m)
        self.loads = [0] * self.n

    def decide(self, bin):
        a = greedy(self.model, self.loads)
        return self.loads[bin] <= a.item()

    def note(self, bin):
        self.loads[bin] += 1

    def reset(self):
        self.loads = [0] * self.n
