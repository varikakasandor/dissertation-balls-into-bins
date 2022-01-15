import torch

from k_thinning.strategy_base import StrategyBase
from k_thinning.full_knowledge.RL.DQN.evaluate import load_best_model, get_best_model_path
from k_thinning.full_knowledge.RL.DQN.train import train, greedy


class FullKnowledgeDQNStrategy(StrategyBase):
    def __init__(self, n, m, k, use_pre_trained=True):
        super(FullKnowledgeDQNStrategy, self).__init__(n, m, k)
        self.model = None
        if use_pre_trained:
            self.model = load_best_model(n=n, m=m, k=k)
            if self.model is None:
                print("There is no trained model yet with the given parameters, so a new one will be trained now.")
        if self.model is None:
            self.model = train(n=n, m=m, k=k)
            torch.save(self.model.state_dict(), get_best_model_path(n=n, m=m, k=k))

        self.choices_left = k

    def decide(self, bin):
        a = greedy(self.model, self.loads, self.choices_left)
        if self.loads[bin] <= a:
            return True
        else:
            self.choices_left -= 1
            return False

    def note(self, bin):
        self.choices_left = self.k

    def reset(self):
        pass
