import torch

from k_thinning.strategies.strategy_base import StrategyBase
from k_thinning.full_knowledge.RL.DQN.evaluate import load_best_model, get_best_model_path
from k_thinning.full_knowledge.RL.DQN.train import train, greedy


class FullKnowledgeDQNStrategy(StrategyBase):
    def __init__(self, n, m, k, use_pre_trained=True):
        super(FullKnowledgeDQNStrategy, self).__init__(n, m, k)
        existing_model = load_best_model(n=n, m=m, k=k)
        if existing_model is None or not use_pre_trained:
            version = ""
            if existing_model is not None and not use_pre_trained:
                print(
                    "There already exist a model with these parameters, so a new temporary version will be saved now.")
                version = "_tmp"
            elif existing_model is None and use_pre_trained:
                print("There is no trained model yet with the given parameters, so a new one will be trained now.")
            self.model = train(n=n, m=m, k=k)
            torch.save(self.model.state_dict(), get_best_model_path(n=n, m=m, k=k)[:-4] + version + '.pth')
        else:
            self.model = existing_model

    def decide(self, bin):
        a = greedy(self.model, self.loads, self.choices_left)
        if self.loads[bin] <= a:
            return True
        else:
            return False

    def note(self, bin):
        pass

    def reset(self):
        pass
