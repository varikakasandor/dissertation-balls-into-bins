import torch

from two_thinning.strategy_base import StrategyBase
from two_thinning.full_knowledge.RL.DQN.evaluate import load_best_model, get_best_model_path
from two_thinning.full_knowledge.RL.DQN.train import train, greedy


class FullKnowledgeDQNStrategy(StrategyBase):
    def __init__(self, n, m, use_pre_trained=True):
        super(FullKnowledgeDQNStrategy, self).__init__(n, m)
        existing_model = load_best_model(n=n, m=m)
        if existing_model is None or not use_pre_trained:
            version = ""
            if existing_model is not None and not use_pre_trained:
                print("There already exist a model with these parameters, so a new temporary version will be saved now.")
                version = "_tmp"
            elif existing_model is None and use_pre_trained:
                print("There is no trained model yet with the given parameters, so a new one will be trained now.")
            self.model = train(n=n, m=m)
            torch.save(self.model.state_dict(), get_best_model_path(n=n, m=m)[:-4] + version + '.pth')
        else:
            self.model=existing_model

    def decide(self, bin):
        a = greedy(self.model, self.loads)
        self.curr_thresholds.append(a)
        return self.loads[bin] <= a

    def note(self, bin):
        pass

    def reset(self):
        pass

    def create_analyses(self, save_path):
        self.create_plot(save_path)

    def create_summary(self, save_path):
        self.create_summary_plot(save_path)