import torch
from math import *

from two_thinning.strategy_base import StrategyBase
from two_thinning.full_knowledge.RL.DQN.evaluate import load_best_model, get_best_model_path
from two_thinning.full_knowledge.RL.DQN.train import train, greedy
from two_thinning.full_knowledge.RL.DQN.neural_network import FullTwoThinningRecurrentNetFC, FullTwoThinningRecurrentNet


class FullKnowledgeDQNStrategy(StrategyBase):
    def __init__(self, n, m, nn_model=FullTwoThinningRecurrentNetFC, nn_type="rnn_fc", use_pre_trained=True):
        super(FullKnowledgeDQNStrategy, self).__init__(n, m)
        existing_model = load_best_model(n=n, m=m, nn_type=nn_type)
        if existing_model is None or not use_pre_trained:
            version = ""
            if existing_model is not None and not use_pre_trained:
                print(
                    "There already exist a model with these parameters, so a new temporary version will be saved now.")
                version = "_tmp"
            elif existing_model is None and use_pre_trained:
                print("There is no trained model yet with the given parameters, so a new one will be trained now.")
            self.model = train(n=n, m=m, nn_model=nn_model, optimise_freq=int(sqrt(m)),
                               max_threshold=max(3, 2 * (m + n - 1) // n))
            torch.save(self.model.state_dict(), get_best_model_path(n=n, m=m, nn_type=nn_type)[:-4] + version + '.pth')
        else:
            self.model = existing_model

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
