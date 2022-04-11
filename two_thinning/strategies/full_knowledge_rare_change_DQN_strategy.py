import torch
from math import *

from two_thinning.strategies.strategy_base import StrategyBase
from two_thinning.full_knowledge.RL.DQN.rare_change.evaluate import load_best_model, get_best_model_path
from two_thinning.full_knowledge.RL.DQN.rare_change.train import train, greedy


class FullKnowledgeRareChangeDQNStrategy(StrategyBase):
    def __init__(self, n, m, use_pre_trained=True):
        super(FullKnowledgeRareChangeDQNStrategy, self).__init__(n, m)
        self.threshold_change_freq = int(sqrt(m))
        self.curr_action = None
        self.round = 0
        existing_model = load_best_model(n=n, m=m)
        if existing_model is None or not use_pre_trained:
            version = ""
            if existing_model is not None and not use_pre_trained:
                print(
                    "There already exist a model with these parameters, so a new temporary version will be saved now.")
                version = "_tmp"
            elif existing_model is None and use_pre_trained:
                print("There is no trained model yet with the given parameters, so a new one will be trained now.")
            self.model = train(n=n, m=m, optimise_freq=int(sqrt(m)), max_threshold=max(3, 2 * (m + n - 1) // n),
                               threshold_change_freq=self.threshold_change_freq)
            torch.save(self.model.state_dict(), get_best_model_path(n=n, m=m)[:-4] + version + '.pth')
        else:
            self.model = existing_model

    def decide(self, bin):
        if self.round % self.threshold_change_freq == 0:
            self.curr_action = greedy(self.model, self.loads)
        self.curr_thresholds.append(self.curr_action)
        return self.loads[bin] <= self.curr_action

    def note(self, bin):
        self.round += 1

    def reset(self):
        self.round = 0
        self.curr_action = None

    def create_analyses(self, save_path):
        self.create_plot(save_path)

    def create_summary(self, save_path):
        self.create_summary_plot(save_path)
