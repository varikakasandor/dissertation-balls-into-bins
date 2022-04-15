from math import *

from two_thinning.strategies.strategy_base import StrategyBase
from two_thinning.full_knowledge.RL.DQN.evaluate import load_best_model, get_best_model_path # Works for normalised too, no difference
from two_thinning.full_knowledge.RL.DQN.train import greedy, train
from two_thinning.full_knowledge.RL.DQN.constants import *
from two_thinning.full_knowledge.RL.DQN.neural_network import *


class FullKnowledgeDQNStrategy(StrategyBase):
    def __init__(self, n, m, use_pre_trained=True, nn_type=NN_TYPE, use_normalised=USE_NORMALISED,
                 max_threshold=MAX_THRESHOLD, **kwargs):

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
            self.model = train(n=n, m=m, max_threshold=max_threshold, use_normalised=use_normalised, **kwargs)
            torch.save(self.model.state_dict(), get_best_model_path(n=n, m=m, nn_type=nn_type)[:-4] + version + '.pth')
        else:
            self.model = existing_model

        self.use_normalised = use_normalised
        self.max_threshold = max_threshold - m // n if use_normalised else max_threshold

    def decide(self, bin):
        a = greedy(self.model, self.loads)
        self.curr_thresholds.append(a)
        if self.use_normalised:
            return self.loads[bin] <= sum(self.loads) / len(self.loads) + a - self.max_threshold
        else:
            return self.loads[bin] <= a

    def note(self, bin):
        pass

    def reset(self):
        pass

    def create_analyses(self, save_path):
        self.create_plot(save_path)

    def create_summary(self, save_path):
        self.create_summary_plot(save_path)
