from math import *

from two_thinning.strategies.strategy_base import StrategyBase
from two_thinning.full_knowledge.RL.DQN.evaluate import load_best_model, get_best_model_path # Works for normalised too, no difference
from two_thinning.full_knowledge.RL.DQN.train import greedy, train
from two_thinning.full_knowledge.RL.DQN.normalised_load.train import train as train_normalised


from two_thinning.full_knowledge.RL.DQN.neural_network import *


class FullKnowledgeDQNStrategy(StrategyBase):
    def __init__(self, n, m, nn_model=GeneralNet, nn_type="general_net", use_normalised_load=False,
                 max_threshold=None, use_pre_trained=True):
        super(FullKnowledgeDQNStrategy, self).__init__(n, m)
        self.use_normalised_load = use_normalised_load
        self.max_threshold = max_threshold if max_threshold is not None else max(3, m // n + ceil(sqrt(log(n))))
        existing_model = load_best_model(n=n, m=m, nn_type=nn_type)
        if existing_model is None or not use_pre_trained:
            version = ""
            if existing_model is not None and not use_pre_trained:
                print(
                    "There already exist a model with these parameters, so a new temporary version will be saved now.")
                version = "_tmp"
            elif existing_model is None and use_pre_trained:
                print("There is no trained model yet with the given parameters, so a new one will be trained now.")
            train_fn = train_normalised if use_normalised_load else train
            self.model = train_fn(n=n, m=m, nn_model=nn_model, optimise_freq=int(sqrt(m)),
                                  max_threshold=self.max_threshold)
            torch.save(self.model.state_dict(), get_best_model_path(n=n, m=m, nn_type=nn_type)[:-4] + version + '.pth')
        else:
            self.model = existing_model

    def decide(self, bin):
        a = greedy(self.model, self.loads)
        self.curr_thresholds.append(a)
        if self.use_normalised_load:
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
