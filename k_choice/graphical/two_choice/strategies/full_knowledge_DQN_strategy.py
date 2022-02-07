import torch
from math import *

from k_choice.graphical.two_choice.full_knowledge.RL.DQN.evaluate import load_best_model, get_best_model_path
from k_choice.graphical.two_choice.full_knowledge.RL.DQN.neural_network import FullGraphicalTwoChoiceFCNet
from k_choice.graphical.two_choice.full_knowledge.RL.DQN.train import train, greedy
from k_choice.graphical.two_choice.graph_base import GraphBase
from k_choice.graphical.two_choice.strategy_base import StrategyBase


class FullKnowledgeDQNStrategy(StrategyBase):

    def __init__(self, graph: GraphBase, m, nn_model=FullGraphicalTwoChoiceFCNet, nn_type="fc_cycle", use_pre_trained=True):
        super(FullKnowledgeDQNStrategy, self).__init__(graph, m)
        existing_model = load_best_model(n=graph.n, m=m, nn_type=nn_type)
        if existing_model is None or not use_pre_trained:
            version = ""
            if existing_model is not None and not use_pre_trained:
                print(
                    "There already exist a model with these parameters, so a new temporary version will be saved now.")
                version = "_tmp"
            elif existing_model is None and use_pre_trained:
                print("There is no trained model yet with the given parameters, so a new one will be trained now.")
            self.model = train(graph=graph, m=m, nn_model=nn_model, optimise_freq=int(sqrt(m)))
            torch.save(self.model.state_dict(), get_best_model_path(n=graph.n, m=m, nn_type=nn_type)[:-4] + version + '.pth')
        else:
            self.model = existing_model

    def decide(self, bin1, bin2):
        chosen = greedy(self.model, self.loads, (bin1, bin2))
        return chosen == bin1

    def reset(self):
        pass
