from math import sqrt, log, ceil, exp
from os.path import join, dirname, abspath
from datetime import datetime

import torch.optim

from k_choice.graphical.two_choice.full_knowledge.RL.DQN.constants import *
from k_choice.graphical.two_choice.graphs.cycle import Cycle
from k_choice.graphical.two_choice.graphs.hypercube import HyperCube
from k_choice.graphical.two_choice.graphs.complete_graph import CompleteGraph
from k_choice.graphical.two_choice.graphs.random_regular_graph import RandomRegularGraph

from k_choice.graphical.two_choice.graphs.graph_base import GraphBase
from k_choice.graphical.two_choice.full_knowledge.RL.DQN.neural_network import *
from helper.helper import std


def get_dqn_hyperparameters(graph: GraphBase, m):
    if isinstance(graph, Cycle) and graph.n == 4 and m == 25:
        return {
            "batch_size": 64,
            "eps_start": 0.4,
            "eps_end": 0.065,
            "eps_decay": 2900,
            "num_episodes": 2000,
            "pre_train_episodes": 75,
            "target_update_freq": 8,
            "memory_capacity": 650,
            "eval_runs": 20,
            "patience": 1500,
            "print_progress": True,
            "optimise_freq": 10,
            "nn_model": GeneralNet,
            "nn_type": "general_net_cycle",
            "loss_function": nn.HuberLoss(),
            "lr": 0.004,
            "nn_hidden_size": 128,
            "nn_num_lin_layers": 2,
            "optimizer_method": torch.optim.RMSprop,
            "potential_fun": MAX_LOAD_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }

    elif isinstance(graph, HyperCube) and graph.n == 4 and m == 25:
        return {
            "batch_size": 64,
            "eps_start": 0.4,
            "eps_end": 0.065,
            "eps_decay": 2900,
            "num_episodes": 2000,
            "pre_train_episodes": 75,
            "target_update_freq": 8,
            "memory_capacity": 650,
            "eval_runs": 20,
            "patience": 1500,
            "print_progress": True,
            "optimise_freq": 10,
            "nn_model": GeneralNet,
            "nn_type": "general_net_hypercube",
            "loss_function": nn.HuberLoss(),
            "lr": 0.004,
            "nn_hidden_size": 128,
            "nn_num_lin_layers": 2,
            "optimizer_method": torch.optim.RMSprop,
            "potential_fun": MAX_LOAD_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }

    elif isinstance(graph, Cycle) and graph.n == 16 and m == 50:
        return {
            "batch_size": 64,
            "eps_start": 0.4,
            "eps_end": 0.06,
            "eps_decay": 4000,
            "num_episodes": 2000,
            "pre_train_episodes": 40,
            "target_update_freq": 20,
            "memory_capacity": 900,
            "eval_runs": 10,
            "patience": 1500,
            "print_progress": True,
            "optimise_freq": 10,
            "nn_model": GeneralNet,
            "nn_type": "general_net_cycle",
            "loss_function": nn.SmoothL1Loss(),
            "lr": 0.01,
            "nn_hidden_size": 128,
            "nn_num_lin_layers": 2,
            "optimizer_method": torch.optim.Adagrad,
            "potential_fun": MAX_LOAD_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }

    elif isinstance(graph, HyperCube) and graph.n == 16 and m == 50:
        return {
            "batch_size": 64,
            "eps_start": 0.4,
            "eps_end": 0.06,
            "eps_decay": 4000,
            "num_episodes": 2000,
            "pre_train_episodes": 40,
            "target_update_freq": 20,
            "memory_capacity": 900,
            "eval_runs": 10,
            "patience": 1500,
            "print_progress": True,
            "optimise_freq": 10,
            "nn_model": GeneralNet,
            "nn_type": "general_net_cycle",
            "loss_function": nn.SmoothL1Loss(),
            "lr": 0.01,
            "nn_hidden_size": 128,
            "nn_num_lin_layers": 2,
            "optimizer_method": torch.optim.Adagrad,
            "potential_fun": MAX_LOAD_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }


    elif isinstance(graph, Cycle) and graph.n == 32 and m == 32:
        return {
            "batch_size": 56,
            "eps_start": 0.467,
            "eps_end": 0.075,
            "eps_decay": 4038,
            "num_episodes": 2500,
            "pre_train_episodes": 20,
            "target_update_freq": 22,
            "memory_capacity": 615,
            "eval_runs": 8,
            "patience": 1500,
            "print_progress": True,
            "optimise_freq": 15,
            "nn_model": GeneralNet,
            "nn_type": "general_net_cycle",
            "loss_function": nn.L1Loss(),
            "lr": 0.0044,
            "nn_hidden_size": 91,
            "nn_num_lin_layers": 1,
            "optimizer_method": torch.optim.Adam,
            "potential_fun": EXPONENTIAL_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }


    elif isinstance(graph, HyperCube) and graph.n == 32 and m == 32:
        return {
            "batch_size": 56,
            "eps_start": 0.467,
            "eps_end": 0.075,
            "eps_decay": 4038,
            "num_episodes": 2500,
            "pre_train_episodes": 20,
            "target_update_freq": 22,
            "memory_capacity": 615,
            "eval_runs": 8,
            "patience": 1500,
            "print_progress": True,
            "optimise_freq": 15,
            "nn_model": GeneralNet,
            "nn_type": "general_net_hypercube",
            "loss_function": nn.L1Loss(),
            "lr": 0.0044,
            "nn_hidden_size": 91,
            "nn_num_lin_layers": 1,
            "optimizer_method": torch.optim.Adam,
            "potential_fun": EXPONENTIAL_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }


    elif isinstance(graph, CompleteGraph) and graph.n == 4 and m == 25:
        return {
            "batch_size": 64,
            "eps_start": 0.4,
            "eps_end": 0.065,
            "eps_decay": 2900,
            "num_episodes": 2000,
            "pre_train_episodes": 75,
            "target_update_freq": 8,
            "memory_capacity": 650,
            "eval_runs": 20,
            "patience": 1500,
            "print_progress": True,
            "optimise_freq": 10,
            "nn_model": GeneralNet,
            "nn_type": "general_net_complete",
            "loss_function": nn.HuberLoss(),
            "lr": 0.004,
            "nn_hidden_size": 128,
            "nn_num_lin_layers": 2,
            "optimizer_method": torch.optim.RMSprop,
            "potential_fun": MAX_LOAD_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }



    elif isinstance(graph, CompleteGraph) and graph.n == 16 and m == 50:
        return {
            "batch_size": 64,
            "eps_start": 0.4,
            "eps_end": 0.06,
            "eps_decay": 4000,
            "num_episodes": 2000,
            "pre_train_episodes": 40,
            "target_update_freq": 20,
            "memory_capacity": 900,
            "eval_runs": 10,
            "patience": 1500,
            "print_progress": True,
            "optimise_freq": 10,
            "nn_model": GeneralNet,
            "nn_type": "general_net_cycle",
            "loss_function": nn.SmoothL1Loss(),
            "lr": 0.01,
            "nn_hidden_size": 128,
            "nn_num_lin_layers": 2,
            "optimizer_method": torch.optim.Adagrad,
            "potential_fun": MAX_LOAD_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }

    elif isinstance(graph, CompleteGraph) and graph.n == 32 and m == 32:
        return {
            "batch_size": 56,
            "eps_start": 0.467,
            "eps_end": 0.075,
            "eps_decay": 4038,
            "num_episodes": 2500,
            "pre_train_episodes": 20,
            "target_update_freq": 22,
            "memory_capacity": 615,
            "eval_runs": 8,
            "patience": 2500,
            "print_progress": True,
            "optimise_freq": 15,
            "nn_model": GeneralNet,
            "nn_type": "general_net_complete",
            "loss_function": nn.L1Loss(),
            "lr": 0.0044,
            "nn_hidden_size": 91,
            "nn_num_lin_layers": 1,
            "optimizer_method": torch.optim.Adam,
            "potential_fun": EXPONENTIAL_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }
