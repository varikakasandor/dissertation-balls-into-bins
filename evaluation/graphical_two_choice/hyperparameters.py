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
            "batch_size": 32,
            "eps_start": 0.4,
            "eps_end": 0.06,
            "eps_decay": 2600,
            "num_episodes": 1000,  # TODO
            "pre_train_episodes": 30,  # TODO
            "target_update_freq": 25,
            "memory_capacity": 650,
            "eval_runs": 10,
            "patience": 1000,
            "print_progress": True,
            "optimise_freq": 15,
            "nn_model": GeneralNet,
            "nn_type": "general_net_cycle",
            "loss_function": nn.L1Loss(),
            "lr": 0.015,
            "nn_hidden_size": 128,
            "nn_num_lin_layers": 3,
            "optimizer_method": torch.optim.Adam,
            "potential_fun": MAX_LOAD_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }

    elif isinstance(graph, HyperCube) and graph.n == 4 and m == 25:
        return {
            "batch_size": 64,
            "eps_start": 0.4,
            "eps_end": 0.065,
            "eps_decay": 2900,
            "num_episodes": 1000,
            "pre_train_episodes": 75,
            "target_update_freq": 8,
            "memory_capacity": 650,
            "eval_runs": 10,
            "patience": 1000,
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
            "num_episodes": 1000,
            "pre_train_episodes": 40,
            "target_update_freq": 20,
            "memory_capacity": 900,
            "eval_runs": 10,
            "patience": 1000,
            "print_progress": True,
            "optimise_freq": 10,
            "nn_model": GeneralNet,
            "nn_type": "general_net_cycle",
            "loss_function": nn.SmoothL1Loss(),
            "lr": 0.01,
            "nn_hidden_size": 128,
            "nn_num_lin_layers": 2,
            "optimizer_method": torch.optim.Adagrad,
            "potential_fun": EXPONENTIAL_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }

    elif isinstance(graph, HyperCube) and graph.n == 16 and m == 50:
        return {
            "batch_size": 64,
            "eps_start": 0.4,
            "eps_end": 0.06,
            "eps_decay": 4000,
            "num_episodes": 1000,
            "pre_train_episodes": 40,
            "target_update_freq": 20,
            "memory_capacity": 650,
            "eval_runs": 10,
            "patience": 1000,
            "print_progress": True,
            "optimise_freq": 10,
            "nn_model": GeneralNet,
            "nn_type": "general_net_hypercube",
            "loss_function": nn.MSELoss(),
            "lr": 0.0045,
            "nn_hidden_size": 256,
            "nn_num_lin_layers": 2,
            "optimizer_method": torch.optim.RMSprop,
            "potential_fun": EXPONENTIAL_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }


    elif isinstance(graph, Cycle) and graph.n == 32 and m == 32:
        return {
            "batch_size": 32,
            "eps_start": 0.4,
            "eps_end": 0.05,
            "eps_decay": 4000,
            "num_episodes": 1000,
            "pre_train_episodes": 50,
            "target_update_freq": 25,
            "memory_capacity": 650,
            "eval_runs": 10,
            "patience": 1000,
            "print_progress": True,
            "optimise_freq": 20,
            "nn_model": GeneralNet,
            "nn_type": "general_net_cycle",
            "loss_function": nn.HuberLoss(),
            "lr": 0.01,
            "nn_hidden_size": 256,
            "nn_num_lin_layers": 3,
            "optimizer_method": torch.optim.RMSprop,
            "potential_fun": POTENTIAL_FUN_NEIGHBOUR_AVG,
            "pacing_fun": EVEN_PACING_FUN
        }


    elif isinstance(graph, HyperCube) and graph.n == 32 and m == 32:
        return {
            "batch_size": 64,
            "eps_start": 0.48,
            "eps_end": 0.08,
            "eps_decay": 4000,
            "num_episodes": 1000,
            "pre_train_episodes": 30,
            "target_update_freq": 20,
            "memory_capacity": 650,
            "eval_runs": 10,
            "patience": 1000,
            "print_progress": True,
            "optimise_freq": 15,
            "nn_model": GeneralNet,
            "nn_type": "general_net_hypercube",
            "loss_function": nn.L1Loss(),
            "lr": 0.005,
            "nn_hidden_size": 128,
            "nn_num_lin_layers": 2,
            "optimizer_method": torch.optim.Adam,
            "potential_fun": EXPONENTIAL_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }


    elif isinstance(graph, CompleteGraph) and graph.n == 4 and m == 25:
        return {
            "batch_size": 32,
            "eps_start": 0.3,
            "eps_end": 0.08,
            "eps_decay": 3000,
            "num_episodes": 1000,
            "pre_train_episodes": 30,
            "target_update_freq": 15,
            "memory_capacity": 650,
            "eval_runs": 10,
            "patience": 1000,
            "print_progress": True,
            "optimise_freq": 20,
            "nn_model": GeneralNet,
            "nn_type": "general_net_complete",
            "loss_function": nn.SmoothL1Loss(),
            "lr": 0.005,
            "nn_hidden_size": 256,
            "nn_num_lin_layers": 3,
            "optimizer_method": torch.optim.Adam,
            "potential_fun": POTENTIAL_FUN_WORST_EDGE,  # TODO: double check
            "pacing_fun": EVEN_PACING_FUN
        }



    elif isinstance(graph, CompleteGraph) and graph.n == 16 and m == 50:
        return {
            "batch_size": 64,
            "eps_start": 0.4,
            "eps_end": 0.08,
            "eps_decay": 4500,
            "num_episodes": 1000,
            "pre_train_episodes": 65,
            "target_update_freq": 30,
            "memory_capacity": 900,
            "eval_runs": 10,
            "patience": 1000,
            "print_progress": True,
            "optimise_freq": 25,
            "nn_model": GeneralNet,
            "nn_type": "general_net_complete",
            "loss_function": nn.SmoothL1Loss(),
            "lr": 0.013,
            "nn_hidden_size": 128,
            "nn_num_lin_layers": 2,
            "optimizer_method": torch.optim.Adam,
            "potential_fun": EXPONENTIAL_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }

    elif isinstance(graph, CompleteGraph) and graph.n == 32 and m == 32:
        return {
            "batch_size": 46,
            "eps_start": 0.3,
            "eps_end": 0.06,
            "eps_decay": 3400,
            "num_episodes": 1000,
            "pre_train_episodes": 65,
            "target_update_freq": 25,
            "memory_capacity": 550,
            "eval_runs": 10,
            "patience": 1000,
            "print_progress": True,
            "optimise_freq": 35,
            "nn_model": GeneralNet,
            "nn_type": "general_net_complete",
            "loss_function": nn.HuberLoss(),
            "lr": 0.005,
            "nn_hidden_size": 160,
            "nn_num_lin_layers": 3,
            "optimizer_method": torch.optim.RMSprop,
            "potential_fun": POTENTIAL_FUN_NEIGHBOUR_AVG,
            "pacing_fun": EVEN_PACING_FUN
        }
