from math import sqrt, log, ceil, exp
from os.path import join, dirname, abspath
from datetime import datetime

import torch.optim

from two_thinning.full_knowledge.RL.DQN.neural_network import *
from two_thinning.full_knowledge.RL.DQN.constants import *
from helper.helper import std


def get_dqn_hyperparameters(n, m):
    if n == 20 and m == 60:
        return {
            "batch_size": 64,
            "eps_start": 0.35,
            "eps_end": 0.04,
            "eps_decay": 2800,
            "num_episodes": 1000,  # TODO
            "pre_train_episodes": 15,  # TODO
            "target_update_freq": 4,
            "memory_capacity": 600,
            "eval_runs": 5,
            "eval_parallel_batch_size": 64,
            "patience": 1000,
            "use_normalised": True,
            "print_progress": True,
            "optimise_freq": 40,
            "max_threshold": 4,
            "nn_model": GeneralNet,
            "nn_type": "general_net",
            "loss_function": nn.SmoothL1Loss(),
            "lr": 0.0065,
            "nn_hidden_size": 256,
            "nn_rnn_num_layers": 3,
            "nn_num_lin_layers": 1,
            "optimizer_method": torch.optim.SGD,
            "potential_fun": MAX_LOAD_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }

    elif n == 50 and m == 200:
        return {
            "batch_size": 32,
            "eps_start": 0.25,
            "eps_end": 0.07,
            "eps_decay": 3500,
            "num_episodes": 1000,  # TODO
            "pre_train_episodes": 65,  # TODO
            "target_update_freq": 8,
            "memory_capacity": 500,
            "eval_runs": 6,
            "eval_parallel_batch_size": 64,
            "patience": 1000,
            "use_normalised": True,
            "print_progress": True,
            "optimise_freq": 30,
            "max_threshold": 8,
            "nn_model": GeneralNet,
            "nn_type": "general_net",
            "loss_function": nn.SmoothL1Loss(),
            "lr": 0.0045,
            "nn_hidden_size": 200,
            "nn_rnn_num_layers": 3,
            "nn_num_lin_layers": 1,
            "optimizer_method": torch.optim.Adam,
            "potential_fun": MAX_LOAD_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }


    elif n == 50 and m == 50:
        return {
            "batch_size": 32,
            "eps_start": 0.25,
            "eps_end": 0.035,
            "eps_decay": 4200,
            "num_episodes": 1000,  # TODO
            "pre_train_episodes": 5,  # TODO
            "target_update_freq": 25,
            "memory_capacity": 750,
            "eval_runs": 5,
            "eval_parallel_batch_size": 64,
            "patience": 1000,
            "use_normalised": True,
            "print_progress": True,
            "optimise_freq": 35,
            "max_threshold": 3,
            "nn_model": GeneralNet,
            "nn_type": "general_net",
            "loss_function": nn.MSELoss(),
            "lr": 0.006,
            "nn_hidden_size": 200,
            "nn_rnn_num_layers": 3,
            "nn_num_lin_layers": 2,
            "optimizer_method": torch.optim.SGD,
            "potential_fun": STD_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }

    elif n == 20 and m == 400:
        return {
            "batch_size": 32,
            "eps_start": 0.2,
            "eps_end": 0.05,
            "eps_decay": 3500,
            "num_episodes": 1000,  # TODO
            "pre_train_episodes": 50,  # TODO
            "target_update_freq": 25,
            "memory_capacity": 450,
            "eval_runs": 5,
            "eval_parallel_batch_size": 64,
            "patience": 1000,
            "use_normalised": True,
            "print_progress": True,
            "optimise_freq": 25,
            "max_threshold": 22,
            "nn_model": GeneralNet,
            "nn_type": "general_net",
            "loss_function": nn.MSELoss(),
            "lr": 0.005,
            "nn_hidden_size": 128,
            "nn_rnn_num_layers": 3,
            "nn_num_lin_layers": 2,
            "optimizer_method": torch.optim.Adam,
            "potential_fun": MAX_LOAD_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }


def get_threshold_hyperparameters(n, m):
    d = {
        "episodes": 10000,
        "epsilon": 0.1,
        "primary_only": True,
        "initial_q_value": 0
    }
    return d
