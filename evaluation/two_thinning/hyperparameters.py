from math import sqrt, log, ceil, exp
from os.path import join, dirname, abspath
from datetime import datetime

from two_thinning.full_knowledge.RL.DQN.neural_network import *
from helper.helper import std


def get_dqn_hyperparameters(n, m):
    def MAX_LOAD_POTENTIAL(loads):
        return -max(loads)

    def EVEN_PACING_FUN(start_size, n, m, all_episodes):
        delta = max((all_episodes - n * m) * 2 // (m * (m - 1)), 0)
        return n + ((m - 1) - start_size) * delta

    d = {
        "batch_size": 64,
        "eps_start": 0.2,
        "eps_end": 0.04,
        "eps_decay": 4200,
        "num_episodes": 2, # TODO
        "pre_train_episodes": 3, # TODO
        "target_update_freq": 20,
        "memory_capacity": 800,
        "eval_runs": 1,
        "eval_parallel_batch_size": 64,
        "patience": 100,
        "use_normalised": True,
        "print_progress": True,
        "optimise_freq": 3 * int(sqrt(m)),
        "max_threshold": max(3, m // n + ceil(sqrt(log(n)))),
        "nn_model": GeneralNet,
        "nn_type": "general_net",
        "loss_function": nn.SmoothL1Loss(),
        "lr": 0.0004,
        "nn_hidden_size": 64,
        "nn_rnn_num_layers": 2,
        "nn_num_lin_layers": 2,
        "optimizer_method": torch.optim.Adam,
        "potential_fun": MAX_LOAD_POTENTIAL,
        "pacing_fun": EVEN_PACING_FUN
    }

    return d


def get_threshold_hyperparameters(n, m):
    d = {
        "episodes": 10000,
        "epsilon": 0.1,
        "primary_only": True,
        "initial_q_value": 0
    }
    return d
