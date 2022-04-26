import torch.optim

from k_thinning.full_knowledge.RL.DQN.constants import *


def get_dqn_hyperparameters(n, m, k):
    if n == 5 and m == 25 and k == 3:
        return {
            "batch_size": 32,
            "eps_start": 0.21,
            "eps_end": 0.035,
            "eps_decay": 3000,
            "num_episodes": 1000,  # TODO
            "pre_train_episodes": 60,  # TODO
            "target_update_freq": 18,
            "memory_capacity": 650,
            "eval_runs": 10,
            "eval_parallel_batch_size": 64,
            "patience": 1000,
            "use_normalised": True,
            "print_progress": True,
            "optimise_freq": 10,
            "max_threshold": 7,
            "nn_model": GeneralNet,
            "nn_type": "general_net",
            "loss_function": nn.MSELoss(),
            "lr": 0.004,
            "nn_hidden_size": 128,
            "nn_rnn_num_layers": 2,
            "nn_num_lin_layers": 3,
            "optimizer_method": torch.optim.SGD,
            "potential_fun": STD_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }

    elif n == 5 and m == 25 and k == 5:
        return {
            "batch_size": 32,
            "eps_start": 0.15,
            "eps_end": 0.06,
            "eps_decay": 3000,
            "num_episodes": 1000,  # TODO
            "pre_train_episodes": 20,  # TODO
            "target_update_freq": 20,
            "memory_capacity": 650,
            "eval_runs": 10,
            "eval_parallel_batch_size": 64,
            "patience": 1000,
            "use_normalised": True,
            "print_progress": True,
            "optimise_freq": 10,
            "max_threshold": 6,
            "nn_model": GeneralNet,
            "nn_type": "general_net",
            "loss_function": nn.SmoothL1Loss(),
            "lr": 0.005,
            "nn_hidden_size": 128,
            "nn_rnn_num_layers": 2,
            "nn_num_lin_layers": 3,
            "optimizer_method": torch.optim.Adam,
            "potential_fun": EXPONENTIAL_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }

    elif n == 5 and m == 25 and k == 10:
        return {
            "batch_size": 32,
            "eps_start": 0.26,
            "eps_end": 0.09,
            "eps_decay": 2500,
            "num_episodes": 1000,  # TODO
            "pre_train_episodes": 20,  # TODO
            "target_update_freq": 18,
            "memory_capacity": 650,
            "eval_runs": 10,
            "eval_parallel_batch_size": 64,
            "patience": 1000,
            "use_normalised": True,
            "print_progress": True,
            "optimise_freq": 30,
            "max_threshold": 6,
            "nn_model": GeneralNet,
            "nn_type": "general_net",
            "loss_function": nn.SmoothL1Loss(),
            "lr": 0.005,
            "nn_hidden_size": 128,
            "nn_rnn_num_layers": 2,
            "nn_num_lin_layers": 3,
            "optimizer_method": torch.optim.Adam,
            "potential_fun": STD_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }

    elif n == 20 and m == 50 and k == 3:
        return {
            "batch_size": 32,
            "eps_start": 0.4,
            "eps_end": 0.055,
            "eps_decay": 3200,
            "num_episodes": 1000,  # TODO
            "pre_train_episodes": 40,  # TODO
            "target_update_freq": 28,
            "memory_capacity": 650,
            "eval_runs": 10,
            "eval_parallel_batch_size": 64,
            "patience": 1000,
            "use_normalised": True,
            "print_progress": True,
            "optimise_freq": 30,
            "max_threshold": 3,
            "nn_model": GeneralNet,
            "nn_type": "general_net",
            "loss_function": nn.SmoothL1Loss(),
            "lr": 0.005,
            "nn_hidden_size": 128,
            "nn_rnn_num_layers": 2,
            "nn_num_lin_layers": 3,
            "optimizer_method": torch.optim.Adam,
            "potential_fun": EXPONENTIAL_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }

    elif n == 20 and m == 50 and k == 5:
        return {
            "batch_size": 32,
            "eps_start": 0.2,
            "eps_end": 0.06,
            "eps_decay": 4300,
            "num_episodes": 1000,  # TODO
            "pre_train_episodes": 40,  # TODO
            "target_update_freq": 24,
            "memory_capacity": 650,
            "eval_runs": 10,
            "eval_parallel_batch_size": 64,
            "patience": 1000,
            "use_normalised": True,
            "print_progress": True,
            "optimise_freq": 40,
            "max_threshold": 3,
            "nn_model": GeneralNet,
            "nn_type": "general_net",
            "loss_function": nn.SmoothL1Loss(),
            "lr": 0.005,
            "nn_hidden_size": 200,
            "nn_rnn_num_layers": 3,
            "nn_num_lin_layers": 3,
            "optimizer_method": torch.optim.Adam,
            "potential_fun": STD_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }

    elif n == 20 and m == 50 and k == 10:
        return {
            "batch_size": 64,
            "eps_start": 0.2,
            "eps_end": 0.06,
            "eps_decay": 3000,
            "num_episodes": 1000,  # TODO
            "pre_train_episodes": 10,  # TODO
            "target_update_freq": 10,
            "memory_capacity": 650,
            "eval_runs": 10,
            "eval_parallel_batch_size": 64,
            "patience": 1000,
            "use_normalised": True,
            "print_progress": True,
            "optimise_freq": 15,
            "max_threshold": 4,
            "nn_model": GeneralNet,
            "nn_type": "general_net",
            "loss_function": nn.SmoothL1Loss(),
            "lr": 0.005,
            "nn_hidden_size": 128,
            "nn_rnn_num_layers": 2,
            "nn_num_lin_layers": 3,
            "optimizer_method": torch.optim.Adam,
            "potential_fun": EXPONENTIAL_POTENTIAL,
            "pacing_fun": EVEN_PACING_FUN
        }



def get_threshold_hyperparameters(n, m, k):
    d = {
        "episodes": 10000,
        "epsilon": 0.1,
        "initial_q_value": 0
    }
    return d
