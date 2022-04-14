import torch
import wandb
from os.path import join, dirname, abspath
from datetime import datetime

from two_thinning.full_knowledge.RL.DQN.evaluate import evaluate
from two_thinning.full_knowledge.RL.DQN.neural_network import *
from two_thinning.full_knowledge.RL.DQN.train import train as two_thinning_train

N = 5
M = 13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = False
NN_MODEL = GeneralNet
NN_TYPE = "general_net"


def POTENTIAL_FUN(loads):
    return -max(loads)  # TODO: take into account more bins
    # return -std(loads)


def REWARD_FUN(loads, error_ratio=1.5):
    return -max(loads)
    # return -std(loads)
    # return 1 if max(loads) < error_ratio * sum(loads) / len(loads) else 0


def tuning_function(config=None):
    loss_mapping = {
        "SmoothL1Loss": nn.SmoothL1Loss(),
        "MSELoss": nn.MSELoss(),
        "HuberLoss": nn.HuberLoss(),
        "L1Loss": nn.L1Loss()
    }
    optimizer_mapping = {
        "Adam": torch.optim.Adam,
        "Adagrad": torch.optim.Adagrad,
        "SGD": torch.optim.SGD,
        "RMSprop": torch.optim.RMSprop
    }
    with wandb.init(config=config):
        # TODO: add option to use graphical two choice or k-thinning
        SAVE_PATH = join((dirname(dirname(abspath(__file__)))), "training_progression",
                         f'{str(datetime.now().strftime("%Y_%m_%d %H_%M_%S_%f"))}_{N}_{M}')  # recreate for every run with fresh timestamp
        config = wandb.config
        trained_model = two_thinning_train(n=N, m=M, memory_capacity=config["memory_capacity"],
                                           num_episodes=config["train_episodes"],
                                           pre_train_episodes=config["pre_train_episodes"],
                                           loss_function=loss_mapping[config["loss_function"]], lr=config["lr"],
                                           reward_fun=REWARD_FUN, batch_size=config["batch_size"],
                                           eps_start=config["eps_start"],
                                           eps_end=config["eps_end"], nn_hidden_size=config["hidden_size"],
                                           nn_rnn_num_layers=config["rnn_num_layers"],
                                           nn_num_lin_layers=config["num_lin_layers"],
                                           eps_decay=config["eps_decay"], optimise_freq=config["optimise_freq"],
                                           target_update_freq=config["target_update_freq"],
                                           eval_runs=config["eval_runs_train"], patience=config["patience"],
                                           potential_fun=POTENTIAL_FUN, max_threshold=config["max_threshold"],
                                           eval_parallel_batch_size=config["eval_parallel_batch_size"],
                                           print_progress=PRINT_PROGRESS, use_normalised=config["use_normalised"],
                                           optimizer_method=optimizer_mapping[config["optimizer_method"]],
                                           nn_model=NN_MODEL, device=DEVICE, report_wandb=True, save_path=SAVE_PATH)
        score = evaluate(trained_model, n=N, m=M, reward_fun=REWARD_FUN, eval_runs_eval=config["eval_runs_eval"],
                         eval_parallel_batch_size=config["eval_parallel_batch_size"])
        wandb.log({"score": score})


if __name__ == "__main__":
    wandb.login()

    sweep_config = {
        'method': 'bayes'
    }
    metric = {
        'name': 'score',
        'goal': 'maximize'
    }
    sweep_config['metric'] = metric
    parameters_dict = {
        "train_episodes": {
            "values": [100]
        },
        "patience": {
            "values": [100]
        },
        "eval_runs_eval": {
            "values": [100]
        },
        "eval_parallel_batch_size": {
            "values": [32]
        },
        "pre_train_episodes": {
            'distribution': 'int_uniform',
            'min': 0,
            'max': 100
        },
        'batch_size': {
            'distribution': 'int_uniform',
            'min': 16,
            'max': 64
        },
        "eps_start": {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 0.5
        },
        "eps_end": {
            'distribution': 'uniform',
            'min': 0.03,
            'max': 0.1
        },
        "eps_decay": {
            'distribution': 'uniform',
            'min': 100,
            'max': 5000
        },
        "target_update_freq": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 30
        },
        "memory_capacity": {
            'distribution': 'int_uniform',
            'min': 300,
            'max': 1000
        },
        "eval_runs_train": {
            'distribution': 'int_uniform',
            'min': 3,
            'max': 10  # TODO: increase?
        },
        "optimise_freq": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 50
        },
        "max_threshold": { # TODO: always set independently for new N,M
            'distribution': 'int_uniform',
            'min': 2,
            'max': 5
        },
        "loss_function": {
            "values": ["SmoothL1Loss", "MSELoss", "HuberLoss", "L1Loss"]
        },
        "optimizer_method": {
            "values": ["Adam", "Adagrad", "SGD", "RMSprop"]
        },
        "lr": {
            "distribution": "log_uniform",
            "min": -8,
            "max": -4
        },
        "hidden_size": {
            'distribution': 'int_uniform',
            'min': 16,
            'max': 256
        },
        "rnn_num_layers": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 2
        },
        "num_lin_layers": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 2
        },
        "use_normalised": {
            "values": [True, False]
        }
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project=f"two_thinning_{N}_{M}")
    wandb.agent(sweep_id, tuning_function, count=200)
