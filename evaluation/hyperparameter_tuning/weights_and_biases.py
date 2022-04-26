import torch
import wandb
from os.path import join, dirname, abspath
from datetime import datetime

from k_thinning.full_knowledge.RL.DQN.constants import *
from k_thinning.full_knowledge.RL.DQN.evaluate import evaluate
from k_thinning.full_knowledge.RL.DQN.neural_network import *
from k_thinning.full_knowledge.RL.DQN.train import train

N = 5
M = 25
K = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = False
NN_MODEL = GeneralNet
NN_TYPE = "general_net"
PACING_FUN = EVEN_PACING_FUN

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
    potential_mapping = {
        "max_load": MAX_LOAD_POTENTIAL,
        "std": STD_POTENTIAL,
        "exponential": EXPONENTIAL_POTENTIAL
    }
    with wandb.init(config=config):
        # TODO: add option to use graphical two choice or k-thinning
        SAVE_PATH = join((dirname(dirname(abspath(__file__)))), "training_progression",
                         f'{str(datetime.now().strftime("%Y_%m_%d %H_%M_%S_%f"))}_{N}_{M}')  # recreate for every run with fresh timestamp
        config = wandb.config
        trained_model = train(n=N, m=M, memory_capacity=config["memory_capacity"],
                              num_episodes=config["train_episodes"],
                              pre_train_episodes=config["pre_train_episodes"],
                              loss_function=loss_mapping[config["loss_function"]], lr=config["lr"],
                              reward_fun=REWARD_FUN, batch_size=config["batch_size"],
                              eps_start=config["eps_start"], pacing_fun=PACING_FUN,
                              eps_end=config["eps_end"], nn_hidden_size=config["hidden_size"],
                              nn_rnn_num_layers=config["rnn_num_layers"],
                              nn_num_lin_layers=config["num_lin_layers"],
                              eps_decay=config["eps_decay"], optimise_freq=config["optimise_freq"],
                              target_update_freq=config["target_update_freq"],
                              eval_runs=config["eval_runs_train"], patience=config["patience"],
                              potential_fun=potential_mapping[config["potential_fun"]], max_threshold=config["max_threshold"],
                              eval_parallel_batch_size=config["eval_parallel_batch_size"],
                              print_progress=PRINT_PROGRESS, use_normalised=config["use_normalised"],
                              optimizer_method=optimizer_mapping[config["optimizer_method"]],
                              nn_model=NN_MODEL, device=DEVICE, report_wandb=True, save_path=SAVE_PATH)
        score = evaluate(trained_model, n=N, m=M, reward_fun=REWARD_FUN, eval_runs_eval=config["eval_runs_eval"],
                         max_threshold=config["max_threshold"], use_normalised=config["use_normalised"])
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
            "values": [200]
        },
        "patience": {
            "values": [300]
        },
        "eval_runs_eval": {
            "values": [25] # 30
        },
        "eval_parallel_batch_size": {
            "values": [32]
        },
        "pre_train_episodes": {
            'distribution': 'int_uniform',
            'min': 1,
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
            'min': 2000, # 100
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
            'max': 1000 # 600
        },
        "eval_runs_train": {
            'distribution': 'int_uniform',
            'min': 2,
            'max': 6  # TODO: increase?
        },
        "optimise_freq": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 50  # 30
        },
        "max_threshold": {  # TODO: always set independently for new N,M
            'distribution': 'int_uniform',
            'min': 4,
            'max': 7
        },
        "loss_function": {
            "values": ["MSELoss", "HuberLoss", "L1Loss", "SmoothL1Loss"]
        },
        "optimizer_method": {
            "values": ["Adam", "Adagrad", "SGD", "RMSprop"]
        },
        "lr": {
            "distribution": "log_uniform",
            "min": -6,
            "max": -4
        },
        "hidden_size": {
            'distribution': 'int_uniform',
            'min': 64,
            'max': 256
        },
        "rnn_num_layers": {
            'distribution': 'int_uniform',
            'min': 1,  # 2
            'max': 3
        },
        "num_lin_layers": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 3
        },
        "potential_fun": {
            "values": ["max_load", "std", "exponential"]
        },
        "use_normalised": {
            "values": [True, False]
        }
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project=f"k_thinning_{N}_{M}_{K}_final")
    wandb.agent(sweep_id, tuning_function, count=500)
