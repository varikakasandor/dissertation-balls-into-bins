import wandb

from k_choice.graphical.two_choice.graphs.cycle import Cycle
from k_choice.graphical.two_choice.graphs.complete_graph import CompleteGraph
from k_choice.graphical.two_choice.graphs.random_regular_graph import RandomRegularGraph
from k_choice.graphical.two_choice.graphs.hypercube import HyperCube
from k_choice.graphical.two_choice.full_knowledge.RL.DQN.constants import *
from k_choice.graphical.two_choice.full_knowledge.RL.DQN.evaluate import evaluate
from k_choice.graphical.two_choice.full_knowledge.RL.DQN.neural_network import *
from k_choice.graphical.two_choice.full_knowledge.RL.DQN.train import train

N = 32
GRAPH = CompleteGraph(N)
M = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = False
NN_MODEL = GeneralNet
NN_TYPE = "general_net_complete"
PACING_FUN = EVEN_PACING_FUN


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
        "exponential": EXPONENTIAL_POTENTIAL,
        "edge": POTENTIAL_FUN_WORST_EDGE,
        "neighbour": POTENTIAL_FUN_NEIGHBOUR_AVG
    }
    with wandb.init(config=config):
        config = wandb.config
        trained_model = train(graph=GRAPH, m=M, memory_capacity=config["memory_capacity"],
                              num_episodes=config["train_episodes"],
                              pre_train_episodes=config["pre_train_episodes"],
                              loss_function=loss_mapping[config["loss_function"]], lr=config["lr"],
                              reward_fun=REWARD_FUN, batch_size=config["batch_size"],
                              eps_start=config["eps_start"], pacing_fun=PACING_FUN,
                              eps_end=config["eps_end"], nn_hidden_size=config["hidden_size"],
                              nn_num_lin_layers=config["num_lin_layers"],
                              eps_decay=config["eps_decay"], optimise_freq=config["optimise_freq"],
                              target_update_freq=config["target_update_freq"],
                              eval_runs=config["eval_runs_train"], patience=config["patience"],
                              potential_fun=potential_mapping[config["potential_fun"]],
                              print_progress=PRINT_PROGRESS,
                              optimizer_method=optimizer_mapping[config["optimizer_method"]],
                              nn_model=NN_MODEL, device=DEVICE, report_wandb=True)
        score = evaluate(trained_model, graph=GRAPH, m=M, reward_fun=REWARD_FUN,
                         eval_runs_eval=config["eval_runs_eval"])
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
            "values": [25]  # 30
        },
        "eval_runs_train": {
            "values": [8]
        },
        "pre_train_episodes": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 100
        },
        'batch_size': {
            'distribution': 'int_uniform',
            'min': 16,
            'max': 70
        },
        "eps_start": {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 0.6
        },
        "eps_end": {
            'distribution': 'uniform',
            'min': 0.03,
            'max': 0.1
        },
        "eps_decay": {
            'distribution': 'uniform',
            'min': 2000,  # 100
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
            'max': 1000  # 600
        },
        "optimise_freq": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 50  # 30
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
        "num_lin_layers": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 5
        },
        "potential_fun": {
            "values": ["max_load", "std", "exponential", "edge", "neighbour"]
        }
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project=f"graphical_two_choice_{N}_{M}_complete_final")
    wandb.agent(sweep_id, tuning_function, count=500)
