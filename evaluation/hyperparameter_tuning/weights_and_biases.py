import torch
import wandb

from two_thinning.full_knowledge.RL.DQN.evaluate import evaluate
from two_thinning.full_knowledge.RL.DQN.neural_network import FullTwoThinningClippedRecurrentNetFC
from two_thinning.full_knowledge.RL.DQN.train import train

N = 10
M = 100  # So that max load of 5 is achievable with higher probability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_EPISODES = 1000
EVAL_RUNS_EVAL = 100
EVAL_PARALLEL_BATCH_SIZE = 32
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = False
NN_MODEL = FullTwoThinningClippedRecurrentNetFC
NN_TYPE = "rnn_clipped_fc"


def POTENTIAL_FUN(loads):
    return -max(loads)  # TODO: take into account more bins
    # return -std(loads)


def REWARD_FUN(loads, error_ratio=1.5):
    return -max(loads)
    # return -std(loads)
    # return 1 if max(loads) < error_ratio * sum(loads) / len(loads) else 0


def tuning_function(config=None):
    with wandb.init(config=config):
        config = wandb.config
        trained_model = train(n=N, m=M, memory_capacity=config["memory_capacity"],
                              num_episodes=TRAIN_EPISODES,
                              reward_fun=REWARD_FUN, batch_size=config["batch_size"], eps_start=config["eps_start"],
                              eps_end=config["eps_end"],
                              eps_decay=config["eps_decay"], optimise_freq=config["optimise_freq"],
                              target_update_freq=config["target_update_freq"],
                              eval_runs=config["eval_runs_train"], patience=config["patience"],
                              potential_fun=POTENTIAL_FUN, max_threshold=config["max_threshold"],
                              eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE, print_progress=PRINT_PROGRESS,
                              nn_model=NN_MODEL, device=DEVICE, report_wandb=True)
        score = evaluate(trained_model, n=N, m=M, reward_fun=REWARD_FUN, eval_runs_eval=EVAL_RUNS_EVAL,
                         eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE)
        wandb.log({"score": score})


if __name__ == "__main__":
    print(wandb)
    wandb.login()

    sweep_config = {
        'method': 'random'  # TODO: set back to bayes
    }
    metric = {
        'name': 'score',
        'goal': 'maximize'
    }
    sweep_config['metric'] = metric
    parameters_dict = {
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
            'min': 0.01,
            'max': 0.1
        },
        "eps_decay": {
            'distribution': 'uniform',
            'min': 100,
            'max': 10000
        },
        "target_update_freq": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 30
        },
        "memory_capacity": {
            'distribution': 'int_uniform',
            'min': 16,
            'max': 1000
        },
        "eval_runs_train": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 30
        },
        "patience": {
            'distribution': 'int_uniform',
            'min': 30,
            'max': 1000,
        },
        "optimise_freq": {
            'distribution': 'int_uniform',
            'min': 1,
            'max': M
        },
        "max_threshold": {
            'distribution': 'int_uniform',
            'min': 3,
            'max': M
        },
    }

    sweep_config['parameters'] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project=f"two_thinning_{N}_{M}")

    wandb.agent(sweep_id, tuning_function, count=100)
