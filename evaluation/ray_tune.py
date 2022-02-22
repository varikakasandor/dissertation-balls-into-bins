import torch
from ray import tune

from two_thinning.full_knowledge.RL.DQN.evaluate import evaluate
from two_thinning.full_knowledge.RL.DQN.neural_network import FullTwoThinningRecurrentNetFC
from two_thinning.full_knowledge.RL.DQN.train import train

N = 3
M = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_RUNS_EVAL = 100
EVAL_PARALLEL_BATCH_SIZE = 32
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = True
NN_MODEL = FullTwoThinningRecurrentNetFC
NN_TYPE = "rnn_fc"


def POTENTIAL_FUN(loads):
    return -max(loads)  # TODO: take into account more bins
    #return -std(loads)

def REWARD_FUN(loads, error_ratio=1.5):
    return -max(loads)
    #return -std(loads)
    #return 1 if max(loads) < error_ratio * sum(loads) / len(loads) else 0


"""
Tuneable hyperparameters:

BATCH_SIZE = 64
EPS_START = 0.2
EPS_END = 0.05
EPS_DECAY = 2000
TRAIN_EPISODES = 3000
TARGET_UPDATE_FREQ = 10
MEMORY_CAPACITY = 10 * BATCH_SIZE
EVAL_RUNS_TRAIN = 10
PATIENCE = 1000
OPTIMISE_FREQ = int(sqrt(M))  # TODO: completely ad-hoc
MAX_THRESHOLD = max(3, 2 * (M + N - 1) // N)
"""

def tuning_function(config):
    trained_model = train(n=N, m=M, memory_capacity=config["memory_capacity"], num_episodes=config["train_episodes"],
                  reward_fun=REWARD_FUN, batch_size=config["batch_size"], eps_start=config["eps_start"], eps_end=config["eps_end"],
                  eps_decay=config["eps_decay"], optimise_freq=config["optimise_freq"], target_update_freq=config["target_update_freq"],
                  eval_runs=config["eval_runs_train"], patience=config["patience"], potential_fun=POTENTIAL_FUN, max_threshold=config["max_threshold"],
                  eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE, print_progress=PRINT_PROGRESS, nn_model=NN_MODEL, device=DEVICE)
    score = evaluate(trained_model, n=N, m=M, reward_fun=REWARD_FUN, eval_runs_eval=EVAL_RUNS_EVAL, eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE)
    tune.report(score=score)


def analyse_hyperparameters():
    config = {
        "batch_size": tune.randint(16, 64),
        "eps_start": tune.uniform(0.05, 0.5),
        "eps_end": tune.uniform(0.01, 0.1),
        "eps_decay": tune.uniform(100, 10000),
        "train_episodes": tune.randint(300, 3000),
        "target_update_freq": tune.randint(1, 30),
        "memory_capacity": tune.randint(16, 1000),
        "eval_runs_train": tune.randint(1, 30),
        "patience": tune.randint(30, 1000),
        "optimise_freq": tune.randint(1, M),
        "max_threshold": tune.randint(3, M),

    }
    analysis = tune.run(
        tuning_function,
        config=config,
        resources_per_trial={"cpu": 8, "gpu": 1},fail_fast="raise")

    print("Best config: ", analysis.get_best_config(
        metric="score", mode="min"))

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df

    print(df)


if __name__ == "__main__":
    analyse_hyperparameters()