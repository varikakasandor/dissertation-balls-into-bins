import os
from datetime import datetime

import ray
import torch
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch

from two_thinning.full_knowledge.RL.DQN.evaluate import evaluate
from two_thinning.full_knowledge.RL.DQN.neural_network import FullTwoThinningRecurrentNetFC
from two_thinning.full_knowledge.RL.DQN.train import train

N = 5
M = 30
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


def tuning_function(config):
    trained_model = train(n=N, m=M, memory_capacity=config["memory_capacity"], num_episodes=config["train_episodes"],
                  reward_fun=REWARD_FUN, batch_size=config["batch_size"], eps_start=config["eps_start"], eps_end=config["eps_end"],
                  eps_decay=config["eps_decay"], optimise_freq=config["optimise_freq"], target_update_freq=config["target_update_freq"],
                  eval_runs=config["eval_runs_train"], patience=config["patience"], potential_fun=POTENTIAL_FUN, max_threshold=config["max_threshold"],
                  eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE, print_progress=PRINT_PROGRESS, nn_model=NN_MODEL, device=DEVICE)
    score = evaluate(trained_model, n=N, m=M, reward_fun=REWARD_FUN, eval_runs_eval=EVAL_RUNS_EVAL, eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE)
    tune.report(score=score)


def analyse_hyperparameters():
    ray.init(object_store_memory=78643200)
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
        resources_per_trial={"cpu": 8, "gpu": 1},
        metric="score",
        mode="max",
        #"""# Limit to two concurrent trials (otherwise we end up with random search)
        #search_alg=ConcurrencyLimiter(
        #    BayesOptSearch(random_search_steps=4),
        #    max_concurrent=2),"""
        num_samples=1000,
        #stop={"training_iteration": 3},
        verbose=2)

    print("Best config: ", analysis.get_best_config(
        metric="score", mode="max"))
    df = analysis.results_df
    print(df)

    time_stamp = str(datetime.now().strftime("%Y_%m_%d %H_%M_%S_%f"))
    df_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comparison_dataframes", f"{time_stamp}.csv")
    df.to_csv(df_save_path)

    ray.shutdown()


if __name__ == "__main__":
    analyse_hyperparameters()