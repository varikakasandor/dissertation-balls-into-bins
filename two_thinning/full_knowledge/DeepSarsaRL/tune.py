from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import numpy as np
import torch

from two_thinning.full_knowledge.DeepSarsaRL.train import train

n = 10
m = n

train_episodes = 300


def reward(x):
    return -np.max(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_config(n=n, m=m, reward=reward, config):
    train(n=config['n'], m=config['m'], epsilon=config['epsilon'], reward=reward, episodes=train_episodes, device=device)

def tune_hyperparameters(train_episodes=train_episodes):
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=train_episodes,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")