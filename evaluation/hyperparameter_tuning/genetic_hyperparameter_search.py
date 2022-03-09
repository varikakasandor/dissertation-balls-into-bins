import functools
import os
from datetime import datetime
import torch
import random
import json

from two_thinning.full_knowledge.RL.DQN.evaluate import evaluate
from two_thinning.full_knowledge.RL.DQN.neural_network import FullTwoThinningClippedRecurrentNetFC
from two_thinning.full_knowledge.RL.DQN.train import train

N = 5
M = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


#############################################


@functools.lru_cache(maxsize=None)
def objective(config_):
    config = {k: v for (k, v) in config_}
    trained_model = train(n=N, m=M, memory_capacity=config["memory_capacity"], num_episodes=config["train_episodes"],
                          reward_fun=REWARD_FUN, batch_size=config["batch_size"], eps_start=config["eps_start"],
                          eps_end=config["eps_end"],
                          eps_decay=config["eps_decay"], optimise_freq=config["optimise_freq"],
                          target_update_freq=config["target_update_freq"],
                          eval_runs=config["eval_runs_train"], patience=config["patience"], potential_fun=POTENTIAL_FUN,
                          max_threshold=config["max_threshold"],
                          eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE, print_progress=PRINT_PROGRESS,
                          nn_model=NN_MODEL, device=DEVICE)
    score = evaluate(trained_model, n=N, m=M, reward_fun=REWARD_FUN, eval_runs_eval=EVAL_RUNS_EVAL,
                     eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE)
    return score


def selection(pop, scores, k=3):
    selection_ix = random.randrange(len(pop))
    for ix in random.choices(range(len(pop)), k=k - 1):
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


def crossover(p1: dict, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    for param in p1.keys():
        if random.random() < r_cross:
            tmp = c1[param]
            c1[param] = c2[param]
            c2[param] = tmp
    return [c1, c2]


def generate_random_hyperparameters():
    config = {
        "batch_size": random.randint(16, 64),
        "eps_start": random.uniform(0.05, 0.5),
        "eps_end": random.uniform(0.01, 0.1),
        "eps_decay": random.uniform(100, 10000),
        "train_episodes": random.randint(300, 1000),  # TODO: set back to 3000
        "target_update_freq": random.randint(1, 30),
        "memory_capacity": random.randint(16, 1000),
        "eval_runs_train": random.randint(1, 30),
        "patience": random.randint(30, 1000),
        "optimise_freq": random.randint(1, M),
        "max_threshold": random.randint(3, M)
    }
    return config


def initialize():
    return generate_random_hyperparameters()


# mutation operator
def mutation(config, r_mut):
    alternative = generate_random_hyperparameters()
    for k in config.keys():
        if random.random() < r_mut:
            config[k] = alternative[k]


# genetic algorithm
def genetic_algorithm(n_iter=100, n_pop=100, r_cross=0.9, r_mut=len(initialize())):
    pop = [initialize() for _ in range(n_pop)]
    best, best_eval = 0, - M - 1  # minus infinity basically
    for gen in range(n_iter):
        print(f"ITERATION {gen} STARTED!")
        print(f"Current population: {pop}")
        scores = [objective(frozenset(c.items())) for c in pop]
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print("----------- %d, NEW BEST SCORE f(%s) = %.3f" % (gen, pop[i], scores[i]))
                with open(f"best_hyperparams_{N}_{M}.json", 'w') as fp:
                    json.dump({**(pop[i]), "score": scores[i]}, fp)

        selected = [selection(pop, scores) for _ in range(n_pop)]
        children = []
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i + 1]
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)
                children.append(c)
        pop = children

    print('FINAL RESULT: f(%s) = %f' % (best, best_eval))
    return [best, best_eval]


##############################################


if __name__ == "__main__":
    genetic_algorithm(n_iter=30, n_pop=30)
