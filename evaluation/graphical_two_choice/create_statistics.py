from os.path import exists

import numpy as np
import pandas as pd
import scipy.stats as st

from helper.helper import flatten
from k_choice.graphical.two_choice.graphs.complete_graph import CompleteGraph
from k_choice.graphical.two_choice.graphs.cycle import Cycle
from k_choice.graphical.two_choice.graphs.hypercube import HyperCube
from k_choice.graphical.two_choice.strategies.full_knowledge_DQN_strategy import FullKnowledgeDQNStrategy

GMS = ((Cycle(4), 25), (HyperCube(4), 25), (CompleteGraph(4), 25),
       (Cycle(16), 50), (HyperCube(16), 50), (CompleteGraph(16), 50),
       (Cycle(32), 32), (HyperCube(32), 32), (CompleteGraph(32), 32))

STRATEGIES = ("greedy", "random", "local_reward_optimiser", "dp", "dqn")


def calculate_statistics(graph, m, strategy, alpha=0.95):
    read_path = f"data/{graph.name}_{graph.n}_{m}_{strategy}.csv"
    if exists(read_path):
        df = pd.read_csv(read_path)
        scores = df["score"].to_list()
        scores = -np.array(scores[:100] * 5 if strategy == "blabla" else scores[-500:])
        mean = np.mean(scores)
        sem = st.sem(scores)
        if sem > 0:
            lower, upper = st.norm.interval(alpha=alpha, loc=mean, scale=sem)
            return mean, (upper - lower) / 2
        else:
            return mean, 0
    else:
        return -1, -1


def create_csv(gms=GMS, strategies=STRATEGIES):
    cols = flatten([[f"mean_{graph.name}_{graph.n}_{m}", f"confidence_{graph.name}_{graph.n}_{m}"] for graph, m in gms])
    vals = []
    for strategy in strategies:
        row = []
        for graph, m in gms:
            mean, confidence = calculate_statistics(graph=graph, m=m, strategy=strategy)
            row.extend([mean, confidence])

        vals.append(row)

    df = pd.DataFrame(data=vals, columns=cols, index=strategies)
    output_path = f"data/comparison.csv"
    df.to_csv(output_path)
    return df


if __name__ == "__main__":
    create_csv()
