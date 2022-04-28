from os.path import exists

import matplotlib.pyplot as plt
import pandas as pd

from k_choice.graphical.two_choice.graphs.hypercube import HyperCube
from k_choice.graphical.two_choice.graphs.random_regular_graph import RandomRegularGraph
from k_choice.graphical.two_choice.strategies.greedy_strategy import GreedyStrategy
from k_choice.graphical.two_choice.environment import run_strategy
from k_choice.graphical.two_choice.full_knowledge.RL.DQN.constants import MAX_LOAD_REWARD

N = 32
M = 32
RUNS_PER_D = 1000
CREATE_PLOT = True


def analyse_random_regular(n=N, m=M, runs_per_d=RUNS_PER_D, create_plot=CREATE_PLOT):
    vals = []
    for d in range(1, n):
        for run in range(runs_per_d):
            if d < n / 2:
                graph = RandomRegularGraph(n=n, d=d)
            else:
                graph_transposed = RandomRegularGraph(n=n, d=n - 1 - d)
                graph = graph_transposed.transpose()
            score = run_strategy(graph=graph, m=m, strategy=GreedyStrategy(graph=graph, m=m), reward=MAX_LOAD_REWARD,
                                 print_behaviour=False)
            maxload = -score
            vals.append([d, maxload])

    if create_plot:
        plt.boxplot(vals)
        plt.savefig("RandomRegular-Greedy-Analysis.jpg")

    df = pd.DataFrame(data=vals, columns=["d", "score"])
    output_path = f'data/{n}_{m}_random_regular_greedy_analysis.csv'
    df.to_csv(output_path, mode='a', index=False, header=not exists(output_path))


if __name__ == "__main__":
    analyse_random_regular()
