import csv
from math import ceil, log

from k_choice.graphical.two_choice.full_knowledge.RL.DQN.constants import *
from k_choice.graphical.two_choice.full_knowledge.RL.DQN.evaluate import load_best_model

N = 4
M = 25


def analyse_0x0y(n=N, m=M):
    model = load_best_model(n=n, m=m, nn_type="general_net_cycle")

    def get_decision(loads, edge):
        x, y = edge
        vals = model(torch.as_tensor(loads).unsqueeze(0)).squeeze(0).detach()
        decision = -1 if vals[x] >= vals[y] else 1 if vals[y] >= vals[x] else 0
        return decision

    table = [[get_decision([0, i, 0, j], (1, 2)) if i + j < m else 9 for j in range(m)] for i in range(m)]
    with open(f"data/counterexample_analysis_dqn_{n}_{m}.csv", "w", newline='') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(table)


if __name__ == "__main__":
    analyse_0x0y()
