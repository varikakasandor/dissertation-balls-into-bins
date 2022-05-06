import time
import csv

from k_choice.graphical.two_choice.graphs.cycle import Cycle
from k_choice.graphical.two_choice.graphs.hypercube import HyperCube
from k_choice.graphical.two_choice.graphs.complete_graph import CompleteGraph

from helper.helper import number_of_increasing_partitions

N = 4
M = 6
GRAPH = Cycle(N)
DICT_LIMIT = 40000000  # M * N * number_of_increasing_partitions(N, M)
PRINT_BEHAVIOUR = True


def REWARD_FUN(loads):
    return -max(loads)


def dp(loads_tuple, memo, strategy, graph=GRAPH, m=M, reward_fun=REWARD_FUN, dict_limit=DICT_LIMIT):
    if loads_tuple in memo:
        return memo[loads_tuple]
    loads = list(loads_tuple)
    if sum(loads) == m:
        return reward_fun(loads)


    next_vals = []
    for node in range(graph.n):
        loads[node] += 1
        next_vals.append(dp(tuple(loads), memo, strategy, reward_fun=reward_fun, graph=graph, m=m))
        loads[node] -= 1


    avg = 0
    for (x, y) in graph.edge_list:
        avg += max(next_vals[x], next_vals[y])
        decision = -1 if (next_vals[x] > next_vals[y]) else (1 if next_vals[y] > next_vals[x] else 0)
        strategy[(loads_tuple, (x, y))] = decision
    avg /= graph.e

    if len(memo) < dict_limit:
        memo[loads_tuple] = avg
    return avg


def find_best_strategy(graph=GRAPH, m=M, reward_fun=REWARD_FUN, print_behaviour=PRINT_BEHAVIOUR):
    memo = {}
    strategy = {}
    score = dp(tuple([0] * graph.n), memo, strategy, reward_fun=reward_fun, graph=graph, m=m)
    if print_behaviour:
        print(f"With {N} bins and {M} balls on a {GRAPH}, the best achievable expected maximum load is {-score}")
    return strategy


def analyse_0x0y(strategy, n=N, m=M):
    table = [[strategy[(0, i, 0, j), (1, 2)] if i + j < m else 9 for j in range(m)] for i in range(m)]
    with open(f"../../../../evaluation/graphical_two_choice/data/counterexample_analysis_{n}_{m}.csv", "w", newline='') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(table)

if __name__ == "__main__":
    start_time = time.time()
    strategy = find_best_strategy()

    for ((loads, (x, y)), decision) in strategy.items():
        if ((loads[x] < loads[y]) and decision == 1) or ((loads[y] < loads[x]) and decision == -1):
            print(loads, (x, y), decision)

    #analyse_0x0y(strategy)

    print("--- %s seconds ---" % (time.time() - start_time))
