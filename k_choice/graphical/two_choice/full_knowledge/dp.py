import time

from k_choice.graphical.two_choice.graphs.cycle import Cycle
from k_choice.graphical.two_choice.graphs.hypercube import HyperCube
from k_choice.graphical.two_choice.graphs.complete_graph import CompleteGraph

from helper.helper import number_of_increasing_partitions

N = 10
M = 10
GRAPH = CompleteGraph(N)
DICT_LIMIT = 400000  # M * N * number_of_increasing_partitions(N, M)


def REWARD_FUN(loads):
    return -max(loads)


def dp_helper(loads_tuple, memo, strategy, graph=GRAPH, m=M, reward=REWARD_FUN, dict_limit=DICT_LIMIT):
    if loads_tuple in memo:
        return memo[loads_tuple]
    loads = list(loads_tuple)
    if sum(loads) == m:
        return reward(loads)

    next_vals = []
    for node in range(graph.n):
        loads[node] += 1
        next_vals.append(dp_helper(tuple(loads), memo, strategy, reward=reward, graph=graph, m=m))
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


def dp(graph=GRAPH, m=M, reward=REWARD_FUN):
    memo = {}
    strategy = {}
    score = dp_helper(tuple([0] * graph.n), memo, strategy, reward=reward, graph=graph, m=m)
    return score , strategy


if __name__ == "__main__":
    start_time = time.time()
    score, strategy = dp()
    print(
        f"With {N} bins and {M} balls on a {GRAPH}, the best achievable expected maximum load is {-score}")

    # for ((loads, (x, y)), decision) in strategy.items():
    #    if ((loads[x] < loads[y]) and decision == 1) or ((loads[y] < loads[x]) and decision == -1):
    #        print(loads, (x, y), decision)

    print("--- %s seconds ---" % (time.time() - start_time))
