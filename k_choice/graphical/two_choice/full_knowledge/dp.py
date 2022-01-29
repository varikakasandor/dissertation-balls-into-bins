import functools
import time

from k_choice.graphical.two_choice.graphs.cycle import Cycle

from helper.helper import number_of_increasing_partitions

N = 3
M = 5
REWARD = max
GRAPH = Cycle(N)


@functools.lru_cache(maxsize=400000)  # m * n * number_of_increasing_partitions(m, n))
def dp_helper(loads_tuple, chosen_edge, graph=GRAPH, m=M, reward=REWARD):
    loads = list(loads_tuple)
    if sum(loads) == m:
        return reward(loads)

    options = []
    for node in chosen_edge:
        loads[node] += 1
        avg = 0
        for edge in graph.edge_list:
            avg += dp_helper(tuple(loads), edge, reward=reward, graph=graph, m=m)
        avg /= graph.e
        options.append(avg)
        loads[node] -= 1

    return min(options)


def dp(graph=GRAPH, m=M, reward=REWARD):
    avg = 0
    for edge in graph.edge_list:
        avg += dp_helper(tuple([0] * graph.n), edge, reward=reward, graph=graph, m=m)
    return avg / graph.e


if __name__ == "__main__":
    start_time = time.time()

    print(f"With {M} balls and {N} bins on a cycle, the best achievable expected maximum load is {dp(graph=GRAPH, reward=REWARD, m=M)}")

    print("--- %s seconds ---" % (time.time() - start_time))
