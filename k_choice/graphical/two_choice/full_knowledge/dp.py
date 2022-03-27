import time

from k_choice.graphical.two_choice.graphs.cycle import Cycle
from k_choice.graphical.two_choice.graphs.hypercube import HyperCube

from helper.helper import number_of_increasing_partitions

N = 4
M = 15
REWARD = max
GRAPH = Cycle(N)
DICT_LIMIT = 400000  # M * N * number_of_increasing_partitions(N, M)


def dp_helper(loads_tuple, chosen_edge, strategy, graph=GRAPH, m=M, reward=REWARD, dict_limit=DICT_LIMIT):
    if (loads_tuple, chosen_edge) in strategy:
        val, _ = strategy[(loads_tuple, chosen_edge)]
        return val
    loads = list(loads_tuple)
    if sum(loads) == m:
        return reward(loads)

    options = []
    for node in chosen_edge:
        loads[node] += 1
        avg = 0
        for edge in graph.edge_list:
            avg += dp_helper(tuple(loads), edge, strategy, reward=reward, graph=graph, m=m)
        avg /= graph.e
        options.append(avg)
        loads[node] -= 1

    val = min(options)
    if len(strategy) < dict_limit:
        decision = -1 if (options[0] < options[1]) else (1 if options[1] < options[0] else 0)
        strategy[(loads_tuple, chosen_edge)] = (val, decision)
    return val


def dp(graph=GRAPH, m=M, reward=REWARD):
    avg = 0
    strategy = {}
    for edge in graph.edge_list:
        avg += dp_helper(tuple([0] * graph.n), edge, strategy, reward=reward, graph=graph, m=m)
    return avg / graph.e, strategy


if __name__ == "__main__":
    start_time = time.time()

    score, strategy = dp()
    print(score)

    for ((loads, (x, y)), (val, decision)) in strategy.items():
        if ((loads[x] < loads[y]) and decision == 1) or ((loads[y] < loads[x]) and decision == -1):
            print(loads, (x, y), decision)

    """for _ in range(10):
        print("----------------")
        loads = [0] * N
        for _ in range(M):
            (x, y) = random.choice(GRAPH.edge_list)
            decision = strategy[(tuple(loads), (x, y))]
            print(loads, (x, y), decision)
            if ((loads[x] < loads[y]) and decision == 1) or ((loads[y] < loads[x]) and decision == -1):
                print("!!!!!!!!!!!")
            chosen = x if decision != 1 else y
            loads[chosen] += 1

    print(dp_helper((0, 1, 0, 2), (2, 3)))"""

    """print(
        f"With {M} balls and {N} bins on a cycle, the best achievable expected maximum load is {dp(graph=GRAPH, reward=REWARD, m=M)}")


    print(strategy)"""

    print("--- %s seconds ---" % (time.time() - start_time))
