import functools
import time
import random

from k_choice.graphical.two_choice.graphs.cycle import Cycle

from helper.helper import number_of_increasing_partitions

N = 4
M = 6
REWARD = max
GRAPH = Cycle(N)

strategy = {}


def cached(func):
    func.cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func.cache[args]
        except KeyError:
            func.cache[args] = result = func(*args)
            return result

    return wrapper


# @functools.lru_cache(maxsize=400000)  # m * n * number_of_increasing_partitions(m, n))
@cached
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

    global strategy
    strategy[(loads_tuple, chosen_edge)] = -1 if (options[0] < options[1]) else (1 if options[1] < options[0] else 0)
    return min(options)


def dp(graph=GRAPH, m=M, reward=REWARD):
    avg = 0
    for edge in graph.edge_list:
        avg += dp_helper(tuple([0] * graph.n), edge, reward=reward, graph=graph, m=m)
    return avg / graph.e


if __name__ == "__main__":
    start_time = time.time()

    # Create strategy
    dp(graph=GRAPH, reward=REWARD, m=M)

    for ((loads, (x, y)), decision) in strategy.items():
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
