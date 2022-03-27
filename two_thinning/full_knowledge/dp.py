import functools
import time

from helper.helper import number_of_increasing_partitions

N = 10
M = 10
REWARD = max
DICT_LIMIT = 400000  # M * N * number_of_increasing_partitions(N, M)


def find_earliest(loads_tuple, chosen):
    while chosen > 0 and loads_tuple[chosen] == loads_tuple[chosen - 1]:
        chosen -= 1
    return chosen


def dp(loads_tuple, chosen, strategy, n=N, m=M, reward=REWARD, dict_limit=DICT_LIMIT):
    if (loads_tuple, chosen) in strategy:
        val, _ = strategy[(loads_tuple, chosen)]
        return val

    earliest_occurence = find_earliest(loads_tuple, chosen)
    if earliest_occurence != chosen:
        return dp(loads_tuple, earliest_occurence, strategy, n=n, m=m, reward=reward)


    loads = list(loads_tuple)
    if sum(loads) == m:
        return reward(loads)

    accept = 0
    loads[chosen] += 1
    loads_sorted = sorted(loads)
    for i in range(n):
        accept += dp(tuple(loads_sorted), i, strategy, n=n, m=m, reward=reward)
    loads[chosen] -= 1
    accept /= n

    reject = 0
    for i in range(n):
        loads[i] += 1
        loads_sorted = sorted(loads)
        for j in range(n):
            reject += dp(tuple(loads_sorted), j, strategy, n=n, m=m, reward=reward)
        loads[i] -= 1
    reject /= (n * n)

    val = min(accept, reject)
    if len(strategy) < dict_limit:
        decision = -1 if (accept < reject) else (1 if reject < accept else 0)
        strategy[(loads_tuple, chosen)] = (val, decision)

    return val


def find_best_thresholds(n=N, m=M, reward=REWARD):
    strategy = {}
    print(
        f"With {m} balls and {n} bins the best achievable expected maximum load with two-thinning is {dp(tuple([0] * n), 0, strategy, n=n, m=m, reward=reward)}")
    return strategy


if __name__ == "__main__":
    start_time = time.time()

    strategy = find_best_thresholds()
    print(strategy)
    print("--- %s seconds ---" % (time.time() - start_time))
