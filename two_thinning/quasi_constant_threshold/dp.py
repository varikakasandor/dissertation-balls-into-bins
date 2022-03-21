import functools
import time

from two_thinning.quasi_constant_threshold.maths_results import optimal_threshold, optimal_result
from helper.helper import argmin

n = 5
m = 15
reward = max


@functools.lru_cache(maxsize=400000)  # n * m * number_of_increasing_partitions(m, n))
def dp(loads_and_first_tuple, chosen, threshold, n=n, m=m, reward=reward):
    loads_and_first = list(loads_and_first_tuple)
    loads = list([x[0] for x in loads_and_first])
    if sum(loads) == m:
        return reward(loads)
    elif loads_and_first[chosen][1] <= threshold:
        loads_and_first[chosen] = (loads_and_first[chosen][0]+1, loads_and_first[chosen][1]+1)
        res = 0
        loads_sorted = sorted(loads_and_first)
        for i in range(n):
            res += dp(tuple(loads_sorted), i, threshold, n, m, reward)
        return res / n
    else:
        res = 0
        for i in range(n):
            loads_and_first[i] = (loads_and_first[i][0]+1, loads_and_first[i][1])
            loads_sorted = sorted(loads_and_first)
            for j in range(n):
                res += dp(tuple(loads_sorted), j, threshold, n, m, reward)
            loads_and_first[i] = (loads_and_first[i][0]-1, loads_and_first[i][1])
        return res / (n * n)


def find_best_quasi_constant_threshold(n=n, m=m, reward=reward):
    options = [dp(tuple([(0, 0)] * n), 0, threshold, n=n, m=m, reward=reward) for threshold in range(m + 1)]
    best = argmin(options)
    print(
        f"With {m} balls and {n} bins the best constant threshold is {best}, its expected maximum load is {options[best]}")


if __name__ == "__main__":
    start_time = time.time()
    find_best_quasi_constant_threshold()
    print(
        f"According to the mathematical bounds the optimal constant threshold should be around {optimal_threshold(n)}, reaching an optimal result {optimal_result((n))}")
    print("--- %s seconds ---" % (time.time() - start_time))
