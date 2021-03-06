import functools
import time

from helper.helper import argmin

n = 20
m = 20
reward = max


@functools.lru_cache(maxsize=400000)  # n * m * number_of_increasing_partitions(m, n))
def dp(loads_tuple, chosen, threshold, n=n, m=m, reward=reward):
    if chosen > 0 and loads_tuple[chosen] == loads_tuple[chosen-1]:
        chosen -= 1
        while chosen > 0 and loads_tuple[chosen] == loads_tuple[chosen-1]:
            chosen -= 1
        return dp(loads_tuple, chosen, threshold=threshold, n=n, m=m, reward=reward)
    loads = list(loads_tuple)
    loads = list(loads_tuple)
    if sum(loads) == m:
        return reward(loads)
    elif loads[chosen] <= threshold:
        loads[chosen] += 1
        res = 0
        loads_sorted = sorted(loads)
        for i in range(n):
            res += dp(tuple(loads_sorted), i, threshold, n, m, reward)
        return res / n
    else:
        res = 0
        for i in range(n):
            loads[i] += 1
            loads_sorted = sorted(loads)
            for j in range(n):
                res += dp(tuple(loads_sorted), j, threshold, n, m, reward)
            loads[i] -= 1
        return res / (n * n)


@functools.lru_cache(maxsize=400000)
def dp_simpler_state(loads_tuple, threshold, n=n, m=m, reward=reward):
    loads = list(loads_tuple)
    if sum(loads) == m:
        return reward(loads)
    else:
        res = 0
        for chosen in range(n):
            if loads[chosen] <= threshold:
                loads[chosen] += 1
                res += dp_simpler_state(tuple(loads), threshold, n, m, reward)
                loads[chosen] -= 1
            else:
                subres = 0
                for rejected in range(n):
                    loads[rejected] += 1
                    subres += dp_simpler_state(tuple(loads), threshold, n, m, reward)
                    loads[rejected] -= 1
                res += subres / n
        return res / n


def find_best_constant_threshold(n=n, m=m, reward=reward):
    options = [dp(tuple([0] * n), 0, threshold, n=n, m=m, reward=reward) for threshold in range(m + 1)]
    best = argmin(options)
    print(
        f"With {m} balls and {n} bins the best constant threshold is {best}, its expected maximum load is {options[best]}")


if __name__ == "__main__":
    start_time = time.time()
    find_best_constant_threshold()
    print("--- %s seconds ---" % (time.time() - start_time))
