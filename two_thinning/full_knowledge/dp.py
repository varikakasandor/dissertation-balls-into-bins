import functools
import time

from helper.helper import number_of_increasing_partitions

N = 20
M = 20
REWARD = max


@functools.lru_cache(maxsize=400000)  # m * n * number_of_increasing_partitions(m, n))
def dp(loads_tuple, chosen, n=N, m=M, reward=REWARD):
    if chosen > 0 and loads_tuple[chosen] == loads_tuple[chosen-1]:
        chosen -= 1
        while chosen > 0 and loads_tuple[chosen] == loads_tuple[chosen-1]:
            chosen -= 1
        return dp(loads_tuple, chosen, n=n, m=m, reward=reward)
    loads = list(loads_tuple)
    if sum(loads) == m:
        return reward(loads)

    accept = 0
    loads[chosen] += 1
    loads_sorted = sorted(loads)
    for i in range(n):
        accept += dp(tuple(loads_sorted), i, n, m, reward)
    loads[chosen] -= 1
    accept /= n

    reject = 0
    for i in range(n):
        loads[i] += 1
        loads_sorted = sorted(loads)
        for j in range(n):
            reject += dp(tuple(loads_sorted), j, n, m, reward)
        loads[i] -= 1
    reject /= (n * n)

    return min(accept, reject)


def find_best_thresholds(n=N, m=M, reward=REWARD):
    print(
        f"With {m} balls and {n} bins the best achievable expected maximum load with two-thinning is {dp(tuple([0] * n), 0, n=n, m=m, reward=reward)}")


if __name__ == "__main__":
    start_time = time.time()

    print(f"With {M} balls and {N} bins the best achievable expected maximum load is {dp(tuple([0] * N), 0)}")

    print("--- %s seconds ---" % (time.time() - start_time))
