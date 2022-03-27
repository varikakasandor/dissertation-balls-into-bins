import functools
import time

from helper.helper import number_of_increasing_partitions

N = 10
M = 15
K = 2
REWARD = max


@functools.lru_cache(maxsize=400000)  # m * n * number_of_increasing_partitions(m, n))
def dp(loads_tuple, chosen, n=N, m=M, k=K, rejects_left=K - 1, reward=REWARD):
    if chosen > 0 and loads_tuple[chosen] == loads_tuple[chosen-1]:
        chosen -= 1
        while chosen > 0 and loads_tuple[chosen] == loads_tuple[chosen-1]:
            chosen -= 1
        return dp(loads_tuple, chosen, n=n, m=m, k=k, rejects_left=rejects_left, reward=reward)
    loads = list(loads_tuple)
    if sum(loads) == m:
        return reward(loads)

    accept = 0
    loads[chosen] += 1
    loads_sorted = sorted(loads)
    for i in range(n):
        accept += dp(tuple(loads_sorted), i, n, m, k, k - 1, reward)
    loads[chosen] -= 1
    accept /= n

    reject = 0
    if rejects_left > 1:  # can reject, as there are more choices coming
        for j in range(n):
            reject += dp(tuple(loads), j, n, m, k, rejects_left - 1, reward)
        reject /= n
    else:
        for i in range(n):
            loads[i] += 1
            loads_sorted = sorted(loads)
            for j in range(n):
                reject += dp(tuple(loads_sorted), j, n, m, k, k - 1, reward)
            loads[i] -= 1
        reject /= (n * n)

    return min(accept, reject)


if __name__ == "__main__":
    start_time = time.time()

    print(f"With {M} balls and {N} bins the best achievable expected maximum load with {K}-thinning is {dp(tuple([0] * N), 0)}")

    print("--- %s seconds ---" % (time.time() - start_time))
