import functools
import time

from helper import number_of_increasing_partitions

n = 20
m = n


@functools.lru_cache(maxsize=m * n * number_of_increasing_partitions(m, n))
def dp(loads_tuple, chosen):
    loads = list(loads_tuple)
    if sum(loads) == m:
        return max(loads)

    accept = 0
    loads[chosen] += 1
    loads_sorted = sorted(loads)
    for i in range(n):
        accept += dp(tuple(loads_sorted), i)
    loads[chosen] -= 1
    accept /= n

    reject = 0
    for i in range(n):
        loads[i] += 1
        loads_sorted = sorted(loads)
        for j in range(n):
            reject += dp(tuple(loads_sorted), j)
        loads[i] -= 1
    reject /= (n * n)

    return min(accept, reject)


if __name__ == "__main__":
    start_time = time.time()

    print(f"With {m} balls, {n} bins the best achievable expected maximum load is {dp(tuple([0] * n), 0)}")

    print("--- %s seconds ---" % (time.time() - start_time))
