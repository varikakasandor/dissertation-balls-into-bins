import functools

import numpy as np

N = 20
M = 50
K = 3

def flatten(l):
    return [item for sublist in l for item in sublist]

@functools.lru_cache()
def number_of_increasing_partitions(m=M, n=N):
    if m == 0 or n == 1:
        return 1
    if m >= n:
        return number_of_increasing_partitions(m, n - 1) + number_of_increasing_partitions(m - n, n)
    # Either all the bins get 1 ball, or we move on from the least loaded bin. This is a nice trick to make it O(n*m) instead of cubic.
    return number_of_increasing_partitions(m, n - 1)


def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)


def argmin(l):
    f = lambda i: l[i]
    return min(range(len(l)), key=f)


def variance(loads):
    return np.var(np.array(loads))


def std(loads):
    return np.std(np.array(loads))


if __name__ == "__main__":
    num_states = 0
    for i in range(M + 1):
        curr = number_of_increasing_partitions(m=i, n=N)
        num_states += curr
    num_steps = num_states * N * K
    num_seconds = num_steps / (10 ** 8)
    print(num_states, num_steps, num_seconds)
