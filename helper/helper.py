import functools
from math import *

n = 8
m = 20


@functools.lru_cache()
def number_of_increasing_partitions(n, k):
    if n == 0 or k == 1:
        return 1
    if n >= k:
        return number_of_increasing_partitions(n, k - 1) + number_of_increasing_partitions(n - k, k)
    return number_of_increasing_partitions(n, k - 1)


def argmax(l):
    f = lambda i: l[i]
    return max(range(len(l)), key=f)

def argmin(l):
    f = lambda i: l[i]
    return min(range(len(l)), key=f)

def variance(loads):
    mean = sum(loads) / len(loads)
    variance = sum([(x - mean) ** 2 for x in loads]) / len(loads)
    return variance

def std(loads):
    return sqrt(variance(loads))

if __name__ == "__main__":
    print(number_of_increasing_partitions(m, n))
