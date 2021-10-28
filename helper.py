import functools

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


if __name__ == "__main__":
    print(number_of_increasing_partitions(m, n))
