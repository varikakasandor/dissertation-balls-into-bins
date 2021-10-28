import functools
import random

from two_thinning.constant_threshold.maths_results import optimal_threshold

n = 20
m = 20


def simulate_one_run(threshold, n=n, m=m):
    loads = [0] * n
    for _ in range(m):
        chosen = random.randrange(n)
        if loads[chosen] <= threshold:
            loads[chosen] += 1
        else:
            arbitrary = random.randrange(n)
            loads[arbitrary] += 1
    return max(loads)


def simulate_many_runs(threshold, runs=100):
    return sum([simulate_one_run(threshold) for _ in range(runs)]) / runs


def simulate_and_compare(top=10):
    performances = [(simulate_many_runs(threshold), threshold) for threshold in range(m + 1)]
    best_performances = sorted(performances)[:top]
    for performance, threshold in best_performances:
        print(
            f"With {m} balls, {n} bins and constant threshold {threshold} the maximum load is on average approximately {performance}")


if __name__ == "__main__":
    simulate_and_compare()
    print(
        f"According to the mathematical bounds the optimal constant threshold should be around {optimal_threshold(n)}")
