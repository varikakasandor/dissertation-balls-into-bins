import random

from two_thinning.quasi_constant_threshold.maths_results import optimal_threshold

n = 10  # It is just at n=5000 that the optimal constant threshold moves from 1 to 2
m = 20
reward = max
runs = 100
top = 10
max_threshold = 5
primary_only = False


def simulate_one_run(threshold, reward=reward, n=n, m=m, primary_only=primary_only):
    loads = [0] * n
    primary = [0] * n
    for _ in range(m):
        chosen = random.randrange(n)
        based_on = primary if primary_only else loads
        if based_on[chosen] <= threshold:
            loads[chosen] += 1
            primary[chosen] += 1
        else:
            arbitrary = random.randrange(n)
            loads[arbitrary] += 1
    return reward(loads)


def simulate_many_runs(threshold, reward=reward, n=n, m=m, runs=runs):
    return sum([simulate_one_run(threshold, reward=reward, n=n, m=m) for _ in range(runs)]) / runs


def simulate_and_compare(reward=reward, n=n, m=m, runs=runs, top=top, max_threshold=max_threshold):
    performances = [(simulate_many_runs(threshold, reward=reward, n=n, m=m, runs=runs), threshold) for threshold in
                    range(max_threshold + 1)]
    best_performances = sorted(performances)[:top]
    for performance, threshold in best_performances:
        print(
            f"With {m} balls, {n} bins and constant threshold {threshold} the maximum load is on average approximately {performance}")
    return best_performances


if __name__ == "__main__":
    simulate_and_compare()
    print(
        f"According to the mathematical bounds the optimal constant threshold should be around {optimal_threshold(n)}")
