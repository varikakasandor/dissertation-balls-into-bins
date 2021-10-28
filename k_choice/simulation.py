import random

from k_choice.maths_results import one_choice_high_probability_maximum_load, two_choice_high_probability_maximum_load

n = 1000
m = n


def simulate_one_run(choices, n=n, m=m):
    loads = [0] * n
    for _ in range(m):
        options = random.sample(range(n), choices)
        chosen = min(options, key=lambda x: loads[x])
        loads[chosen] += 1
    return max(loads)


def simulate_many_runs(choices, runs=100, n=n, m=m):
    avg = sum([simulate_one_run(choices, n, m) for _ in range(runs)]) / runs
    print(f"With {m} balls and {n} bins {choices}-choice achieves on simulation approximately {avg} maximum load.")


def one_choice_simulate_many_runs(runs=100, n=n, m=m):
    simulate_many_runs(1, runs, n, m)


def two_choice_simulate_many_runs(runs=100, n=n, m=m):
    simulate_many_runs(2, runs, n, m)


if __name__ == "__main__":
    one_choice_simulate_many_runs()
    two_choice_simulate_many_runs()
    print(f"According to mathematics, the maximum load of 1-choice should be around {one_choice_high_probability_maximum_load(n,m)} with high probability.")
    print(f"According to mathematics, the maximum load of 2-choice should be around {two_choice_high_probability_maximum_load(n,m)} with high probability.")
