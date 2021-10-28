import random

from one_choice.maths_results import high_probability_maximum_load

n = 10
m = n


def simulate_one_run():
    loads = [0] * n
    for _ in range(m):
        chosen = random.randrange(n)
        loads[chosen] += 1
    return max(loads)


def simulate_many_runs(runs=100):
    avg = sum([simulate_one_run() for _ in range(runs)]) / runs
    print(f"With {m} balls and {n} bins one-choice achieves on simulation approximately {avg} maximum load.")


if __name__ == "__main__":
    simulate_many_runs()
    print(f"According to mathematics, the maximum load should be around {high_probability_maximum_load(n)} with high probability.")
