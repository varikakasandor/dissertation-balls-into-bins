import random

from k_choice.maths_results import one_choice_high_probability_maximum_load, two_choice_high_probability_maximum_load

N = 3
M = 5
RUNS = 100
REWARD = max
PRINT_BEHAVIOUR = True

def simulate_one_run(choices, n=N, m=M, reward=REWARD):
    loads = [0] * n
    for _ in range(m):
        options = random.sample(range(n), choices)
        chosen = min(options, key=lambda x: loads[x])
        loads[chosen] += 1
    return reward(loads)


def simulate_many_runs(choices, runs=RUNS, n=N, m=M, reward=REWARD, print_behaviour=PRINT_BEHAVIOUR):
    avg = sum([simulate_one_run(choices, n=n, m=m, reward=reward) for _ in range(runs)]) / runs
    if print_behaviour:
        print(f"With {m} balls and {n} bins {choices}-choice achieves on simulation approximately {avg} maximum load, {avg - m / n} normalised maximum load.")
    return avg


def one_choice_simulate_many_runs(runs=RUNS, n=N, m=M, reward=REWARD, print_behaviour=PRINT_BEHAVIOUR):
    return simulate_many_runs(1, runs=runs, n=n, m=m, reward=reward, print_behaviour=print_behaviour)


def two_choice_simulate_many_runs(runs=RUNS, n=N, m=M, reward=REWARD, print_behaviour=PRINT_BEHAVIOUR):
    return simulate_many_runs(2, runs=runs, n=n, m=m, reward=reward, print_behaviour=print_behaviour)


if __name__ == "__main__":
    one_choice_simulate_many_runs()
    two_choice_simulate_many_runs()
    print(
        f"According to mathematics, the maximum load of 1-choice should be around {one_choice_high_probability_maximum_load(N, M)} with high probability.")
    print(
        f"According to mathematics, the maximum load of 2-choice should be around {two_choice_high_probability_maximum_load(N, M)} with high probability.")
