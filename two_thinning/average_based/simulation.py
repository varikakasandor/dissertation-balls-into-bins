import random

n = 20
m = n
reward = max

def simulate_one_run(thresholds, reward=reward, n=n, m=m):
    loads = [0] * n
    for i in range(m):
        chosen = random.randrange(n)
        if loads[chosen] <= thresholds[i]:
            loads[chosen] += 1
        else:
            arbitrary = random.randrange(n)
            loads[arbitrary] += 1
    return reward(loads)


def simulate_many_runs(thresholds, reward=reward, runs=100, n=n, m=m):
    return sum([simulate_one_run(thresholds, reward=reward, n=n, m=m) for _ in range(runs)]) / runs

