import random


n=20
m=20

def simulate_one_run(thresholds):
    loads=[0]*n
    for i in range(m):
        chosen=random.randrange(n)
        if loads[chosen]<=thresholds[i]:
            loads[chosen]+=1
        else:
            arbitrary=random.randrange(n)
            loads[arbitrary]+=1
    return max(loads)


def simulate_many_runs(thresholds, runs=100):
    return sum([simulate_one_run(thresholds) for _ in range(runs)]) / runs


