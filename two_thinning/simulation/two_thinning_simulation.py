import functools
import random

#from two_thinning.helper import number_of_increasing_partitions


n=20
m=20

def two_thinning_constant_threshold_simulate_one_run(threshold):
    loads=[0]*n
    for _ in range(m):
        chosen=random.randrange(n)
        if loads[chosen]<=threshold:
            loads[chosen]+=1
        else:
            arbitrary=random.randrange(n)
            loads[arbitrary]+=1
    return max(loads)


def two_thinning_constant_threshold_simulate_many_runs(threshold, runs=100):
    return sum([two_thinning_constant_threshold_simulate_one_run(threshold) for _ in range(runs)])/runs


def two_thinning_constant_threshold_simulate(top=10):
    performances=[(two_thinning_constant_threshold_simulate_many_runs(threshold), threshold) for threshold in range(m+1)]
    best_performances=sorted(performances)[:top]
    for performance, threshold in best_performances:
        print(f"With {m} balls, {n} bins and constant threshold {threshold} the maximum load is on average approximately {performance}")


if __name__=="__main__":
    two_thinning_constant_threshold_simulate()
    #print(f"According to the mathematical bounds the optimal constant threshold should be around {two_thinning_constant_threshold_maths(n)}")