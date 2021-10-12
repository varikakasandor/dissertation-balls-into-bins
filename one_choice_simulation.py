import functools
import random

from mathematical_results import one_choice_maths

n=10

def one_choice_simulate_one_run():
    loads=[0]*n
    for _ in range(n):
        chosen=random.randrange(n)
        loads[chosen]+=1
    return max(loads)


def one_choice_simulate_many_runs(runs=100):
    avg=sum([one_choice_simulate_one_run() for _ in range(runs)])/runs
    print(f"With {n} balls and {n} bins one-choice achieves on simulation approximately {avg} maximum load.")



if __name__=="__main__":
    one_choice_simulate_many_runs()
    print(f"According to the mathematical bounds the optimal constant threshold should be around {one_choice_maths(n)}")