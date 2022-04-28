from os.path import exists

import pandas as pd
from matplotlib import pyplot as plt

from two_thinning.strategies.local_reward_optimiser_strategy import LocalRewardOptimiserStrategy
from two_thinning.environment import run_strategy_multiple_times
from two_thinning.full_knowledge.RL.DQN.constants import *

MAX_N = 1000
RUNS_PER_N = 50
CREATE_PLOT = True


def linear(n):
    return n


def analyse_local_reward_optimiser(max_n=MAX_N, m=linear, runs_per_n=RUNS_PER_N, create_plot=CREATE_PLOT):
    vals = []
    for n in range(2, max_n):
        scores = run_strategy_multiple_times(n=n, m=m(n), runs=runs_per_n,
                                             strategy=LocalRewardOptimiserStrategy(n=n, m=m(n),
                                                                                   reward_fun=MAX_LOAD_REWARD,
                                                                                   potential_fun=MAX_LOAD_POTENTIAL),
                                             reward=MAX_LOAD_REWARD, print_behaviour=False)
        for score in scores:
            max_load = - score
            vals.append([n, max_load])

    if create_plot:
        plt.boxplot(vals)
        plt.savefig("LocalRewardOptimiser-Analysis.jpg")

    df = pd.DataFrame(data=vals, columns=["n", "max_load"])
    output_path = f'data/local_reward_optimiser_analysis.csv'
    df.to_csv(output_path, mode='a', index=False, header=not exists(output_path))


if __name__=="__main__":
    analyse_local_reward_optimiser()