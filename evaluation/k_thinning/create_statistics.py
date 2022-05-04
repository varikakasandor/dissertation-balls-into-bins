from os.path import exists

import numpy as np
import pandas as pd
import scipy.stats as st

from helper.helper import flatten

NMKS = ((5, 25, 2), (5, 25, 3), (5, 25, 5), (5, 25, 10), (20, 50, 2), (20, 50, 3), (20, 50, 5), (20, 50, 10))
STRATEGIES = ("always_accept", "random", "local_reward_optimiser", "quantile", "dp", "threshold", "dqn")


def calculate_statistics(n, m, k, strategy, alpha=0.95):
    read_path = f"data/{n}_{m}_{k}_{strategy}.csv"  # if k > 2 else f"../two_thinning/data/{n}_{m}_{strategy}.csv"
    if exists(read_path):
        df = pd.read_csv(read_path)
        scores = -np.array(df["score"].to_list())[-500:]
        mean = np.mean(scores)
        sem = st.sem(scores)
        if sem > 0:
            lower, upper = st.norm.interval(alpha=alpha, loc=mean, scale=sem)
            return mean, (upper - lower) / 2
        else:
            return mean, 0
    else:
        return -1, -1


def create_csv(nmks=NMKS, strategies=STRATEGIES):
    cols = flatten([[f"mean_{n}_{m}_{k}", f"confidence_{n}_{m}_{k}"] for n, m, k in nmks])
    vals = []
    for strategy in strategies:
        row = []
        for n, m, k in nmks:
            mean, confidence = calculate_statistics(n=n, m=m, k=k, strategy=strategy)
            row.extend([mean, confidence])

        vals.append(row)

    df = pd.DataFrame(data=vals, columns=cols, index=strategies)
    output_path = f"data/comparison.csv"
    df.to_csv(output_path)
    return df


if __name__ == "__main__":
    create_csv()
