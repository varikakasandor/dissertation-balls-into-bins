import os
from os.path import exists, dirname, join, abspath
import scipy.stats as st

from helper.helper import flatten

import numpy as np
import pandas as pd

NMS = ((5, 5), (5, 10), (5, 25), (20, 20), (20, 60), (20, 400), (50, 50), (50, 200), (50, 2500))
STRATEGIES = ("always_accept", "local_reward_optimiser", "mean_thinning", "dp", "dqn", "threshold")


def calculate_statistics(n, m, strategy, alpha=0.95):
    read_path = f"data/{n}_{m}_{strategy}.csv"
    if exists(read_path):
        df = pd.read_csv(read_path)
        scores = -np.array(df["score"].to_list())[-500:]
        mean = np.mean(scores)
        """lower = np.percentile(scores, 2.5)
        upper = np.percentile(scores, 97.5)"""
        sem = st.sem(scores)
        lower, upper = st.norm.interval(alpha=alpha, loc=mean, scale=sem)
        return mean, (upper-lower)/2
    else:
        return -1, -1


def create_csv(nms=NMS, strategies=STRATEGIES):
    cols = flatten([[f"mean_{n}_{m}", f"confidence_{n}_{m}"] for n,m in nms])
    vals = []
    for strategy in strategies:
        row = []
        for n,m in nms:
            mean, confidence = calculate_statistics(n=n, m=m, strategy=strategy)
            row.extend([mean, confidence])

        vals.append(row)

    df = pd.DataFrame(data=vals, columns=cols, index=strategies)
    output_path = f"data/comparison.csv"
    df.to_csv(output_path)
    return df


if __name__ == "__main__":
    create_csv()
