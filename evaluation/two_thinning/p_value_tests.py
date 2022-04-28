import pandas as pd
from scipy.stats import ttest_ind


def get_p_value(strategy1, strategy2):
    scores1 = pd.read_csv(f"data/50_2500_{strategy1}.csv")["score"].tolist()
    scores2 = pd.read_csv(f"data/50_2500_{strategy2}.csv")["score"].tolist()
    test_result = ttest_ind(scores1, scores2)
    return test_result.pvalue, test_result.statistic


def dqn_vs_mean_thinning():
    return get_p_value("dqn", "mean_thinning")


def dqn_vs_threshold():
    return get_p_value("dqn", "threshold")


if __name__ == "__main__":
    p_value = dqn_vs_threshold()
    print(p_value)
