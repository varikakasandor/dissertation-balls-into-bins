import pandas as pd
from scipy.stats import ttest_ind


def two_sample_t_test(strategy1, strategy2):
    scores1 = pd.read_csv(f"data/50_2500_{strategy1}.csv")["score"].tolist()
    scores2 = pd.read_csv(f"data/50_2500_{strategy2}.csv")["score"].tolist()
    test_result = ttest_ind(scores1, scores2)
    return test_result.pvalue, test_result.statistic


def dqn_vs_mean_thinning():
    return two_sample_t_test("dqn", "mean_thinning")


def dqn_vs_threshold():
    return two_sample_t_test("dqn", "threshold")


if __name__ == "__main__":
    p_value, t_statistic = dqn_vs_threshold()
    print(p_value, t_statistic)
