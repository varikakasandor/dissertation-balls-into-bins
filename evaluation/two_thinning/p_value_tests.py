import pandas as pd
from scipy.stats import ttest_ind


def two_sample_t_test(strategy1, strategy2, one_sided=True):
    scores1 = pd.read_csv(f"data/50_2500_{strategy1}.csv")["score"].tolist()
    scores2 = pd.read_csv(f"data/50_2500_{strategy2}.csv")["score"].tolist()
    test_result = ttest_ind(scores1, scores2, equal_var=False)
    statistic = test_result.statistic
    p_value = test_result.pvalue / 2 if one_sided else test_result.pvalue
    return p_value, statistic


def dqn_vs_mean_thinning():
    return two_sample_t_test("dqn", "mean_thinning")


def dqn_vs_threshold():
    return two_sample_t_test("dqn", "threshold")


if __name__ == "__main__":
    p_value, t_statistic = dqn_vs_mean_thinning()
    print(p_value, t_statistic)
