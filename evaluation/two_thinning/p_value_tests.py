import pandas as pd
from scipy.stats import ttest_ind, wilcoxon

NMS = ((5, 5), (5, 10), (5, 25), (20, 20), (20, 60), (20, 400), (50, 50), (50, 200), (50, 2500))


def two_sample_t_test(strategy1, strategy2, n=50, m=2500, test=ttest_ind, one_sided=True):
    scores1 = pd.read_csv(f"data/{n}_{m}_{strategy1}.csv")["score"].tolist()[-500:]
    scores2 = pd.read_csv(f"data/{n}_{m}_{strategy2}.csv")["score"].tolist()[-500:]
    test_result = test(scores1, scores2, equal_var=False)
    statistic = test_result.statistic
    p_value = test_result.pvalue / 2 if one_sided else test_result.pvalue
    return p_value, statistic


def dqn_vs_mean_thinning(nms=NMS):
    for n, m in nms:
        p_value, t_statistic = two_sample_t_test("dqn", "mean_thinning", n=n, m=m, test=wilcoxon)
        print(f"n={n}, m={m}, p_value={p_value}, t_statistic={t_statistic}")


def dqn_vs_threshold(nms=NMS):
    for n, m in nms:
        p_value, t_statistic = two_sample_t_test("dqn", "threshold", n=n, m=m)
        print(f"n={n}, m={m}, p_value={p_value}, t_statistic={t_statistic}")

def dp_vs_localrewardoptimiser():
    return two_sample_t_test("dp", "local_reward_optimiser")

if __name__ == "__main__":
    dqn_vs_threshold()
    dqn_vs_mean_thinning()
