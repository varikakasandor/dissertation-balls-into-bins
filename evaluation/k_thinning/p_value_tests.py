import pandas as pd
from scipy.stats import ttest_ind, wilcoxon

NMKS = ((5, 25, 2), (5, 25, 3), (5, 25, 5), (5, 25, 10), (20, 50, 2), (20, 50, 3), (20, 50, 5), (20, 50, 10))


def two_sample_t_test(strategy1, strategy2, n=25, m=50, k=3, test=ttest_ind, one_sided=True):
    scores1 = pd.read_csv(f"data/{n}_{m}_{k}_{strategy1}.csv")["score"].tolist()[-500:]
    scores2 = pd.read_csv(f"data/{n}_{m}_{k}_{strategy2}.csv")["score"].tolist()[-500:]
    try:
        test_result = test(scores1, scores2, equal_var=False)
    except:
        test_result = test(scores1, scores2)
    statistic = test_result.statistic
    p_value = test_result.pvalue / 2 if one_sided else test_result.pvalue
    return p_value, statistic


def compare_strategies(strategy1, strategy2, nmks=NMKS, exclude_n=[], exlclude_m=[], exclude_k=[]):
    for n, m, k in nmks:
        if n in exclude_n or m in exlclude_m or k in exclude_k:
            continue
        p_value, t_statistic = two_sample_t_test(strategy1, strategy2, n=n, m=m, k=k, test=wilcoxon)
        print(f"better={strategy1}, worse={strategy2}, n={n}, m={m}, k={k}, p_value={p_value}, t_statistic={t_statistic}")


if __name__ == "__main__":
    compare_strategies("dqn", "threshold", exclude_k=[2])
    compare_strategies("quantile", "dqn", exclude_k=[2])
    compare_strategies("local_reward_optimiser", "dqn", exclude_k=[2])