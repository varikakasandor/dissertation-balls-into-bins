import functools
import time
from collections import Counter
from math import ceil
from os.path import abspath, dirname, join

import numpy as np
from matplotlib import pylab as pyl
from matplotlib import pyplot as plt

N = 5
M = 20
K = 5
DICT_LIMIT = 10000000  # M * N * K * number_of_increasing_partitions(N, M)
PRINT_BEHAVIOUR = True


def REWARD_FUN(loads):
    return -max(loads)


def rindex(l, elem):
    return len(l) - next(i for i, v in enumerate(reversed(l), 1) if v == elem)


@functools.lru_cache(maxsize=400000)  # m * n * number_of_increasing_partitions(m, n))
def old_dp(loads_tuple, chosen, n=N, m=M, k=K, rejects_left=K - 1, reward_fun=REWARD_FUN):
    if chosen > 0 and loads_tuple[chosen] == loads_tuple[chosen - 1]:
        chosen -= 1
        while chosen > 0 and loads_tuple[chosen] == loads_tuple[chosen - 1]:
            chosen -= 1
        return old_dp(loads_tuple, chosen, n=n, m=m, k=k, rejects_left=rejects_left, reward_fun=reward_fun)
    loads = list(loads_tuple)
    if sum(loads) == m:
        return reward_fun(loads)

    accept = 0
    loads[chosen] += 1
    loads_sorted = sorted(loads)
    for i in range(n):
        accept += old_dp(tuple(loads_sorted), chosen=i, n=n, m=m, k=k, rejects_left=k - 1, reward_fun=reward_fun)
    loads[chosen] -= 1
    accept /= n

    reject = 0
    if rejects_left > 1:  # can reject, as there are more choices coming
        for j in range(n):
            reject += old_dp(tuple(loads), chosen=j, n=n, m=m, k=k, rejects_left=rejects_left - 1,
                             reward_fun=reward_fun)
        reject /= n
    else:
        for i in range(n):
            loads[i] += 1
            loads_sorted = sorted(loads)
            for j in range(n):
                reject += old_dp(tuple(loads_sorted), chosen=j, n=n, m=m, k=k, rejects_left=k - 1,
                                 reward_fun=reward_fun)
            loads[i] -= 1
        reject /= (n * n)

    return max(accept, reject)


def threshold_dp(loads_tuple, choices_left, strategy, n=N, m=M, k=K, reward_fun=REWARD_FUN, dict_limit=DICT_LIMIT):
    if (loads_tuple, choices_left) in strategy:
        best_expected_score, _ = strategy[(loads_tuple, choices_left)]
        return best_expected_score

    loads = list(loads_tuple)
    if sum(loads) == m:
        final_reward = reward_fun(loads)
        strategy[(loads_tuple, choices_left)] = (final_reward, m)
        return final_reward

    reject_avg = 0
    results = []
    loads_cnt = dict(Counter(loads))
    for val, cnt in loads_cnt.items():
        last_occurrence = rindex(loads, val)
        loads[last_occurrence] += 1
        res = threshold_dp(tuple(loads), k, strategy, n=n, m=m, k=k, reward_fun=reward_fun, dict_limit=dict_limit)
        loads[last_occurrence] -= 1
        reject_avg += res * cnt
        results.append((val, cnt, res))
    reject_avg /= n
    if choices_left > 2:
        reject_avg = threshold_dp(loads_tuple, choices_left - 1, strategy, n=n, m=m, k=k, reward_fun=reward_fun,
                                  dict_limit=dict_limit)

    pref_score = 0
    pref_cnt = 0
    for (val, cnt, res) in results:
        pref_score += res * cnt
        pref_cnt += cnt
        if res >= reject_avg or len(results) == 1:  # NOTE: needed for numerical precision for small values of N
            best_threshold = val
            best_expected_score = pref_score / n + (n - pref_cnt) / n * reject_avg

    if len(strategy) < dict_limit:
        strategy[(loads_tuple, choices_left)] = (best_expected_score, best_threshold)
    else:
        print(f"WARNING! Dictionary limit {dict_limit} has been reached! It can slow down performance dramatically!")

    return best_expected_score


def find_best_strategy(n=N, m=M, k=K, reward_fun=REWARD_FUN, use_threshold_dp=True, print_behaviour=PRINT_BEHAVIOUR):
    strategy = {}
    best_expected_score = threshold_dp(tuple([0] * n), k, strategy, n=n, m=m, k=k,
                                       reward_fun=reward_fun) if use_threshold_dp \
        else old_dp(tuple([0] * n), 0, n=n, m=m, k=k, rejects_left=k - 1, reward_fun=reward_fun)
    if print_behaviour:
        print(
            f"With {n} bins and {m} balls the best achievable expected maximum score with two-thinning is {best_expected_score}")
    return strategy


def assert_monotonicity(strategy):
    """
    This confirms that the optimal thresholds for the same load with decreasing number of choices left never increases
    """
    strategy_grouped = {}
    for ((loads_tuple, choices_left), (_, best_threshold)) in strategy.items():
        if loads_tuple not in strategy_grouped:
            strategy_grouped[loads_tuple] = []
        strategy_grouped[loads_tuple].append((choices_left, best_threshold))

    for loads_tuple, l in strategy_grouped.items():
        for choices_left_a, best_threshold_a in l:
            for choices_left_b, best_threshold_b in l:
                if choices_left_a < choices_left_b and best_threshold_a < best_threshold_b:
                    return False, (loads_tuple, choices_left_a, best_threshold_a, choices_left_b, best_threshold_b)

    return True, None


def get_predecessors(loads_tuple, choices_left, strategy, n=N, k=K):
    loads = list(loads_tuple)
    unique_loads = list(Counter(loads))
    predecessors = []
    if choices_left == k:
        for prev_choices_left in range(2, k + 1):
            for val in unique_loads:
                if val == 0:
                    continue
                first_occurrence = loads.index(val)
                loads[first_occurrence] -= 1
                prev_loads_tuple = tuple(loads)
                _, prev_threshold = strategy[(prev_loads_tuple, prev_choices_left)]
                cnt_above_threshold = len([x for x in prev_loads_tuple if x > prev_threshold])
                cnt_prev = prev_loads_tuple.count(val - 1)
                if prev_choices_left == 2:
                    step_probability = cnt_prev / n + cnt_above_threshold / n * cnt_prev / n if val - 1 <= prev_threshold else \
                        cnt_above_threshold / n * cnt_prev / n
                else:
                    step_probability = cnt_prev / n if val - 1 <= prev_threshold else 0
                predecessors.append(((prev_loads_tuple, prev_choices_left), step_probability))
                loads[first_occurrence] += 1
    else:
        _, prev_threshold = strategy[(loads_tuple, choices_left + 1)]
        cnt_above_threshold = len([x for x in loads_tuple if x > prev_threshold])
        step_probability = cnt_above_threshold / n
        predecessors.append(((loads_tuple, choices_left + 1), step_probability))
    return predecessors


def find_probability_dp(loads_tuple, choices_left, reach_probabilities, strategy, n=N, k=K):
    if (loads_tuple, choices_left) in reach_probabilities:
        return reach_probabilities[(loads_tuple, choices_left)]

    if loads_tuple == tuple([0] * n) and choices_left == k:
        return 1

    p_reach = 0
    for (prev_loads_tuple, prev_choices_left), p_move in get_predecessors(loads_tuple, choices_left, strategy, n, k):
        p_reach += p_move * find_probability_dp(prev_loads_tuple, prev_choices_left, reach_probabilities, strategy, n=n, k=k)

    reach_probabilities[(loads_tuple, choices_left)] = p_reach
    return p_reach


def analyse_k(n=N, m=M, max_k=10, reward_fun=REWARD_FUN, use_threshold_dp=True, print_behaviour=False):
    plt.rcParams['font.size'] = '14'
    plt.clf()
    colors = pyl.cm.viridis(np.linspace(0, 1, max_k-1))

    smallest_achievable = ceil(m / n)
    max_load_min = smallest_achievable - 1
    max_load_max = smallest_achievable + 3
    plot_range = range(max_load_min, max_load_max + 1)

    for k in range(2, max_k + 1):
        threshold_strategy = find_best_strategy(n=n, m=m, k=k, reward_fun=reward_fun, use_threshold_dp=use_threshold_dp,
                                                print_behaviour=print_behaviour)
        reach_probabilities = {}
        for (loads_tuple, choices_left) in threshold_strategy:
            find_probability_dp(loads_tuple, choices_left, reach_probabilities, threshold_strategy, n=n, k=k)
        max_load_distribution = [0] * (m + 1)
        for (loads_tuple, choices_left), p in reach_probabilities.items():
            if sum(loads_tuple) == m:
                max_load_distribution[max(loads_tuple)] += p

        raw_max_distribution = max_load_distribution[max_load_min:(max_load_max+1)]
        # log_max_distribution = [log(x) for x in max_load_distribution[max_load_min:(max_load_max+1)]]
        # cdf = list(np.cumsum(max_load_distribution[max_load_min:(max_load_max+1)]))
        to_plot = raw_max_distribution
        plt.plot(plot_range, to_plot, label=f"k={k}", color=colors[k-2])

    plt.xlabel("maximum load")
    plt.ylabel("probability")
    plt.xticks(plot_range)
    plt.legend()
    file_name = f"k_thinning_max_load_distribution_{n}_{m}.pdf"
    save_path = join(dirname(dirname(dirname(abspath(__file__)))), "evaluation", "k_thinning", "data", file_name)
    plt.savefig(save_path)


if __name__ == "__main__":
    start_time = time.time()

    """strategy = {}
    print(threshold_dp(tuple([0] * N), K, strategy))
    print(f"With {M} balls and {N} bins the best achievable expected maximum load with {K}-thinning is {old_dp(tuple([0] * N), 0)}")
    print(assert_monotonicity(strategy))"""

    analyse_k()

    print("--- %s seconds ---" % (time.time() - start_time))
