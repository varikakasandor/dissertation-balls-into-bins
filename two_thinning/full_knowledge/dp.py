import time
from math import log2

from helper.helper import number_of_increasing_partitions
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import entropy

N = 5
M = 25
DICT_LIMIT = 400000  # M * N * number_of_increasing_partitions(N, M)
PRINT_BEHAVIOUR = True


def REWARD(loads):
    return -max(loads)


def rindex(l, elem):
    return len(l) - next(i for i, v in enumerate(reversed(l), 1) if v == elem)


def dp(loads_tuple, chosen_load, strategy, n=N, m=M, reward=REWARD, dict_limit=DICT_LIMIT):
    if (loads_tuple, chosen_load) in strategy:
        val, _ = strategy[(loads_tuple, chosen_load)]
        return val

    loads = list(loads_tuple)
    if sum(loads) == m:
        return reward(loads)

    last_occurence = rindex(loads, chosen_load)

    accept = 0
    loads[last_occurence] += 1
    loads_cnt = dict(Counter(loads))
    for val, cnt in loads_cnt.items():
        accept += cnt * dp(tuple(loads), val, strategy, n=n, m=m, reward=reward)
    loads[last_occurence] -= 1
    accept /= n

    reject = 0
    loads_cnt1 = dict(Counter(loads))
    for val1, cnt1 in loads_cnt1.items():
        last_occurence = rindex(loads, val1)
        loads[last_occurence] += 1
        loads_cnt2 = dict(Counter(loads))
        for val2, cnt2 in loads_cnt2.items():
            reject += cnt1 * cnt2 * dp(tuple(loads), val2, strategy, n=n, m=m, reward=reward)
        loads[last_occurence] -= 1
    reject /= (n * n)

    val = max(accept, reject)
    if len(strategy) < dict_limit:
        decision = -1 if (accept > reject) else (1 if reject > accept else 0)
        strategy[(loads_tuple, chosen_load)] = (val, decision)

    return val


def threshold_dp(loads_tuple, strategy, n=N, m=M, reward=REWARD, dict_limit=DICT_LIMIT):
    if loads_tuple in strategy:
        best_expected_score, _ = strategy[loads_tuple]
        return best_expected_score

    loads = list(loads_tuple)
    if sum(loads) == m:
        final_reward = reward(loads)
        strategy[loads_tuple] = (final_reward, m)  # NOTE: not necessarily needed for this,
        # but it is needed for the probability distribution analysis. Also "m" is arbitrary.
        return final_reward

    avg = 0
    results = []
    loads_cnt = dict(Counter(loads))
    for val, cnt in loads_cnt.items():
        last_occurrence = rindex(loads, val)
        loads[last_occurrence] += 1
        res = threshold_dp(tuple(loads), strategy, n=n, m=m, reward=reward, dict_limit=dict_limit)
        avg += res * cnt
        results.append((val, cnt, res))
        loads[last_occurrence] -= 1
    avg /= n

    pref_score = 0
    pref_cnt = 0
    for (val, cnt, res) in (results):
        pref_score += res * cnt
        pref_cnt += cnt
        if res >= avg or len(results) == 1:  # NOTE: needed for numerical precision for small values of N
            best_threshold = val
            best_expected_score = pref_score / n + (n - pref_cnt) / n * avg

    if len(strategy) < dict_limit:
        strategy[loads_tuple] = (best_expected_score, best_threshold)
    else:
        print(f"WARNING! Dictionary limit {dict_limit} has been reached! It can slow down performance dramatically!")

    return best_expected_score


def find_best_strategy(n=N, m=M, reward=REWARD, use_threshold_dp=True, print_behaviour=PRINT_BEHAVIOUR):
    strategy = {}
    best_expected_score = threshold_dp(tuple([0] * n), strategy, n=n, m=m, reward=reward) if use_threshold_dp \
        else dp(tuple([0] * n), 0, strategy, n=n, m=m, reward=reward)
    if print_behaviour:
        print(
            f"With {n} bins and {m} balls the best achievable expected maximum score with two-thinning is {best_expected_score}")
    return strategy


def assert_threshold_logic(strategy):
    for ((loads, val), (_, decision)) in strategy.items():
        if decision != 1:
            continue
        for other_val in loads:
            if other_val <= val:
                continue
            _, other_decision = strategy[(loads, other_val)]
            if other_decision != 1:
                return False, (loads, (val, other_val))
    return True, None


def subset(l1, l2):
    for (x, y) in zip(l1, l2):
        if x > y:
            return False
    return True


def assert_monotonicity(deciding_strategy):
    # NOTE: this doesn't work with the threshold_strategy, as that is not strictly increasing due to
    # using only the values that are present in the load vector
    for ((loads, val), (_, decision)) in deciding_strategy.items():
        if decision != -1:
            continue
        for ((other_loads, other_val), (_, other_decision)) in deciding_strategy.items():
            if other_decision != -1 and other_val <= val and subset(loads, other_loads):
                return False, ((loads, val, decision), (other_loads, other_val, other_decision))
    return True, None


def get_predecessors(loads_tuple, strategy, n=N):
    loads = list(loads_tuple)
    unique_loads = list(Counter(loads))
    predecessors = []
    for val in unique_loads:
        if val == 0:
            continue
        first_occurrence = loads.index(val)
        loads[first_occurrence] -= 1
        prev_loads_tuple = tuple(loads)
        _, prev_threshold = strategy[prev_loads_tuple]
        cnt_above_threshold = len([x for x in prev_loads_tuple if x > prev_threshold])
        cnt_prev = prev_loads_tuple.count(val - 1)
        step_probability = cnt_prev / n + cnt_above_threshold / n * cnt_prev / n if val - 1 <= prev_threshold else \
            cnt_above_threshold / n * cnt_prev / n
        predecessors.append((prev_loads_tuple, step_probability))
        loads[first_occurrence] += 1
    return predecessors


def find_probability_dp(loads_tuple, reach_probabilities, strategy, n=N):
    if loads_tuple in reach_probabilities:
        return reach_probabilities[loads_tuple]

    if loads_tuple == tuple([0] * n):
        return 1

    p_reach = 0
    for prev_loads_tuple, p_move in get_predecessors(loads_tuple, strategy, n):
        p_reach += p_move * find_probability_dp(prev_loads_tuple, reach_probabilities, strategy, n=n)

    reach_probabilities[loads_tuple] = p_reach
    return p_reach


def analyse_probability_distribution(threshold_strategy, include_intermediate_states=True, n=N, m=M):
    reach_probabilities = {}
    for loads_tuple in threshold_strategy:
        find_probability_dp(loads_tuple, reach_probabilities, threshold_strategy, n=n)

    max_likelihood_final_prob = 0
    final_probs = []
    for loads_tuple, p in reach_probabilities.items():
        if sum(loads_tuple) == m:
            final_probs.append(p)
            if p > max_likelihood_final_prob:
                max_likelihood_final_prob = p
                max_likelihood_final_load = loads_tuple

    print(
        f"The maximum likelihood final load vector is {max_likelihood_final_load} with probability {max_likelihood_final_prob}")
    value, counts = np.unique(final_probs, return_counts=True)

    print(
        f"There are {len(final_probs)} reachable final load vectors, its base 2 logarithm is {log2(len(final_probs))}")

    final_probs_entropy = entropy(final_probs, base=2)
    print(f"The entropy of the final load vectors is {final_probs_entropy}")

    all_probs = list(reach_probabilities.values())
    if include_intermediate_states:
        plt.hist(all_probs, bins=100)
    else:
        plt.hist(final_probs, bins=100)
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    strategy = find_best_strategy()
    analyse_probability_distribution(strategy, include_intermediate_states=False)
    print("--- %s seconds ---" % (time.time() - start_time))
