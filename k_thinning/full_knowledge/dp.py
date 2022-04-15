import functools
import time
from collections import Counter

from helper.helper import number_of_increasing_partitions

N = 10
M = 20
K = 4
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
            reject += old_dp(tuple(loads), chosen=j, n=n, m=m, k=k, rejects_left=rejects_left - 1, reward_fun=reward_fun)
        reject /= n
    else:
        for i in range(n):
            loads[i] += 1
            loads_sorted = sorted(loads)
            for j in range(n):
                reject += old_dp(tuple(loads_sorted), chosen=j, n=n, m=m, k=k, rejects_left=k - 1, reward_fun=reward_fun)
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
        reject_avg = threshold_dp(loads_tuple, choices_left - 1, strategy, n=n, m=m, k=k, reward_fun=reward_fun, dict_limit=dict_limit)

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
    best_expected_score = threshold_dp(tuple([0] * n), k, strategy, n=n, m=m, k=k, reward_fun=reward_fun) if use_threshold_dp \
        else old_dp(tuple([0] * n), 0, n=n, m=m, k=k, rejects_left=k - 1, reward_fun=reward_fun)
    if print_behaviour:
        print(
            f"With {n} bins and {m} balls the best achievable expected maximum score with two-thinning is {best_expected_score}")
    return strategy


if __name__ == "__main__":
    start_time = time.time()

    #print(f"With {M} balls and {N} bins the best achievable expected maximum load with {K}-thinning is {old_dp(tuple([0] * N), 0)}")
    strategy = {}
    print(threshold_dp(tuple([0]*N), K, strategy))
    print(
        f"With {M} balls and {N} bins the best achievable expected maximum load with {K}-thinning is {old_dp(tuple([0] * N), 0)}")

    print("--- %s seconds ---" % (time.time() - start_time))
