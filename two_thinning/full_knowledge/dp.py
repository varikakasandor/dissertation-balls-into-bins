import functools
import time

from helper.helper import number_of_increasing_partitions
from collections import Counter

N = 5
M = 40
DICT_LIMIT = 400000  # M * N * number_of_increasing_partitions(N, M)


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


def find_best_thresholds(n=N, m=M, reward=REWARD):
    strategy = {}
    print(
        f"With {m} balls and {n} bins the best achievable expected maximum load with two-thinning is {dp(tuple([0] * n), 0, strategy, n=n, m=m, reward=reward)}")
    return strategy


def assert_monotonicity(strategy):
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


if __name__ == "__main__":
    start_time = time.time()

    strategy = find_best_thresholds()
    print(assert_monotonicity(strategy))
    print("--- %s seconds ---" % (time.time() - start_time))
