import time

from helper.helper import number_of_increasing_partitions
from collections import Counter

N = 5
M = 20
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


def find_best_thresholds(n=N, m=M, reward=REWARD, print_behaviour=PRINT_BEHAVIOUR):
    strategy = {}
    best_expected_score = dp(tuple([0] * n), 0, strategy, n=n, m=m, reward=reward)
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


def strict_subset(l1, l2):
    not_equal = False
    for (x, y) in zip(l1, l2):
        if x > y:
            return False
        if x < y:
            not_equal = True
    return not_equal


def assert_monotonicity(strategy):
    for ((loads, val), (_, decision)) in strategy.items():
        if decision != -1:
            continue
        for ((other_loads, other_val), (_, other_decision)) in strategy.items():
            if other_decision != -1 and other_val <= val and strict_subset(loads, other_loads):
                return False, ((loads, val, decision), (other_loads, other_val, other_decision))
    return True, None


if __name__ == "__main__":
    start_time = time.time()

    """for m in range(1, M):
        strategy = find_best_thresholds(n=3, m=m, reward=REWARD, print_behaviour=False)
        threshold_logic = assert_threshold_logic(strategy)
        monotone = assert_monotonicity(strategy)
        if not threshold_logic or not monotone:
            print(3, m)"""
    strategy = find_best_thresholds(n=N, m=M, reward=REWARD)
    for ((loads, val), (_, decision)) in strategy.items():
        if decision == 0:
            print(loads, val)
    print("--- %s seconds ---" % (time.time() - start_time))
