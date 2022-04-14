import random
from ray import tune

from two_thinning.constant_threshold.simulation import simulate_one_run

N = 10
M = 20
EPISODES = 10000
EPSILON = 0.1
PRIMARY_ONLY = True

INITIAL_Q_VALUE = 0
def REWARD_FUN(loads):
    return -max(loads)


def choose_random_max(q):
    max_val = max(q)
    random_maxi = random.choice([i for i in range(len(q)) if q[i] == max_val])
    return random_maxi


def train(n=N, m=M, episodes=EPISODES, epsilon=EPSILON, initial_q_value=INITIAL_Q_VALUE, primary_only=PRIMARY_ONLY, reward_fun=REWARD_FUN, use_tune=False):
    # TODO: Strangely, for m=n, it always trains to output the constant threshold 1. \ I wonder why it doesn't
    #  increase the threshold as n grows larger and larger. The probability of having 1 balls in each bin \ decreases
    #  (very sharply!) by n, so it should be worth making the threshold equal to 2, what do I miss?
    q = [initial_q_value] * (m + 1)
    cnt = [0] * (m + 1)
    for _ in range(episodes):
        r = random.random()
        if r < epsilon:
            a = random.randrange(m + 1)
        else:  # TODO: Why does it sample randomly in this case? Why doesn't it sample in a weighted way according to
            # the current q-values?
            a = choose_random_max(q)  # q.index(min(q))
        result = simulate_one_run(a, reward=reward_fun, n=n, m=m, primary_only=primary_only)
        cnt[a] += 1
        q[a] += (result - q[a]) / cnt[a]
        if use_tune:
            tune.report(score=result)

    best_threshold = q.index(max(q))
    #for i in range(m + 1):
    #    print(f"{i}: cnt={cnt[i]}, q={q[i]}")
    return best_threshold


if __name__ == "__main__":
    train()
