import random
from ray import tune

from two_thinning.constant_threshold.simulation import simulate_one_run

n = 10
m = 20
episodes = 10000
epsilon = 0.1

initial_value = 0
reward = max


def choose_random_min(q):
    min_val = min(q)
    random_mini = random.choice([i for i in range(len(q)) if q[i] == min_val])
    return random_mini


def train(n=n, m=m, episodes=episodes, epsilon=epsilon, initial_value=initial_value, reward=reward, use_tune=False):
    q = [initial_value] * (m + 1)
    cnt = [initial_value] * (m + 1)
    for _ in range(episodes):
        r = random.random()
        if r < epsilon:
            a = random.randrange(m + 1)
        else:  # TODO: Why does it sample randomly in this case? Why doesn't it sample in a weighted way according to the current q-values?
            a = choose_random_min(q)  # q.index(min(q))
        result = simulate_one_run(a, reward=reward, n=n, m=m)
        cnt[a] += 1
        q[a] += (result - q[a]) / cnt[a]
        if use_tune:
            tune.report(score=result)

    best_threshold = q.index(min(q))
    #for i in range(m + 1):
    #    print(f"{i}: cnt={cnt[i]}, q={q[i]}")
    return best_threshold


if __name__ == "__main__":
    train()
