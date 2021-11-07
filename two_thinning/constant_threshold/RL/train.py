import random
from ray import tune

from two_thinning.constant_threshold.simulation import simulate_one_run

n = 2
m = 1000
episodes = 10000
epsilon = 0.1

reward = max


def train(n=n, m=m, episodes=episodes, epsilon=epsilon, reward=reward, use_tune=False):
    q = [0] * (m + 1)
    cnt = [0] * (m + 1)
    for _ in range(episodes):
        r = random.random()
        if r < epsilon:
            a = random.randrange(m + 1)
        else:
            a = q.index(min(q))
        result = simulate_one_run(a, reward=reward, n=n, m=m)
        cnt[a] += 1
        q[a] += (result - q[a]) / cnt[a]
        if use_tune:
            tune.report(score=result)

    best_threshold = q.index(min(q))
    #print(f"The best threshold is {best_threshold}, producing on average a maximum load of {q[best_threshold]}")
    return best_threshold


if __name__ == "__main__":
    train()
