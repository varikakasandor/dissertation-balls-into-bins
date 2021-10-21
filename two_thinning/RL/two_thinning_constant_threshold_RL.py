import random

from two_thinning.simulation.two_thinning_simulation import two_thinning_constant_threshold_simulate_one_run

n = 2
m = 1000
episodes = 10000
epsilon = 0.1


def train(n=n, m=m, episodes=episodes, epsilon=epsilon):
    q = [0] * (m + 1)
    cnt = [0] * (m + 1)
    for _ in range(episodes):
        r = random.random()
        if r < epsilon:
            a = random.randrange(m + 1)
        else:
            a = q.index(min(q))
        r = two_thinning_constant_threshold_simulate_one_run(a, n, m)
        cnt[a] += 1
        q[a] += (r - q[a]) / cnt[a]

    best_threshold = q.index(min(q))
    print(f"The best threshold is {best_threshold}, producing on average a maximum load of {q[best_threshold]}")


if __name__ == "__main__":
    train()
