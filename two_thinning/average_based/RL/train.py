import random
from ray import tune

from two_thinning.average_based.simulation import simulate_many_runs


n = 5
m = 5
episodes = 100000
epsilon = 0.1
alpha = 0.1
version = 'Q'
test_runs=300
reward = max


def epsilon_greedy(options, epsilon=epsilon):
    r = random.random()
    if r < epsilon:
        a = random.randrange(len(options))
    else:
        a = options.index(min(options))

    return a, options[a]


def train(n=n, m=m, episodes=episodes, epsilon=epsilon, alpha=alpha, version=version, reward=reward, test_runs=test_runs, use_tune=False):
    q = [[(m + 1)] * (i + 1) for i in range(m)]  # TODO: good initialization
    best_thresholds=None
    best_avg_test_load=None
    for ep in range(episodes):
        loads = [0] * n
        for i in range(m):
            a, _ = epsilon_greedy(q[i], epsilon)
            randomly_selected = random.randrange(n)
            if loads[randomly_selected] <= a:
                loads[randomly_selected] += 1
            else:
                loads[random.randrange(n)] += 1

            if i == m - 1:
                q[i][a] += alpha * (reward(loads) - q[i][a])
            else:
                if version == "Q":
                    q[i][a] += alpha * (min(q[i + 1]) - q[i][a])
                elif version == "Sarsa":
                    _, v = epsilon_greedy(q[i + 1], epsilon)
                    q[i][a] += alpha * (v - q[i][a])
                else:
                    raise NotImplementedError(version)

        curr_thresholds=[q[i].index(min(q[i])) for i in range(m)]
        avg_test_load = simulate_many_runs(curr_thresholds, reward=reward, runs=test_runs, n=n, m=m)
        if best_avg_test_load is None or avg_test_load<best_avg_test_load:
            best_avg_test_load=avg_test_load
            best_thresholds=curr_thresholds
        if use_tune:
            tune.report(avg_test_load=avg_test_load)

    return best_thresholds


if __name__ == "__main__":
    train()
