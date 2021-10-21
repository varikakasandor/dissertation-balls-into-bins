import random

from two_thinning.simulation.two_thinning_simulation import two_thinning_average_based_simulate_many_runs

n = 10
m = n
episodes = 100000
epsilon = 0.1
alpha = 0.1


def reward(loads):
    return max(loads)


def epsilon_greedy(options, epsilon=epsilon):
    r = random.random()
    if r < epsilon:
        a = random.randrange(len(options))
    else:
        a = options.index(min(options))

    return a, options[a]


def evaluate(q_values, runs=10000):
    thresholds = [q_i.index(min(q_i)) for q_i in q_values]
    avg_load = two_thinning_average_based_simulate_many_runs(thresholds, runs=runs)
    print(f"The average load of the derived greedy policy is {avg_load}")


def train(n=n, m=m, episodes=episodes, epsilon=epsilon, version="Q"):
    q = [[(m + 1)] * (i + 1) for i in range(m)]  # TODO: good initialization
    for _ in range(episodes):
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

    for i in range(m):
        print(
            f"After {i} balls have been placed, the ideal threshold is {q[i].index(min(q[i]))} with an expected maximum load of {min(q[i])}")

    evaluate(q)


if __name__ == "__main__":
    train()
