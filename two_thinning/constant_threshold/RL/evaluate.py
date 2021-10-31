from two_thinning.constant_threshold.RL.train import train
from two_thinning.constant_threshold.simulation import simulate_many_runs

n = 10
m = n
episodes = 100000
epsilon = 0.1
reward = max

runs = 10000


def evaluate_q_value(best_threshold, n=n, m=m, reward=reward, runs=runs):
    avg_load = simulate_many_runs(best_threshold, reward=reward, runs=runs, n=n, m=m)
    print(f"With {m} balls and {n} bins the average maximum load of the derived greedy constant threshold policy is {avg_load}")


def evaluate(n=n, m=m, episodes=episodes, epsilon=epsilon, reward=reward, runs=runs):
    best_threshold = train(n=n, m=m, episodes=episodes, epsilon=epsilon, reward=reward)
    evaluate_q_value(best_threshold, n=n, m=m, reward=reward, runs=runs)


if __name__ == "__main__":
    evaluate()
