from two_thinning.average_based.RL.train import train
from two_thinning.average_based.simulation import simulate_many_runs

n = 20
m = n
episodes = 100000
epsilon = 0.1
alpha = 0.1
version = 'Q'
reward = max

runs = 10000


def evaluate_q_values(best_thresholds, n=n, m=m, reward=reward, runs=runs):
    avg_load = simulate_many_runs(best_thresholds, reward=reward, runs=runs, n=n, m=m)
    print(f"With {m} balls and {n} bins the average maximum load of the derived average based greedy policy is {avg_load}")


def evaluate(n=n, m=m, episodes=episodes, epsilon=epsilon, alpha=alpha, version=version, reward=reward, runs=runs):
    best_thresholds = train(n=n, m=m, episodes=episodes, epsilon=epsilon, alpha=alpha, version=version, reward=reward)
    evaluate_q_values(best_thresholds, n=n, m=m, reward=reward, runs=runs)


if __name__ == "__main__":
    evaluate()
