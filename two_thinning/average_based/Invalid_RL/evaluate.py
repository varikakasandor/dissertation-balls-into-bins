from two_thinning.average_based.Invalid_RL.train import train
from two_thinning.average_based.simulation import simulate_many_runs


n = 10
m = 20
episodes = 1000
epsilon = 0.1
alpha = 0.5
version = 'Q'
reward = max

runs = 1000


def evaluate_q_values(best_thresholds, n=n, m=m, reward=reward, runs=runs):
    avg_load = simulate_many_runs(best_thresholds, reward=reward, runs=runs, n=n, m=m)
    return avg_load


def evaluate(n=n, m=m, episodes=episodes, epsilon=epsilon, alpha=alpha, version=version, reward=reward, runs=runs):
    best_thresholds = train(n=n, m=m, episodes=episodes, epsilon=epsilon, alpha=alpha, version=version, reward=reward)
    avg_load = evaluate_q_values(best_thresholds, n=n, m=m, reward=reward, runs=runs)
    print(f"With {m} balls and {n} bins the best thresholds are {best_thresholds}. Its maximum load has an average of {avg_load}.")
    return avg_load


if __name__ == "__main__":
    evaluate()
