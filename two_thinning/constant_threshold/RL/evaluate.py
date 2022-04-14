from two_thinning.constant_threshold.RL.train import train
from two_thinning.constant_threshold.simulation import simulate_many_runs

n = 100
m = 100
episodes = 1000
epsilon = 0.1
initial_q_value=0
reward = max

runs = episodes


def evaluate_q_value(best_threshold, n=n, m=m, reward=reward, runs=runs):
    avg_load = simulate_many_runs(best_threshold, reward=reward, runs=runs, n=n, m=m)
    return avg_load


def evaluate(n=n, m=m, episodes=episodes, epsilon=epsilon, initial_q_value=initial_q_value, reward=reward, runs=runs):
    best_threshold = train(n=n, m=m, episodes=episodes, epsilon=epsilon, initial_q_value=initial_q_value, reward_fun=reward)
    avg_load = evaluate_q_value(best_threshold, n=n, m=m, reward=reward, runs=runs)
    print(f"With {m} balls and {n} bins the best constant threshold is {best_threshold}. It produces an average maximum load of {avg_load}.")


if __name__ == "__main__":
    evaluate()
