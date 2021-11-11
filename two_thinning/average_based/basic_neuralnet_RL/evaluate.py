from two_thinning.average_based.basic_neuralnet_RL.train import train, evaluate_q_values

n = 5
m = 10
episodes = 3000
epsilon = 0.3
reward = max
eval_runs = 100
patience = 7
print_progress = True


def evaluate(n=n, m=m, episodes=episodes, epsilon=epsilon, reward=reward, eval_runs=eval_runs, patience=patience,
             print_progress=print_progress):
    best_thresholds = train(n=n, m=m, episodes=episodes, epsilon=epsilon, reward=reward, patience=patience,
                            print_progress=print_progress)
    avg_load = evaluate_q_values(best_thresholds, n=n, m=m, reward=reward, eval_runs=eval_runs, print_behaviour=True)
    print(f"With {m} balls and {n} bins the best average based model has an average maximum load of {avg_load}.")
    return avg_load


if __name__ == "__main__":
    evaluate()
