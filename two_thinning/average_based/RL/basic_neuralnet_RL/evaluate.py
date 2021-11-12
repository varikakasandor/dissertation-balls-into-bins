import torch
import numpy as np

from two_thinning.average_based.RL.basic_neuralnet_RL.train import train, evaluate_q_values

n = 10
m = n

epsilon = 0.1
train_episodes = 3000
eval_runs = 300
patience = 20
print_progress = True
print_behaviour = False


def reward(x):
    return -np.max(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(n=n, m=m, train_episodes=train_episodes, epsilon=epsilon, reward=reward, eval_runs=eval_runs, patience=patience,
             print_progress=print_progress, print_behaviour=print_behaviour):
    best_thresholds = train(n=n, m=m, epsilon=epsilon, reward=reward, episodes=train_episodes, eval_runs=eval_runs,
                            patience=patience, print_progress=print_progress, print_behaviour=print_behaviour, device=device)
    avg_score = evaluate_q_values(best_thresholds, n=n, m=m, reward=reward, eval_runs=eval_runs, print_behaviour=True)
    print(f"With {m} balls and {n} bins the best average based model has an average score/maximum load of {avg_score}.")
    return avg_score


if __name__ == "__main__":
    evaluate()
