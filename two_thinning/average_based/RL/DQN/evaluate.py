import torch
import os

from two_thinning.average_based.RL.DQN.train import train, evaluate_q_values
from two_thinning.average_based.RL.DQN.neural_network import AverageTwoThinningNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 40
CONTINUOUS_REWARD = True
TRAIN_EPISODES = 1000
TARGET_UPDATE_FREQ = 10
MEMORY_CAPACITY = 100
EVAL_RUNS = 100
PATIENCE = 1000
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = True
N = 10
M = 20


def REWARD_FUN(x):
    return -max(x)


def get_best_model_path(n=N, m=M):
    best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models", f"best_{n}_{m}.pth")
    return best_model_path


def evaluate(trained_model, n=N, m=M, reward_fun=REWARD_FUN, eval_runs=EVAL_RUNS, ):
    avg_score = evaluate_q_values(trained_model, n=n, m=m, reward=reward_fun, eval_runs=eval_runs, print_behaviour=True)
    return avg_score


def evaluate_new_model(n=N, m=M, train_episodes=TRAIN_EPISODES, memory_capacity=MEMORY_CAPACITY, eps_start=EPS_START,
                       eps_end=EPS_END, eps_decay=EPS_DECAY, reward_fun=REWARD_FUN, batch_size=BATCH_SIZE,
                       target_update_freq=TARGET_UPDATE_FREQ, continuous_reward=CONTINUOUS_REWARD,
                       eval_runs=EVAL_RUNS, patience=PATIENCE, print_progress=PRINT_PROGRESS,
                       print_behaviour=PRINT_BEHAVIOUR,
                       device=DEVICE):
    trained_model = train(n=N, m=M, memory_capacity=memory_capacity, num_episodes=train_episodes, reward_fun=reward_fun,
                          batch_size=batch_size, eps_start=eps_start, eps_end=eps_end,
                          continuous_reward=continuous_reward,
                          eps_decay=eps_decay, target_update_freq=target_update_freq, eval_runs=eval_runs,
                          patience=patience,
                          print_behaviour=print_behaviour, print_progress=print_progress, device=device)
    return evaluate(trained_model, n=n, m=m, reward_fun=reward_fun, eval_runs=eval_runs)


def load_best_model(n=N, m=M, device=DEVICE):
    best_model = AverageTwoThinningNet(m, device=device)
    best_model.load_state_dict(torch.load(get_best_model_path(n=n, m=m)))
    best_model.eval()
    return best_model


def compare(n=N, m=M, train_episodes=TRAIN_EPISODES, memory_capacity=MEMORY_CAPACITY, eps_start=EPS_START,
            eps_end=EPS_END, eps_decay=EPS_DECAY, reward_fun=REWARD_FUN, batch_size=BATCH_SIZE,
            target_update_freq=TARGET_UPDATE_FREQ, continuous_reward=CONTINUOUS_REWARD,
            eval_runs=EVAL_RUNS, patience=PATIENCE, print_progress=PRINT_PROGRESS, print_behaviour=PRINT_BEHAVIOUR,
            device=DEVICE):
    current_model = train(n=N, m=M, memory_capacity=memory_capacity, num_episodes=train_episodes, reward_fun=reward_fun,
                          batch_size=batch_size, eps_start=eps_start, eps_end=eps_end,
                          continuous_reward=continuous_reward,
                          eps_decay=eps_decay, target_update_freq=target_update_freq, eval_runs=eval_runs,
                          patience=patience,
                          print_behaviour=print_behaviour, print_progress=print_progress, device=device)
    current_model_performance = evaluate(current_model, n=n, m=m, reward_fun=reward_fun, eval_runs=eval_runs)
    print(f"With {m} balls and {n} bins the trained current DQN model has an average score/maximum load of {current_model_performance}.")

    if os.path.exists(get_best_model_path(n=n, m=m)):
        best_model = load_best_model()
        best_model_performance = evaluate(best_model, n=n, m=m, reward_fun=reward_fun, eval_runs=eval_runs)
        print(f"The average maximum load of the best model is {best_model_performance}.")
        if current_model_performance < best_model_performance:
            torch.save(current_model.state_dict(), get_best_model_path(n=n, m=m))
            print(f"The best model has been updated to the current model.")
    else:
        torch.save(current_model.state_dict(), get_best_model_path(n=n, m=m))
        print(f"This is the first model trained with these parameters. This trained model is now saved.")


if __name__ == "__main__":
    compare()
