import torch
import os

from two_thinning.full_knowledge.RL.DQN.train import train, evaluate_q_values
from two_thinning.full_knowledge.RL.DQN.neural_network import FullTwoThinningNet, FullTwoThinningOneHotNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
CONTINUOUS_REWARD = True
TRAIN_EPISODES = 10000
TARGET_UPDATE_FREQ = 10
MEMORY_CAPACITY = 1000
EVAL_RUNS = 1000
PATIENCE = 100
MAX_LOAD_INCREASE_REWARD = -1  # TODO: -1 is the realistic one, but maybe other values work better
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = True
N = 30
M = 150
MAX_THRESHOLD = max(3, 2 * M // N)  # TODO: find some mathematical bound which is provable
MAX_WEIGHT = 1000



def REWARD_FUN(x):
    return -max(x)


def get_best_model_path(n=N, m=M, one_hot=True):
    one_hot_string = "one_hot_" if one_hot else ""
    best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models", f"best_{one_hot_string}{n}_{m}.pth")
    return best_model_path


def evaluate(trained_model, n=N, m=M, reward_fun=REWARD_FUN, eval_runs=EVAL_RUNS, ):
    avg_score = evaluate_q_values(trained_model, n=n, m=m, reward=reward_fun, eval_runs=eval_runs, print_behaviour=False) # TODO: set back print_behaviour to True
    return avg_score


def evaluate_new_model(n=N, m=M, train_episodes=TRAIN_EPISODES, memory_capacity=MEMORY_CAPACITY, eps_start=EPS_START,
                       eps_end=EPS_END, eps_decay=EPS_DECAY, reward_fun=REWARD_FUN, batch_size=BATCH_SIZE,
                       target_update_freq=TARGET_UPDATE_FREQ, continuous_reward=CONTINUOUS_REWARD, eval_runs=EVAL_RUNS,
                       patience=PATIENCE, max_load_increase_reward=MAX_LOAD_INCREASE_REWARD,
                       max_threshold=MAX_THRESHOLD, max_weight=MAX_WEIGHT,
                       print_progress=PRINT_PROGRESS, print_behaviour=PRINT_BEHAVIOUR, device=DEVICE):
    trained_model = train(n=N, m=M, memory_capacity=memory_capacity, num_episodes=train_episodes, reward_fun=reward_fun,
                          batch_size=batch_size, eps_start=eps_start, eps_end=eps_end,
                          continuous_reward=continuous_reward, max_threshold=max_threshold,
                          eps_decay=eps_decay, target_update_freq=target_update_freq, eval_runs=eval_runs,
                          patience=patience, max_load_increase_reward=max_load_increase_reward, max_weight=max_weight,
                          print_behaviour=print_behaviour, print_progress=print_progress, device=device)
    return evaluate(trained_model, n=n, m=m, reward_fun=reward_fun, eval_runs=eval_runs)


def load_best_model(n=N, m=M, device=DEVICE):
    for max_threshold in range(m + 1):
        try:
            best_model = FullTwoThinningOneHotNet(n=n, max_threshold=max_threshold, max_possible_load=m, device=device)
            best_model.load_state_dict(torch.load(get_best_model_path(n=n, m=m)))
            best_model.eval()
            return best_model
        except:
            continue
    print("ERROR: trained model not found with any max_threshold")



def compare(n=N, m=M, train_episodes=TRAIN_EPISODES, memory_capacity=MEMORY_CAPACITY, eps_start=EPS_START,
            eps_end=EPS_END, eps_decay=EPS_DECAY, reward_fun=REWARD_FUN, batch_size=BATCH_SIZE,
            target_update_freq=TARGET_UPDATE_FREQ, continuous_reward=CONTINUOUS_REWARD, max_threshold=MAX_THRESHOLD,
            eval_runs=EVAL_RUNS, patience=PATIENCE, max_load_increase_reward=MAX_LOAD_INCREASE_REWARD, max_weight=MAX_WEIGHT,
            print_progress=PRINT_PROGRESS, print_behaviour=PRINT_BEHAVIOUR, device=DEVICE):
    current_model = train(n=N, m=M, memory_capacity=memory_capacity, num_episodes=train_episodes, reward_fun=reward_fun,
                          batch_size=batch_size, eps_start=eps_start, eps_end=eps_end,
                          continuous_reward=continuous_reward, max_threshold=max_threshold,
                          eps_decay=eps_decay, target_update_freq=target_update_freq, eval_runs=eval_runs,
                          patience=patience, max_load_increase_reward=max_load_increase_reward, max_weight=max_weight,
                          print_behaviour=print_behaviour, print_progress=print_progress, device=device)
    current_model_performance = evaluate(current_model, n=n, m=m, reward_fun=reward_fun, eval_runs=eval_runs)
    print(
        f"With {m} balls and {n} bins the trained current DQN model has an average score/maximum load of {current_model_performance}.")

    if os.path.exists(get_best_model_path(n=n, m=m)):
        best_model = load_best_model(n=n, m=m, device=device)
        best_model_performance = evaluate(best_model, n=n, m=m, reward_fun=reward_fun, eval_runs=eval_runs)
        print(f"The average maximum load of the best model is {best_model_performance}.")
        if current_model_performance > best_model_performance:
            torch.save(current_model.state_dict(), get_best_model_path(n=n, m=m))
            print(f"The best model has been updated to the current model.")
        else:
            print(f"The best model had better performance than the current one, so it is not updated.")
    else:
        torch.save(current_model.state_dict(), get_best_model_path(n=n, m=m))
        print(f"This is the first model trained with these parameters. This trained model is now saved.")


if __name__ == "__main__":
    compare()
