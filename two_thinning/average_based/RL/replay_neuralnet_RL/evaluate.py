import torch

from two_thinning.average_based.replay_neuralnet_RL.train import train, evaluate_q_values

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 20
TRAIN_EPISODES = 100
TARGET_UPDATE_FREQ = 10
MEMORY_CAPACITY = 10000
EVAL_RUNS = 100
PATIENCE = 60
PRINT_BEHAVIOUR = False
PRINT_PROGRESS = True
N = 20
M = 40


def REWARD_FUN(x):
    return -max(x)


def evaluate(n=N, m=M, train_episodes=TRAIN_EPISODES, memory_capacity=MEMORY_CAPACITY, eps_start=EPS_START,
             eps_end=EPS_END,
             eps_decay=EPS_DECAY, reward_fun=REWARD_FUN, batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ,
             eval_runs=EVAL_RUNS, patience=PATIENCE, print_progress=PRINT_PROGRESS, print_behaviour=PRINT_BEHAVIOUR,
             device=DEVICE):
    trained_model = train(n=N, m=M, memory_capacity=memory_capacity, num_episodes=train_episodes, reward_fun=reward_fun,
                          batch_size=batch_size, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay,
                          target_update_freq=target_update_freq, eval_runs=eval_runs, patience=patience,
                          print_behaviour=print_behaviour, print_progress=print_progress, device=device)
    avg_score = evaluate_q_values(trained_model, n=n, m=m, reward=reward_fun, eval_runs=eval_runs, print_behaviour=True)
    print(f"With {m} balls and {n} bins the trained DQN model has an average score/maximum load of {avg_score}.")
    return avg_score


if __name__ == "__main__":
    evaluate()
