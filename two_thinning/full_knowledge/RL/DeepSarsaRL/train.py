import copy
import random
import time
from math import *
from os import mkdir

import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale
from torch import optim

from two_thinning.full_knowledge.RL.DQN.train import evaluate_q_values_faster, analyse_threshold_progression
from two_thinning.full_knowledge.RL.DeepSarsaRL.constants import *


def epsilon_greedy(model, loads, max_threshold, steps_done, eps_start, eps_end, eps_decay, device):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * exp(-1. * steps_done / eps_decay)
    action_values = model(torch.tensor(loads).unsqueeze(0)).squeeze(0)
    if sample > eps_threshold:
        a = torch.argmax(action_values)
    else:
        a = torch.randint(max_threshold + 1, (1,))[0]
    return a.to(device), action_values[a]


def train(n=N, m=M, num_episodes=TRAIN_EPISODES, reward_fun=REWARD_FUN,
          eps_start=EPS_START, eps_end=EPS_END, report_wandb=False, lr=LR,
          eps_decay=EPS_DECAY, nn_hidden_size=NN_HIDDEN_SIZE, nn_rnn_num_layers=NN_RNN_NUM_LAYERS, nn_num_lin_layers=NN_NUM_LIN_LAYERS,
          eval_runs=EVAL_RUNS_TRAIN, patience=PATIENCE, potential_fun=POTENTIAL_FUN, loss_function=LOSS_FUCNTION,
          max_threshold=MAX_THRESHOLD, eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE, save_path=SAVE_PATH,
          print_progress=PRINT_PROGRESS, nn_model=NN_MODEL, device=DEVICE):
    start_time = time.time()

    mkdir(save_path)

    max_possible_load = m // n + ceil(sqrt(
        log(n))) if nn_model == FullTwoThinningClippedRecurrentNetFC else m  # based on the two-thinning paper, this
    # can be achieved!

    model = nn_model(n=n, max_threshold=max_threshold, max_possible_load=max_possible_load,
                          hidden_size=nn_hidden_size, rnn_num_layers=nn_rnn_num_layers,
                          num_lin_layers=nn_num_lin_layers,
                          device=device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    steps_done = 0
    best_eval_score = None
    not_improved = 0
    threshold_jumps = []
    eval_scores = []

    for ep in range(num_episodes):
        loads = [0] * n
        for i in range(m):
            threshold, old_val = epsilon_greedy(model=model, loads=loads, max_threshold=max_threshold,
                                       steps_done=steps_done,
                                       eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, device=device)
            randomly_selected = random.randrange(n)
            to_place = randomly_selected if loads[randomly_selected] <= threshold.item() else random.randrange(n)
            curr_state = copy.deepcopy(loads)
            loads[to_place] += 1
            next_state = copy.deepcopy(loads)

            new_val = torch.tensor(potential_fun(next_state) - potential_fun(curr_state), dtype=torch.float64).to(device)
            if i == m - 1:
                new_val += reward_fun(next_state)
            else:
                with torch.no_grad():
                    _, next_val = epsilon_greedy(model=model, loads=loads, max_threshold=max_threshold,
                                       steps_done=steps_done,
                                       eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, device=device)
                    new_val += next_val

            loss = loss_function(old_val, new_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps_done += 1



        curr_eval_score = evaluate_q_values_faster(model, n=n, m=m, reward=reward_fun, eval_runs=eval_runs,
                                                   batch_size=eval_parallel_batch_size)
        if best_eval_score is None or curr_eval_score > best_eval_score:
            curr_eval_score = evaluate_q_values_faster(model, n=n, m=m, reward=reward_fun, eval_runs=5 * eval_runs,
                                                       batch_size=eval_parallel_batch_size)
        if report_wandb:
            wandb.log({"score": curr_eval_score})

        eval_scores.append(curr_eval_score)
        threshold_jumps.append(analyse_threshold_progression(model, ep, save_path, delta=max_threshold // 2))

        if best_eval_score is None or curr_eval_score > best_eval_score:
            best_eval_score = curr_eval_score
            not_improved = 0
            if print_progress:
                print(f"At episode {ep} the best eval score has improved to {curr_eval_score}.")
        elif not_improved < patience:
            not_improved += 1
            if print_progress:
                print(f"At episode {ep} no improvement has happened ({curr_eval_score}).")
        else:
            if print_progress:
                print(f"Training has stopped after episode {ep} as the eval score didn't improve anymore.")
            break

    scaled_threshold_jumps = scale(np.array(threshold_jumps))
    scaled_eval_scores = scale(np.array(eval_scores))

    plt.clf()
    plt.plot(list(range(len(scaled_threshold_jumps))), scaled_threshold_jumps,
             label="normalised number of threshold jumps")
    plt.plot(list(range(len(scaled_eval_scores))), scaled_eval_scores, label="normalised evaluation scores")
    plt.title(f"Training Progression")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(join(save_path, "training_progression.png"))
    print(f"--- {(time.time() - start_time)} seconds ---")
    return model

if __name__ == "__main__":
    train()
