import copy
import random
import time
from math import log, ceil
from os import mkdir

import numpy as np
import torch.optim as optim
import wandb
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale

from helper.replay_memory import ReplayMemory
from k_choice.simulation import sample_one_choice
from two_thinning.full_knowledge.RL.DQN.curriculum_learning.constants import *
from two_thinning.full_knowledge.RL.DQN.train import epsilon_greedy, optimize_model, evaluate_q_values_faster, \
    analyse_threshold_progression


def train(n=N, m=M, memory_capacity=MEMORY_CAPACITY, reward_fun=REWARD_FUN,
          batch_size=BATCH_SIZE, eps_start=EPS_START, eps_end=EPS_END, report_wandb=False, lr=LR, pacing_fun=PACING_FUN,
          eps_decay=EPS_DECAY, optimise_freq=OPTIMISE_FREQ, target_update_freq=TARGET_UPDATE_FREQ,
          nn_hidden_size=NN_HIDDEN_SIZE, nn_rnn_num_layers=NN_RNN_NUM_LAYERS, nn_num_lin_layers=NN_NUM_LIN_LAYERS,
          eval_runs=EVAL_RUNS_TRAIN, patience=PATIENCE, potential_fun=POTENTIAL_FUN, loss_function=LOSS_FUCNTION,
          max_threshold=MAX_THRESHOLD, eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE, save_path=SAVE_PATH,
          print_progress=PRINT_PROGRESS, nn_model=NN_MODEL, device=DEVICE):
    start_time = time.time()
    mkdir(save_path)
    max_possible_load = m // n + ceil(sqrt(
        log(n))) if nn_model == FullTwoThinningClippedRecurrentNetFC else m  # based on the two-thinning paper, this can be achieved!

    policy_net = nn_model(n=n, max_threshold=max_threshold, max_possible_load=max_possible_load,
                          hidden_size=nn_hidden_size, rnn_num_layers=nn_rnn_num_layers,
                          num_lin_layers=nn_num_lin_layers,
                          device=device)
    target_net = nn_model(n=n, max_threshold=max_threshold, max_possible_load=max_possible_load,
                          hidden_size=nn_hidden_size, rnn_num_layers=nn_rnn_num_layers,
                          num_lin_layers=nn_num_lin_layers,
                          device=device)
    best_net = nn_model(n=n, max_threshold=max_threshold, max_possible_load=max_possible_load,
                        hidden_size=nn_hidden_size, rnn_num_layers=nn_rnn_num_layers, num_lin_layers=nn_num_lin_layers,
                        device=device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(memory_capacity)

    steps_done = 0
    best_eval_score = None
    not_improved = 0
    threshold_jumps = []
    eval_scores = []

    for start_size in reversed(range(m)):
        for ep in range(pacing_fun(start_size)):
            loads = sample_one_choice(n=n, m=start_size)
            for i in range(start_size, m):
                threshold = epsilon_greedy(policy_net=policy_net, loads=loads, max_threshold=max_threshold,
                                           steps_done=steps_done,
                                           eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, device=device)
                randomly_selected = random.randrange(n)
                to_place = randomly_selected if loads[randomly_selected] <= threshold.item() else random.randrange(n)
                curr_state = copy.deepcopy(loads)
                loads[to_place] += 1

                next_state = copy.deepcopy(loads)
                reward = reward_fun(next_state) if i == m - 1 else 0  # "real" reward
                reward += potential_fun(next_state) - potential_fun(curr_state)
                reward = torch.DoubleTensor([reward]).to(device)
                memory.push(curr_state, threshold, next_state, reward, i == m - 1)

                steps_done += 1

                if steps_done % optimise_freq == 0:
                    optimize_model(memory=memory, policy_net=policy_net, target_net=target_net, optimizer=optimizer,
                                   batch_size=batch_size, criterion=loss_function,
                                   device=device)

            curr_eval_score = evaluate_q_values_faster(policy_net, n=n, m=m, reward=reward_fun, eval_runs=eval_runs,
                                                       batch_size=eval_parallel_batch_size)
            if best_eval_score is None or curr_eval_score > best_eval_score:
                curr_eval_score = evaluate_q_values_faster(policy_net, n=n, m=m, reward=reward_fun,
                                                           eval_runs=5 * eval_runs,
                                                           batch_size=eval_parallel_batch_size)  # only update the best if it is really better, so run more tests
            if report_wandb:
                wandb.log({"score": curr_eval_score})

            eval_scores.append(curr_eval_score)
            threshold_jumps.append(analyse_threshold_progression(policy_net, ep, save_path, delta=max_threshold // 2))

            if best_eval_score is None or curr_eval_score > best_eval_score:
                best_eval_score = curr_eval_score
                best_net.load_state_dict(policy_net.state_dict())
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

            if ep % target_update_freq == 0:  # TODO: decouple target update and optional user halting
                """user_text, timed_out = timedInput(prompt="Press Y if you would like to stop the training now!\n", timeout=2)

                if not timed_out and user_text == "Y":
                    print("Training has been stopped by the user.")
                    return best_net
                else:
                    if not timed_out:
                        print("You pressed the wrong button, it has no effect. Training continues.")"""
                target_net.load_state_dict(policy_net.state_dict())

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
    return best_net

if __name__ == "__main__":
    train()