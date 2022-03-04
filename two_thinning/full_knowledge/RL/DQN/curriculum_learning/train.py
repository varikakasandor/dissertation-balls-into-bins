import copy
import random
import time
from math import log, ceil

import torch.optim as optim

from helper.replay_memory import ReplayMemory
from k_choice.simulation import sample_one_choice
from two_thinning.full_knowledge.RL.DQN.curriculum_learning.constants import *
from two_thinning.full_knowledge.RL.DQN.train import epsilon_greedy, optimize_model, evaluate_q_values_faster


def train(n=N, m=M, memory_capacity=MEMORY_CAPACITY, num_episodes=TRAIN_EPISODES, reward_fun=REWARD_FUN,
          batch_size=BATCH_SIZE, eps_start=EPS_START, eps_end=EPS_END,
          eps_decay=EPS_DECAY, optimise_freq=OPTIMISE_FREQ, target_update_freq=TARGET_UPDATE_FREQ,
          eval_runs=EVAL_RUNS_TRAIN, patience=PATIENCE, potential_fun=POTENTIAL_FUN,
          max_threshold=MAX_THRESHOLD, eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE,
          print_progress=PRINT_PROGRESS, nn_model=NN_MODEL, device=DEVICE):
    start_time = time.time()

    policy_net = nn_model(n=n, max_threshold=max_threshold, max_possible_load=m // n + ceil(sqrt(log(n))),
                          device=device)  # based on the two-thinning paper, this can be achieved!
    target_net = nn_model(n=n, max_threshold=max_threshold, max_possible_load=m // n + ceil(sqrt(log(n))),
                          device=device)
    best_net = nn_model(n=n, max_threshold=max_threshold, max_possible_load=m // n + ceil(sqrt(log(n))), device=device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(memory_capacity)

    steps_done = 0
    best_eval_score = None
    not_improved = 0

    def epochs_given_start_size(start_size):
        return num_episodes // m

    for start_size in reversed(range(m)):
        for ep in range(epochs_given_start_size(start_size)):
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
                                   batch_size=batch_size,
                                   device=device)


            curr_eval_score = evaluate_q_values_faster(policy_net, n=n, m=m, reward=reward_fun, eval_runs=eval_runs,
                                                       batch_size=eval_parallel_batch_size)
            if best_eval_score is None or curr_eval_score > best_eval_score:
                curr_eval_score = evaluate_q_values_faster(policy_net, n=n, m=m, reward=reward_fun, eval_runs=5 * eval_runs,
                                                           batch_size=eval_parallel_batch_size)  # only update the best if it is really better, so run more tests
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

            if ep % target_update_freq == 0:
                """user_text, timed_out = timedInput(prompt="Press Y if you would like to stop the training now!\n", timeout=2)
    
                if not timed_out and user_text == "Y":
                    print("Training has been stopped by the user.")
                    return best_net
                else:
                    if not timed_out:
                        print("You pressed the wrong button, it has no effect. Training continues.")"""
                target_net.load_state_dict(policy_net.state_dict())

    print(f"--- {(time.time() - start_time)} seconds ---")
    return best_net


if __name__ == "__main__":
    train()
