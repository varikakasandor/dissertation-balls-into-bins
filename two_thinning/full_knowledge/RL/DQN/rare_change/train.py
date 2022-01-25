import copy
import random
import time

import torch.optim as optim

from helper.replay_memory import ReplayMemory
from two_thinning.full_knowledge.RL.DQN.constants import *
from two_thinning.full_knowledge.RL.DQN.train import epsilon_greedy, greedy, optimize_model


def evaluate_q_values_faster(model, n=N, m=M, reward=REWARD_FUN, eval_runs=EVAL_RUNS_TRAIN,
                             threshold_change_freq=THRESHOLD_CHANGE_FREQ, batch_size=EVAL_PARALLEL_BATCH_SIZE):
    batches = [batch_size] * (eval_runs // batch_size)
    if eval_runs % batch_size != 0:
        batches.append(eval_runs % batch_size)

    with torch.no_grad():
        sum_loads = 0
        for batch in batches:
            loads = [[0] * n for _ in range(batch)]
            for start_period in range(0, m, threshold_change_freq):
                a = greedy(model, loads, batched=True)
                for _ in range(start_period, min(start_period + threshold_change_freq, m)):
                    first_choices = random.choices(range(n), k=batch)
                    second_choices = random.choices(range(n), k=batch)
                    for j in range(batch):  # TODO: speed up for loop
                        if loads[j][first_choices[j]] <= a[j]:
                            loads[j][first_choices[j]] += 1
                        else:
                            loads[j][second_choices[j]] += 1
            sum_loads += sum([reward(l) for l in loads])
        avg_score = sum_loads / eval_runs
        return avg_score


def train(n=N, m=M, memory_capacity=MEMORY_CAPACITY, num_episodes=TRAIN_EPISODES, reward_fun=REWARD_FUN,
          continuous_reward=CONTINUOUS_REWARD, batch_size=BATCH_SIZE, eps_start=EPS_START, eps_end=EPS_END,
          eps_decay=EPS_DECAY, optimise_freq=OPTIMISE_FREQ, target_update_freq=TARGET_UPDATE_FREQ,
          eval_runs=EVAL_RUNS_TRAIN, patience=PATIENCE, threshold_change_freq=THRESHOLD_CHANGE_FREQ,
          max_threshold=MAX_THRESHOLD, eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE,
          print_progress=PRINT_PROGRESS, nn_model=NN_MODEL, device=DEVICE):
    start_time = time.time()

    policy_net = nn_model(n=n, max_threshold=max_threshold, max_possible_load=m, device=device)
    target_net = nn_model(n=n, max_threshold=max_threshold, max_possible_load=m, device=device)
    best_net = nn_model(n=n, max_threshold=max_threshold, max_possible_load=m, device=device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(memory_capacity)

    steps_done = 0
    best_eval_score = None
    not_improved = 0

    for ep in range(num_episodes):
        loads = [0] * n
        for start_period in range(0, m, threshold_change_freq):
            used_loads = copy.deepcopy(loads)
            threshold = epsilon_greedy(policy_net=policy_net, loads=used_loads, max_threshold=max_threshold,
                                       steps_done=steps_done,
                                       eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, device=device)
            for i in range(start_period, min(start_period + threshold_change_freq, m)):
                randomly_selected = random.randrange(n)
                to_place = randomly_selected if loads[randomly_selected] <= threshold.item() else random.randrange(n)
                loads[to_place] += 1

            loads = sorted(loads)
            reward = 0
            for i in range(n):
                reward += (loads[i] - used_loads[i]) * i

            reward /= (n*threshold_change_freq)
            curr_state = used_loads
            reward = torch.DoubleTensor([reward]).to(device)
            next_state = copy.deepcopy(loads) if start_period + threshold_change_freq < m-1 else None
            memory.push(curr_state, threshold, next_state, reward)

            steps_done += 1

            if steps_done % optimise_freq == 0:
                optimize_model(memory=memory, policy_net=policy_net, target_net=target_net, optimizer=optimizer,
                               batch_size=batch_size, steps_done=steps_done, saturate_steps=50 * m,
                               device=device)  # TODO: should I not call it after every step instead only after every
                # episode? TODO: 10*m -> num_episodes*m

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

        if ep % target_update_freq == 0:  # TODO: decouple target update and optional user halting
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
