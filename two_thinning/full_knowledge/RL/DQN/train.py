import time
import copy
import random
from math import exp, log, floor, ceil
from os import mkdir
from sklearn.preprocessing import scale
import numpy as np

import wandb
from matplotlib import pyplot as plt
import torch.optim as optim

from helper.replay_memory import ReplayMemory, Transition
from two_thinning.full_knowledge.RL.DQN.constants import *

# from pytimedinput import timedInput # Works only with interactive interpreter


def epsilon_greedy(policy_net, loads, max_threshold, steps_done, eps_start, eps_end, eps_decay, device):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * exp(-1. * steps_done / eps_decay)
    if sample > eps_threshold:
        with torch.no_grad():
            options = policy_net(torch.tensor(loads).unsqueeze(0)).squeeze(0)
            return options.max(0)[1].type(dtype=torch.int64)
    else:
        return torch.as_tensor(random.randrange(max_threshold + 1), dtype=torch.int64).to(device)


def greedy(policy_net, loads, batched=False):
    with torch.no_grad():
        if batched:
            options = policy_net(torch.tensor(loads))
            return options.max(1)[1].type(dtype=torch.int64).tolist()
        else:
            options = policy_net(torch.tensor(loads).unsqueeze(0)).squeeze(0)
            return options.max(0)[1].type(dtype=torch.int64).item()  # TODO: instead torch.argmax (?)


def evaluate_q_values_faster(model, n=N, m=M, reward=REWARD_FUN, eval_runs=EVAL_RUNS_TRAIN,
                             batch_size=EVAL_PARALLEL_BATCH_SIZE):
    batches = [batch_size] * (eval_runs // batch_size)
    if eval_runs % batch_size != 0:
        batches.append(eval_runs % batch_size)

    with torch.no_grad():
        sum_loads = 0
        for batch in batches:
            loads = [[0] * n for _ in range(batch)]
            for _ in range(m):
                a = greedy(model, loads, batched=True)
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


def evaluate_q_values(model, n=N, m=M, reward=REWARD_FUN, eval_runs=EVAL_RUNS_TRAIN, print_behaviour=PRINT_BEHAVIOUR):
    with torch.no_grad():
        sum_loads = 0
        for _ in range(eval_runs):
            loads = [0] * n
            for _ in range(m):
                a = greedy(model, loads)
                if print_behaviour:
                    print(f"With loads {loads} the trained model chose {a}")
                randomly_selected = random.randrange(n)
                if loads[randomly_selected] <= a:
                    loads[randomly_selected] += 1
                else:
                    loads[random.randrange(n)] += 1
            sum_loads += reward(loads)
        avg_score = sum_loads / eval_runs
        return avg_score


def calc_number_of_jumps(thresholds, delta=4):
    num_jumps = 0
    for i in range(len(thresholds) - 1):
        if abs(thresholds[i] - thresholds[i + 1]) >= delta:
            num_jumps += 1
    return num_jumps


def analyse_threshold_progression(model, ep, save_folder, delta, n=N, m=M):
    with torch.no_grad():
        loads = [0] * n
        thresholds = []
        for _ in range(m):
            a = greedy(model, loads)
            thresholds.append(a)
            randomly_selected = random.randrange(n)
            if loads[randomly_selected] <= a:
                loads[randomly_selected] += 1
            else:
                loads[random.randrange(n)] += 1
        plt.clf()
        plt.plot(list(range(m)), thresholds)
        plt.xlabel("Index of ball")
        plt.ylabel("Chosen threshold")
        plt.title(f"Epoch {ep}")
        plt.savefig(join(save_folder, f"Epoch {ep}.png"))
        return calc_number_of_jumps(thresholds, delta=delta)


def optimize_model(memory, policy_net, target_net, optimizer, batch_size, criterion, device):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: not s, batch.done)), dtype=torch.bool).to(device)  # flip
    non_final_next_states = torch.tensor(
        [next_state for (done, next_state) in zip(batch.done, batch.next_state) if not done])

    state_action_values = policy_net(torch.tensor([x for x in batch.state]))
    state_action_values = state_action_values.gather(1,
                                                     torch.as_tensor([[a] for a in batch.action]).to(device)).squeeze()

    next_state_values = torch.zeros(batch_size).double().to(device)
    if torch.any(non_final_mask):  # needed for curriculum learning, as at the start all entries can be final
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values =torch.as_tensor(batch.reward).to(device) + next_state_values
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)  # Gradient clipping
    optimizer.step()


def train(n=N, m=M, memory_capacity=MEMORY_CAPACITY, num_episodes=TRAIN_EPISODES, reward_fun=REWARD_FUN,
          batch_size=BATCH_SIZE, eps_start=EPS_START, eps_end=EPS_END, report_wandb=False, lr=LR,
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

    for ep in range(num_episodes):
        loads = [0] * n
        for i in range(m):
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
                               device=device)  # TODO: should I not call it after every step instead only after every episode? TODO: 10*m -> num_episodes*m

        curr_eval_score = evaluate_q_values_faster(policy_net, n=n, m=m, reward=reward_fun, eval_runs=eval_runs,
                                                   batch_size=eval_parallel_batch_size)
        if best_eval_score is None or curr_eval_score > best_eval_score:
            curr_eval_score = evaluate_q_values_faster(policy_net, n=n, m=m, reward=reward_fun, eval_runs=5 * eval_runs,
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
    plt.plot(list(range(len(scaled_threshold_jumps))), scaled_threshold_jumps, label="normalised number of threshold jumps")
    plt.plot(list(range(len(scaled_eval_scores))), scaled_eval_scores, label="normalised evaluation scores")
    plt.title(f"Training Progression")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(join(save_path, "training_progression.png"))
    print(f"--- {(time.time() - start_time)} seconds ---")
    return best_net


if __name__ == "__main__":
    train()
