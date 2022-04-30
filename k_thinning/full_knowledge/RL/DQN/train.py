import copy
import random
import time
from math import exp
from os import mkdir

import torch.optim as optim
import wandb
from matplotlib import pyplot as plt

from helper.replay_memory import ReplayMemory, Transition
from k_choice.simulation import sample_one_choice
from k_thinning.full_knowledge.RL.DQN.constants import *


# from pytimedinput import timedInput # Works only with interactive interpreter


def epsilon_greedy(policy_net, loads, choices_left, max_threshold, steps_done, eps_start, eps_end, eps_decay, device):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * exp(-1. * steps_done / eps_decay)
    if sample > eps_threshold:
        with torch.no_grad():
            options = policy_net(torch.tensor(loads + [choices_left]).unsqueeze(0)).squeeze(0)
            return options.max(0)[1].type(dtype=torch.int64)
    else:
        return torch.as_tensor(random.randrange(max_threshold + 1), dtype=torch.int64).to(device)


def greedy(policy_net, loads, choices_left):
    with torch.no_grad():
        options = policy_net(torch.tensor(loads + [choices_left]).unsqueeze(0)).squeeze(0)
        return options.max(0)[1].type(dtype=torch.int64).item()  # TODO: instead torch.argmax (?)


def evaluate_q_values(model, n=N, m=M, k=K, reward=REWARD_FUN, eval_runs=EVAL_RUNS_TRAIN,
                      max_threshold=MAX_THRESHOLD, use_normalised=USE_NORMALISED,
                      print_behaviour=PRINT_BEHAVIOUR):  # TODO: do fast version as for two_choice
    with torch.no_grad():
        sum_loads = 0
        for _ in range(eval_runs):
            loads = [0] * n
            for i in range(m):
                choices_left = k
                to_increase = None
                while choices_left > 1:
                    a = greedy(model, loads, choices_left)
                    a = i / n + a - max_threshold if use_normalised else a
                    if print_behaviour:
                        print(f"With loads {loads}, having {choices_left} choices left, the trained model chose {a}")
                    to_increase = random.randrange(n)
                    if loads[to_increase] <= a:
                        break
                    else:
                        choices_left -= 1

                if choices_left == 1:
                    to_increase = random.randrange(n)
                loads[to_increase] += 1

            sum_loads += reward(loads)
        avg_score = sum_loads / eval_runs
        return avg_score


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
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # argmax = target_net(non_final_next_states).max(1)[1].detach() # TODO: double Q learning
    # next_state_values[non_final_mask] = policy(non_final_next_states)[argmax].detach() # TODO: double Q learning
    expected_state_action_values = next_state_values + torch.as_tensor(batch.reward).to(device)

    loss = criterion(state_action_values, expected_state_action_values)  # .unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)  # Gradient clipping
    optimizer.step()


def train(n=N, m=M, k=K, memory_capacity=MEMORY_CAPACITY, num_episodes=TRAIN_EPISODES, reward_fun=REWARD_FUN,
          batch_size=BATCH_SIZE, eps_start=EPS_START, eps_end=EPS_END, report_wandb=False, lr=LR, pacing_fun=PACING_FUN,
          eps_decay=EPS_DECAY, optimise_freq=OPTIMISE_FREQ, target_update_freq=TARGET_UPDATE_FREQ,
          pre_train_episodes=PRE_TRAIN_EPISODES, use_normalised=USE_NORMALISED,
          nn_hidden_size=NN_HIDDEN_SIZE, nn_rnn_num_layers=NN_RNN_NUM_LAYERS, nn_num_lin_layers=NN_NUM_LIN_LAYERS,
          eval_runs=EVAL_RUNS_TRAIN, patience=PATIENCE, potential_fun=POTENTIAL_FUN, loss_function=LOSS_FUCNTION,
          max_threshold=MAX_THRESHOLD, eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE, save_path=SAVE_PATH,
          print_progress=PRINT_PROGRESS, nn_model=NN_MODEL, optimizer_method=OPTIMIZER_METHOD, device=DEVICE):
    start_time = time.time()
    mkdir(save_path)

    max_possible_load = m
    max_threshold = max_threshold - m // n if use_normalised else max_threshold  # !!!
    nn_max_threshold = 2 * max_threshold if use_normalised else max_threshold

    policy_net = nn_model(n=n, max_threshold=nn_max_threshold, k=k, max_possible_load=max_possible_load,
                          hidden_size=nn_hidden_size, rnn_num_layers=nn_rnn_num_layers,
                          num_lin_layers=nn_num_lin_layers,
                          device=device)
    target_net = nn_model(n=n, max_threshold=nn_max_threshold, k=k, max_possible_load=max_possible_load,
                          hidden_size=nn_hidden_size, rnn_num_layers=nn_rnn_num_layers,
                          num_lin_layers=nn_num_lin_layers,
                          device=device)
    best_net = nn_model(n=n, max_threshold=nn_max_threshold, k=k, max_possible_load=max_possible_load,
                        hidden_size=nn_hidden_size, rnn_num_layers=nn_rnn_num_layers, num_lin_layers=nn_num_lin_layers,
                        device=device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optimizer_method(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(memory_capacity)

    steps_done = 0
    best_eval_score = None
    not_improved = 0
    threshold_jumps = []
    eval_scores = []

    start_loads = []
    for start_size in reversed(range(m)):  # pretraining (i.e. curriculum learning)
        for _ in range(pacing_fun(start_size=start_size, n=n, m=m, all_episodes=pre_train_episodes)):
            start_loads.append(sample_one_choice(n=n, m=start_size))
    for _ in range(num_episodes):  # training
        start_loads.append([0] * n)

    for ep, loads in enumerate(start_loads):
        for i in range(m):
            to_place = None
            threshold = None
            randomly_selected = None
            choices_left = k
            while choices_left > 1:
                threshold = epsilon_greedy(policy_net=policy_net, loads=loads, choices_left=choices_left,
                                           max_threshold=max_threshold, steps_done=steps_done,
                                           eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, device=device)
                randomly_selected = random.randrange(n)
                if loads[randomly_selected] <= threshold.item():
                    to_place = randomly_selected
                    break
                elif choices_left > 2:
                    curr_state = loads + [choices_left]
                    next_state = loads + [choices_left - 1]
                    reward = 0
                    reward = torch.DoubleTensor([reward]).to(device)
                    memory.push(curr_state, threshold, next_state, reward, False)
                    steps_done += 1

                choices_left -= 1

            if choices_left == 1:  # nothing was good for the model
                to_place = random.randrange(n)
                choices_left += 1

            curr_state = loads + [choices_left]  # in a format that can directly go into the neural network
            loads[to_place] += 1
            next_state = (loads + [k])

            reward = reward_fun(next_state[:-1]) if i == m - 1 else 0  # "real" reward
            reward += potential_fun(next_state[:-1]) - potential_fun(curr_state[:-1])
            reward = torch.DoubleTensor([reward]).to(device)
            memory.push(curr_state, threshold, next_state, reward, i == m - 1)

            steps_done += 1

            if steps_done % optimise_freq == 0:
                optimize_model(memory=memory, policy_net=policy_net, target_net=target_net, optimizer=optimizer,
                               batch_size=batch_size, criterion=loss_function, device=device)

        curr_eval_score = evaluate_q_values(policy_net, n=n, m=m, k=k, max_threshold=max_threshold, reward=reward_fun,
                                            eval_runs=eval_runs, use_normalised=use_normalised)
        if best_eval_score is None or curr_eval_score > best_eval_score:
            curr_eval_score = evaluate_q_values(policy_net, n=n, m=m, k=k, max_threshold=max_threshold, reward=reward_fun,
                                                eval_runs=5 * eval_runs, use_normalised=use_normalised)
        if report_wandb:
            wandb.log({"score": curr_eval_score})

        eval_scores.append(curr_eval_score)
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
            target_net.load_state_dict(policy_net.state_dict())

    max_loads = [-x for x in eval_scores]
    plt.plot(max_loads)
    plt.xlabel("episode")
    plt.ylabel("average maximum load over 25 runs")
    file_name = f"training_progression_{n}_{m}_{k}"
    training_save_path = join(dirname(dirname(dirname(dirname(dirname(abspath(__file__)))))), "evaluation", "k_thinning", "data", file_name)
    plt.savefig(training_save_path)
    print(f"--- {(time.time() - start_time)} seconds ---")
    return best_net


if __name__ == "__main__":
    train()
