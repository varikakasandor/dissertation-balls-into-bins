import copy
import random
from math import exp

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


def greedy(policy_net, loads):
    with torch.no_grad():
        options = policy_net(torch.tensor(loads).unsqueeze(0)).squeeze(0)
        return options.max(0)[1].type(dtype=torch.int64)  # TODO: instead torch.argmax (?)


def evaluate_q_values(model, n=N, m=M, reward=REWARD_FUN, eval_runs=EVAL_RUNS_TRAIN, print_behaviour=PRINT_BEHAVIOUR):
    with torch.no_grad():
        sum_loads = 0
        for _ in range(eval_runs):
            loads = [0] * n
            for i in range(m):
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


def optimize_model(memory, policy_net, target_net, optimizer, batch_size, steps_done, saturate_steps, device):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(device)
    non_final_next_states = torch.tensor([s for s in batch.next_state if s is not None])

    state_action_values = policy_net(torch.tensor([x for x in batch.state]))
    state_action_values = state_action_values.gather(1,
                                                     torch.as_tensor([[a] for a in batch.action]).to(device)).squeeze()

    next_state_values = torch.zeros(batch_size).double().to(device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # argmax = target_net(non_final_next_states).max(1)[1].detach() # TODO: double Q learning
    # next_state_values[non_final_mask] = policy(non_final_next_states)[argmax].detach() # TODO: double Q learning
    curr_weight = sqrt(min(steps_done,saturate_steps) / saturate_steps)/2 # Converges to 0.5 starting from 0
    expected_state_action_values = curr_weight * next_state_values + (1 - curr_weight) * torch.as_tensor(
        batch.reward).to(device)  # TODO: remove weighting

    criterion = nn.SmoothL1Loss()  # Huber loss TODO: maybe not the best
    loss = criterion(state_action_values, expected_state_action_values)  # .unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)  # Gradient clipping
    optimizer.step()


def train(n=N, m=M, memory_capacity=MEMORY_CAPACITY, num_episodes=TRAIN_EPISODES, reward_fun=REWARD_FUN,
          continuous_reward=CONTINUOUS_REWARD, batch_size=BATCH_SIZE, eps_start=EPS_START, eps_end=EPS_END,
          eps_decay=EPS_DECAY, optimise_freq=OPTIMISE_FREQ, target_update_freq=TARGET_UPDATE_FREQ, eval_runs=EVAL_RUNS_TRAIN, patience=PATIENCE,
          max_threshold=MAX_THRESHOLD, max_load_increase_reward=MAX_LOAD_INCREASE_REWARD,
          print_behaviour=PRINT_BEHAVIOUR, print_progress=PRINT_PROGRESS, nn_model=NN_MODEL, device=DEVICE):
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
        max_load = 0
        for i in range(m):
            threshold = epsilon_greedy(policy_net=policy_net, loads=loads, max_threshold=max_threshold,
                                       steps_done=steps_done,
                                       eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, device=device)
            randomly_selected = random.randrange(n)
            to_place = randomly_selected if loads[randomly_selected] <= threshold.item() else random.randrange(n)
            larger = len([j for j in range(n) if loads[j] > loads[to_place]])
            curr_state = copy.deepcopy(loads)
            # increased_max_load = (max_load == loads[to_place])
            loads[to_place] += 1
            next_state = copy.deepcopy(loads) if i != m - 1 else None
            # max_load = max(max_load, loads[to_place])

            if continuous_reward:
                reward = larger  # max_load_increase_reward if increased_max_load else 0
            else:
                reward = reward_fun(loads) if i == m - 1 else 0
            reward = torch.DoubleTensor([reward]).to(device)
            memory.push(curr_state, threshold, next_state, reward)

            steps_done += 1

            if steps_done % optimise_freq == 0:
                optimize_model(memory=memory, policy_net=policy_net, target_net=target_net, optimizer=optimizer,
                               batch_size=batch_size, steps_done=steps_done, saturate_steps=10 * m,
                               device=device)  # TODO: should I not call it after every step instead only after every episode? TODO: 10*m -> num_episodes*m



        curr_eval_score = evaluate_q_values(policy_net, n=n, m=m, reward=reward_fun, eval_runs=eval_runs,
                                            print_behaviour=print_behaviour)
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

    return best_net


if __name__ == "__main__":
    train()
