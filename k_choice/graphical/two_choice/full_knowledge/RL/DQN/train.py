import time
import copy
import random
from math import exp, ceil, log

import torch.optim as optim
import wandb
from matplotlib import pyplot as plt

from helper.replay_memory import ReplayMemory, Transition
from k_choice.graphical.two_choice.full_knowledge.RL.DQN.constants import *
from k_choice.graphical.two_choice.graphs.graph_base import GraphBase
from k_choice.simulation import sample_one_choice


def epsilon_greedy(policy_net, loads, edge, steps_done, eps_start, eps_end, eps_decay, device):
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * exp(-1. * steps_done / eps_decay)
    if sample > eps_threshold:
        x, y = edge
        with torch.no_grad():
            vals = policy_net(torch.as_tensor(loads).unsqueeze(0)).squeeze(0).detach()
            if vals[x] >= vals[y]:
                return x
            else:
                return y
    else:
        return torch.tensor(random.choice(list(edge))).to(device)


def greedy(policy_net, loads, edge, batched=False):
    with torch.no_grad():
        if batched:
            pass
        else:
            x, y = edge
            vals = policy_net(torch.as_tensor(loads).unsqueeze(0)).squeeze(0).detach()
            if vals[x] >= vals[y]:
                return x
            else:
                return y


def evaluate_q_values(model, graph: GraphBase, m=M, reward=REWARD_FUN, eval_runs=EVAL_RUNS_TRAIN,
                      print_behaviour=PRINT_BEHAVIOUR):
    with torch.no_grad():
        sum_loads = 0
        for _ in range(eval_runs):
            loads = [0] * graph.n
            for i in range(m):
                edge = random.choice(graph.edge_list)
                chosen = greedy(model, loads, edge)
                if print_behaviour:
                    print(f"With loads {loads}, and edge {edge} the trained model chose {chosen}")
                loads[chosen] += 1
            sum_loads += reward(loads)
        avg_score = sum_loads / eval_runs
        return avg_score


def optimize_model(memory, policy_net, target_net, optimizer, batch_size, criterion, device):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: not s, batch.done)), dtype=torch.bool).to(device)  # flip
    non_final_next_loads = torch.tensor(
        [next_state_loads for (done, (next_state_loads, _)) in zip(batch.done, batch.next_state) if not done])
    non_final_next_edge = torch.tensor([list(next_state_edge) for (done, (_, next_state_edge)) in
                                        zip(batch.done, batch.next_state) if not done]).to(device)

    state_action_values = policy_net(torch.tensor([loads for (loads, _) in batch.state]))
    state_action_values = state_action_values.gather(1,
                                                     torch.as_tensor([[a] for a in batch.action]).to(device)).squeeze()

    next_state_values = torch.zeros(batch_size).double().to(device)
    options = target_net(non_final_next_loads)
    next_state_values[non_final_mask] = options.gather(1, non_final_next_edge).max(1)[0].detach()
    expected_state_action_values = next_state_values + torch.as_tensor(batch.reward).to(device)

    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train(graph: GraphBase = GRAPH, m=M, memory_capacity=MEMORY_CAPACITY, num_episodes=TRAIN_EPISODES,
          loss_function=LOSS_FUCNTION,
          reward_fun=REWARD_FUN, potential_fun=POTENTIAL_FUN, report_wandb=False, pre_train_episodes=PRE_TRAIN_EPISODES,
          batch_size=BATCH_SIZE, eps_start=EPS_START, eps_end=EPS_END, lr=LR, pacing_fun=PACING_FUN,
          nn_num_lin_layers=NN_NUM_LIN_LAYERS,
          eps_decay=EPS_DECAY, optimise_freq=OPTIMISE_FREQ, target_update_freq=TARGET_UPDATE_FREQ,
          nn_hidden_size=NN_HIDDEN_SIZE,
          eval_runs=EVAL_RUNS_TRAIN, patience=PATIENCE, print_progress=PRINT_PROGRESS, nn_model=NN_MODEL,
          optimizer_method=OPTIMIZER_METHOD, device=DEVICE):
    start_time = time.time()

    max_possible_load = min(m, m // graph.n + 2 * ceil(sqrt(log(graph.n))))
    policy_net = nn_model(n=graph.n, max_possible_load=max_possible_load, hidden_size=nn_hidden_size,
                          num_lin_layers=nn_num_lin_layers, device=device)
    target_net = nn_model(n=graph.n, max_possible_load=max_possible_load, hidden_size=nn_hidden_size,
                          num_lin_layers=nn_num_lin_layers, device=device)
    best_net = nn_model(n=graph.n, max_possible_load=max_possible_load, hidden_size=nn_hidden_size,
                        num_lin_layers=nn_num_lin_layers, device=device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optimizer_method(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(memory_capacity)

    steps_done = 0
    best_eval_score = None
    not_improved = 0
    eval_scores = []

    start_loads = []
    for start_size in reversed(range(m)):  # pretraining (i.e. curriculum learning)
        for _ in range(pacing_fun(start_size=start_size, graph=graph, m=m, all_episodes=pre_train_episodes)):
            start_loads.append(sample_one_choice(n=graph.n, m=start_size))
    for _ in range(num_episodes):  # training
        start_loads.append([0] * graph.n)

    for ep, loads in enumerate(start_loads):
        edge = random.choice(graph.edge_list)
        for i in range(m):
            torch.cuda.empty_cache()
            chosen = epsilon_greedy(policy_net=policy_net, loads=loads, edge=edge, steps_done=steps_done,
                                    eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, device=device)
            old_loads = copy.deepcopy(loads)
            curr_state = (old_loads, edge)
            loads[chosen] += 1
            next_edge = random.choice(graph.edge_list)
            next_state = (copy.deepcopy(loads), next_edge)
            if i == m - 1:
                reward = reward_fun(loads) - potential_fun(graph, old_loads)
            else:
                reward = potential_fun(graph, loads) - potential_fun(graph, old_loads)
            reward = torch.DoubleTensor([reward]).to(device)
            memory.push(curr_state, chosen, next_state, reward, i == m - 1)

            edge = next_edge
            steps_done += 1

            if steps_done % optimise_freq == 0:
                optimize_model(memory=memory, policy_net=policy_net, target_net=target_net, optimizer=optimizer,
                               batch_size=batch_size, criterion=loss_function, device=device)
        curr_eval_score = evaluate_q_values(policy_net, graph=graph, m=m, reward=reward_fun, eval_runs=eval_runs)
        if best_eval_score is None or curr_eval_score >= best_eval_score:
            curr_eval_score = evaluate_q_values(policy_net, graph=graph, m=m, reward=reward_fun,
                                                eval_runs=5 * eval_runs)

        if report_wandb:
            wandb.log({"score": curr_eval_score})

        eval_scores.append(curr_eval_score)
        if best_eval_score is None or curr_eval_score >= best_eval_score:
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

    eval_max_loads = [-x for x in eval_scores]
    plt.plot(eval_max_loads)
    plt.xlabel("episode")
    plt.ylabel(f"average maximum load over {5 * eval_runs} runs")
    plt.savefig(f"../../../../../../evaluation/graphical_two_choice/data/training_progression_{graph.name}_{graph.n}_{m}.png")
    print(f"--- {(time.time() - start_time)} seconds ---")
    return best_net


if __name__ == "__main__":
    train()
