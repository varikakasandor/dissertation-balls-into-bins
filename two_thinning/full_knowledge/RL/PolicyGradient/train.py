import copy
import random
import time
from os import mkdir

import numpy as np
import torch.optim as optim
import wandb
from matplotlib import pyplot as plt
from sklearn.preprocessing import scale

from two_thinning.full_knowledge.RL.PolicyGradient.constants import *


def sample_action(actor_net, loads):
    probs = actor_net(torch.tensor(loads).unsqueeze(0)).squeeze(0).detach()
    probs_detached = probs.detach().cpu().numpy()
    threshold = np.random.choice(len(probs_detached), p=probs_detached)
    return threshold



def evaluate_q_values(actor_net, n=N, m=M, reward=REWARD_FUN, eval_runs=EVAL_RUNS_TRAIN,
                      print_behaviour=PRINT_BEHAVIOUR):
    with torch.no_grad():
        sum_loads = 0
        for _ in range(eval_runs):
            loads = [0] * n
            for _ in range(m):
                threshold = sample_action(actor_net, loads)
                if print_behaviour:
                    print(f"With loads {loads} the trained model chose {threshold}")
                randomly_selected = random.randrange(n)
                if loads[randomly_selected] <= threshold:
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


def analyse_threshold_progression(actor_net, ep, save_folder, delta, n=N, m=M):
    with torch.no_grad():
        loads = [0] * n
        thresholds = []
        for _ in range(m):
            a = sample_action(actor_net, loads)
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


def train(n=N, m=M, num_episodes=TRAIN_EPISODES, reward_fun=REWARD_FUN, batch_size=BATCH_SIZE, report_wandb=False,
          eps_decay=EPS_DECAY, optimise_freq=OPTIMISE_FREQ, nn_hidden_size=NN_HIDDEN_SIZE,
          nn_rnn_num_layers=NN_RNN_NUM_LAYERS, nn_num_lin_layers=NN_NUM_LIN_LAYERS, lr=LR,
          eval_runs=EVAL_RUNS_TRAIN, patience=PATIENCE, potential_fun=POTENTIAL_FUN, loss_function=LOSS_FUCNTION,
          max_threshold=MAX_THRESHOLD, eval_parallel_batch_size=EVAL_PARALLEL_BATCH_SIZE, save_path=SAVE_PATH,
          print_progress=PRINT_PROGRESS, critic_model=CRITIC_MODEL, actor_model=ACTOR_MODEL, device=DEVICE):
    start_time = time.time()

    mkdir(save_path)

    max_possible_load = m // n + ceil(sqrt(log(n)))

    critic_net = critic_model(n=n, max_threshold=max_threshold, max_possible_load=max_possible_load,
                              hidden_size=nn_hidden_size, rnn_num_layers=nn_rnn_num_layers,
                              num_lin_layers=nn_num_lin_layers,
                              device=device)

    actor_net = actor_model(n=n, max_threshold=max_threshold, max_possible_load=max_possible_load,
                            hidden_size=nn_hidden_size, rnn_num_layers=nn_rnn_num_layers,
                            num_lin_layers=nn_num_lin_layers,
                            device=device)

    best_net = actor_model(n=n, max_threshold=max_threshold, max_possible_load=max_possible_load,
                            hidden_size=nn_hidden_size, rnn_num_layers=nn_rnn_num_layers,
                            num_lin_layers=nn_num_lin_layers,
                            device=device)

    actor_optimizer = optim.Adam(actor_net.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=lr)

    steps_done = 0
    best_eval_score = None
    not_improved = 0
    threshold_jumps = []
    eval_scores = []

    for ep in range(num_episodes):
        loads = [0] * n
        for i in range(m):
            torch.cuda.empty_cache()
            probs = actor_net(torch.tensor(loads).unsqueeze(0)).squeeze(0)
            probs_detached = probs.detach().cpu().numpy()
            threshold = np.random.choice(len(probs_detached), p=probs_detached)
            randomly_selected = random.randrange(n)
            to_place = randomly_selected if loads[randomly_selected] <= threshold else random.randrange(n)
            curr_state = copy.deepcopy(loads)
            loads[to_place] += 1
            next_state = copy.deepcopy(loads)

            if i == m - 1:
                reward = reward_fun(next_state) - potential_fun(curr_state)  # assumes potential_fun(final_states)=0
            else:
                reward = potential_fun(next_state) - potential_fun(curr_state)
            reward = torch.tensor(reward).to(device)

            curr_state_value = critic_net(torch.tensor(curr_state).unsqueeze(0)).squeeze(0)
            next_state_value = critic_net(torch.tensor(next_state).unsqueeze(0)).squeeze(0).detach() if i < m - 1 else torch.tensor(0.0, dtype=torch.float64)
            new_curr_state_value = reward + next_state_value
            delta = curr_state_value - new_curr_state_value  # minus of what is in the book, but I use built in optimiser which does grad. descent not grad. ascent
            critic_loss = loss_function(curr_state_value, new_curr_state_value)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            to_diff = delta.detach().item() * (probs.log())[threshold]
            actor_optimizer.zero_grad()
            to_diff.backward()
            actor_optimizer.step()

            steps_done += 1

        curr_eval_score = evaluate_q_values(actor_net, n=n, m=m, reward=reward_fun, eval_runs=eval_runs)  # TODO: set back to faster version
        if best_eval_score is None or curr_eval_score > best_eval_score:
            curr_eval_score = evaluate_q_values(actor_net, n=n, m=m, reward=reward_fun, eval_runs=5 * eval_runs)  # only update the best if it is really better, so run more tests
        if report_wandb:
            wandb.log({"score": curr_eval_score})

        eval_scores.append(curr_eval_score)
        threshold_jumps.append(analyse_threshold_progression(actor_net, ep, save_path, delta=max_threshold // 2))

        if best_eval_score is None or curr_eval_score > best_eval_score:
            best_eval_score = curr_eval_score
            best_net.load_state_dict(actor_net.state_dict())
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
    return best_net


if __name__ == "__main__":
    train()
