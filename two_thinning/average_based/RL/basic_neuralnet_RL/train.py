import numpy as np
import torch
import torch.nn as nn

from two_thinning.average_based.RL.basic_neuralnet_RL.neural_network import AverageTwoThinningNet

n = 10
m = n

epsilon = 0.1
train_episodes = 3000
eval_runs = 300
patience = 20
print_progress = True
print_behaviour = False


def reward(x):
    return -np.max(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def greedy(model, ball_number):
    action_values = model(torch.DoubleTensor([ball_number]))
    a = torch.argmax(action_values)
    return a


def epsilon_greedy(model, ball_number, epsilon=epsilon):
    action_values = model(torch.DoubleTensor([ball_number]))
    r = torch.rand(1)
    if r < epsilon:
        a = torch.randint(len(action_values), (1,))[0]
    else:
        a = torch.argmax(action_values)
    return a, action_values[a]


def evaluate_q_values(model, n=n, m=m, reward=reward, eval_runs=eval_runs, print_behaviour=print_behaviour):
    with torch.no_grad():
        sum_loads = 0
        for _ in range(eval_runs):
            loads = np.zeros(n)
            for i in range(m):
                a = greedy(model, i)
                if print_behaviour:
                    print(f"With loads {loads} the trained model chose {a}")
                randomly_selected = np.random.randint(n)
                if loads[randomly_selected] <= a:
                    loads[randomly_selected] += 1
                else:
                    loads[np.random.randint(n)] += 1
            sum_loads += reward(loads)
        avg_score = sum_loads / eval_runs
        return avg_score


def train(n=n, m=m, epsilon=epsilon, reward=reward, episodes=train_episodes, eval_runs=eval_runs, patience=patience,
          print_progress=print_progress, print_behaviour=print_behaviour, device=device):
    curr_model = AverageTwoThinningNet(m, device)
    best_model = AverageTwoThinningNet(m, device)
    optimizer = torch.optim.Adam(curr_model.parameters())
    mse_loss = nn.MSELoss()

    best_eval_score = None
    not_improved = 0

    for ep in range(episodes):
        loads = np.zeros(n)
        for i in range(m):
            a, old_val = epsilon_greedy(curr_model, i, epsilon)
            randomly_selected = np.random.randint(n)
            if loads[randomly_selected] <= a:
                loads[randomly_selected] += 1
            else:
                loads[np.random.randint(n)] += 1

            if i == m - 1:
                new_val = torch.as_tensor(reward(loads)).to(device)
            else:
                _, new_val = epsilon_greedy(curr_model, i + 1, epsilon)
                new_val = new_val.detach()

            loss = mse_loss(old_val, new_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        curr_eval_score = evaluate_q_values(curr_model, n=n, m=m, reward=reward, eval_runs=eval_runs,
                                            print_behaviour=print_behaviour)
        if best_eval_score is None or curr_eval_score > best_eval_score:
            best_eval_score = curr_eval_score
            best_model.load_state_dict(curr_model.state_dict())
            not_improved = 0
            if print_progress:
                print(f"At episode {ep} the best eval score has improved to {curr_eval_score}.")
        elif not_improved < patience:
            not_improved += 1
            if print_progress:
                print(f"At episode {ep} no improvement happened.")
        else:
            if print_progress:
                print(f"Training has stopped after episode {ep} as the eval score didn't improve anymore.")
            break

    return best_model


if __name__ == "__main__":
    train()
