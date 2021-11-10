import numpy as np
import torch
import torch.nn as nn

from two_thinning.average_based.RL.neural_network import AverageTwoThinningNet

n = 10
m = n

epsilon = 0.1  # TODO: set (exponential) decay
train_episodes = 300


def reward(x):
    return -np.max(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def epsilon_greedy(model, ball_number, epsilon=epsilon):
    action_values = model(torch.as_tensor(ball_number))
    r = torch.rand(1)
    if r < epsilon:
        a = torch.randint(len(action_values), (1,))[0]
    else:
        a = torch.argmax(action_values)
    return a, action_values[a]


def train(n=n, m=m, epsilon=epsilon, reward=reward, episodes=train_episodes, device=device):
    model = AverageTwoThinningNet(m, device)
    optimizer = torch.optim.Adam(model.parameters())
    mse_loss = nn.MSELoss()

    for _ in range(episodes):
        loads = np.zeros(n)
        for i in range(m):
            a, old_val = epsilon_greedy(model, i, epsilon)
            randomly_selected = np.random.randint(n)
            if loads[randomly_selected] <= a:
                loads[randomly_selected] += 1
            else:
                loads[np.random.randint(n)] += 1

            if i == m - 1:
                new_val = torch.as_tensor(reward(loads)).to(device)
            else:
                _, new_val = epsilon_greedy(model, i+1, epsilon)
                new_val = new_val.detach()

            loss = mse_loss(old_val, new_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


if __name__=="__main__":
    train()