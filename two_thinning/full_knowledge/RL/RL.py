import numpy as np
import torch
import torch.nn as nn

from two_thinning.full_knowledge.RL.neural_network import TwoThinningNet

n = 10
m = n

# alpha = 0.1
epsilon = 0.1  # TODO: set (exponential) decay
train_episodes = 300
eval_episodes = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def epsilon_greedy(model, loads, epsilon=epsilon):
    action_values = model(loads)
    r = torch.rand(1)
    if r < epsilon:
        a = torch.randint(len(action_values), (1,))[0]
    else:
        a = torch.argmax(action_values)
    return a, action_values[a]


def train(n=n, m=m, episodes=train_episodes, epsilon=epsilon, device=device):
    model = TwoThinningNet(n, m, device)
    model.to(device).double()
    optimizer = torch.optim.Adam(model.parameters())
    mse_loss = nn.MSELoss()

    for _ in range(episodes):
        loads = np.zeros(n)
        for i in range(m):
            a, old_val = epsilon_greedy(model, torch.from_numpy(loads).double(), epsilon)
            randomly_selected = np.random.randint(n)
            if loads[randomly_selected] <= a:
                loads[randomly_selected] += 1
            else:
                loads[np.random.randint(n)] += 1

            if i == m - 1:
                new_val = torch.as_tensor(-np.max(loads)).to(device)
            else:
                _, new_val = epsilon_greedy(model, torch.from_numpy(loads).double(), epsilon)
                new_val = new_val.detach()

            loss = mse_loss(old_val, new_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
