import numpy as np
import torch
import torch.nn as nn

from two_thinning.full_knowledge.RL.DeepSarsaRL_single_output.neural_network import FullTwoThinningNet

n = 10
m = 20

epsilon = 0.1  # TODO: set (exponential) decay
train_episodes = 3000


def reward(x):
    return -np.max(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def epsilon_greedy(model, loads, epsilon=epsilon, m=m):
    options=torch.DoubleTensor([model(torch.cat([torch.from_numpy(loads),torch.unsqueeze(torch.as_tensor(i),-1)])) for i in range(m+1)])
    r = torch.rand(1)
    if r < epsilon:
        a = torch.randint(m+1, (1,))[0]
    else:
        a = torch.argmax(options)
    return a, options[a]


def train(n=n, m=m, epsilon=epsilon, reward=reward, episodes=train_episodes, device=device):
    model = FullTwoThinningNet(n, device)
    optimizer = torch.optim.Adam(model.parameters())
    mse_loss = nn.MSELoss()

    for _ in range(episodes):
        loads = np.zeros(n)
        for i in range(m):
            a, old_val = epsilon_greedy(model, loads, epsilon=epsilon, m=m)
            randomly_selected = np.random.randint(n)
            if loads[randomly_selected] <= a:
                loads[randomly_selected] += 1
            else:
                loads[np.random.randint(n)] += 1

            if i == m - 1:
                new_val = torch.as_tensor(reward(loads)).to(device)
            else:
                _, new_val = epsilon_greedy(model, loads, epsilon=epsilon, m=m)
                new_val = new_val.detach()

            loss = mse_loss(old_val, new_val)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


if __name__=="__main__":
    train()