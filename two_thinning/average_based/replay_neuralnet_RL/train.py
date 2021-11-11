import math
import random

import torch
import torch.nn as nn
import torch.optim as optim

from two_thinning.average_based.replay_neuralnet_RL.neural_network import AverageTwoThinningNet
from two_thinning.average_based.replay_neuralnet_RL.replay_memory import ReplayMemory, Transition

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
REWARD_FUN = max
TRAIN_EPISODES = 100
TARGET_UPDATE_FREQ = 10
MEMORY_CAPACITY = 10000
N = 10
M = 10


def epsilon_greedy(policy_net, ball_number, m, steps_done, device):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            options = policy_net(torch.DoubleTensor([ball_number]))
            return options.max(0)[1].type(dtype=torch.int64)
    else:
        return torch.as_tensor(random.randrange(m + 1), dtype=torch.int64).to(device)


def optimize_model(memory, policy_net, target_net, optimizer, batch_size, device):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(device)
    non_final_next_states = torch.DoubleTensor([[s] for s in batch.next_state if s is not None])

    state_action_values = policy_net(torch.DoubleTensor([[x] for x in batch.state]))
    state_action_values = state_action_values.gather(1, torch.as_tensor([[a] for a in batch.action]).to(device)).squeeze()

    next_state_values = torch.zeros(batch_size).double().to(device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = next_state_values + torch.as_tensor(batch.reward).to(device)

    criterion = nn.SmoothL1Loss()  # Huber loss TODO: maybe not the best
    loss = criterion(state_action_values, expected_state_action_values)  # .unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)  # Gradient clipping
    optimizer.step()


def train(n=N, m=M, memory_capacity=MEMORY_CAPACITY, num_episodes=TRAIN_EPISODES, reward_fun=REWARD_FUN, batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQ, device=DEVICE):
    policy_net = AverageTwoThinningNet(m, device=device)
    target_net = AverageTwoThinningNet(m, device=device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(memory_capacity)

    steps_done = 0

    for i_episode in range(num_episodes):
        loads = [0] * n
        for i in range(m):

            threshold = epsilon_greedy(policy_net=policy_net, ball_number=i, m=m, steps_done=steps_done, device=device)
            randomly_selected = random.randrange(n)
            if loads[randomly_selected] <= threshold.item():
                loads[randomly_selected]+=1
            else:
                loads[random.randrange(n)]+=1

            reward = torch.DoubleTensor([0 if i+1<m else reward_fun(loads)]).to(device)

            curr_state = i
            next_state = i+1
            memory.push(curr_state, threshold, next_state, reward)

            optimize_model(memory=memory, policy_net=policy_net, target_net=target_net, optimizer=optimizer, batch_size=batch_size, device=device)

            steps_done += 1

        if i_episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')

    return policy_net


if __name__=="__main__":
    train()