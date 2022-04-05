import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNet(nn.Module):

    def __init__(self, n, max_threshold, max_possible_load, hidden_size=64, rnn_num_layers=1, num_lin_layers=1,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(CriticNet, self).__init__()
        self.n = n
        self.max_possible_load = max_possible_load
        self.max_threshold = max_threshold
        self.device = device
        self.hidden_size = hidden_size  # self.max_threshold + 1

        self.rnn = nn.RNN(input_size=self.max_possible_load + 1, num_layers=rnn_num_layers,
                          hidden_size=self.hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.linear_block = []
        for _ in range(num_lin_layers - 1):
            self.linear_block.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.linear_block.append(self.relu)
        self.linear_block.append(nn.Linear(self.hidden_size, 1))
        self.linear_block = nn.Sequential(*self.linear_block)
        self.to(self.device).double()

    def forward(self, x):
        x = x.minimum(torch.tensor(self.max_possible_load))
        x = F.one_hot(x.sort()[0], num_classes=self.max_possible_load + 1).double().to(self.device)
        x = self.rnn(x)[0][:, -1, :].squeeze(1)
        x = self.linear_block(x).squeeze(-1)
        return x



class ActorNet(nn.Module):

    def __init__(self, n, max_threshold, max_possible_load, hidden_size=64, rnn_num_layers=1, num_lin_layers=1,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(ActorNet, self).__init__()
        self.n = n
        self.max_possible_load = max_possible_load
        self.max_threshold = max_threshold
        self.device = device
        self.hidden_size = hidden_size  # self.max_threshold + 1

        self.rnn = nn.RNN(input_size=self.max_possible_load + 1, num_layers=rnn_num_layers,
                          hidden_size=self.hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.linear_block = []
        for _ in range(num_lin_layers - 1):
            self.linear_block.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.linear_block.append(self.relu)
        self.linear_block.append(nn.Linear(self.hidden_size, self.max_threshold + 1))
        self.linear_block = nn.Sequential(*self.linear_block)
        self.softmax = nn.Softmax(dim=-1)
        self.to(self.device).double()

    def forward(self, x):
        x = x.minimum(torch.tensor(self.max_possible_load))
        x = F.one_hot(x.sort()[0], num_classes=self.max_possible_load + 1).double().to(self.device)
        x = self.rnn(x)[0][:, -1, :].squeeze(1)
        x = self.linear_block(x)
        x = self.softmax(x)
        return x
