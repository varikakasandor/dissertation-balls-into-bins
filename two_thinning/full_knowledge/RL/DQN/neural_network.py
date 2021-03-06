import torch
import torch.nn as nn
import torch.nn.functional as F


class FullTwoThinningNet(nn.Module):

    def __init__(self, n, max_threshold, inner1_size=None, inner2_size=None, hidden_size=64,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FullTwoThinningNet, self).__init__()
        self.n = n
        self.max_threshold = max_threshold
        self.device = device
        self.inner1_size = inner1_size if inner1_size else max(self.max_threshold + 1, self.n // 2)  # TODO: 256
        self.inner2_size = inner2_size if inner2_size else max(self.max_threshold + 1,
                                                               self.inner1_size // 2)  # TODO: 128

        self.fc = nn.Sequential(  # TODO: maybe try with just two layers
            nn.Linear(self.n, self.inner1_size),
            nn.ReLU(),
            nn.Linear(self.inner1_size, self.inner2_size),
            nn.ReLU(),
            nn.Linear(self.inner2_size, self.max_threshold + 1)
        )

        self.to(self.device).double()

    def forward(self, x):
        x = x.double().sort()[0].to(self.device)
        res = self.fc(x)
        return res


class FullTwoThinningOneHotNet(nn.Module):

    def __init__(self, n, max_threshold, max_possible_load, hidden_size=64, inner1_size=None, inner2_size=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FullTwoThinningOneHotNet, self).__init__()
        self.n = n
        self.max_possible_load = max_possible_load
        self.max_threshold = max_threshold
        self.device = device
        self.inner1_size = inner1_size if inner1_size else max(self.max_threshold + 1,
                                                               (self.n // 2) * (
                                                                       self.max_possible_load + 1) // 2)  # TODO: 256
        self.inner2_size = inner2_size if inner2_size else max(self.max_threshold + 1,
                                                               self.inner1_size // 4)  # TODO: 128

        self.fc = nn.Sequential(  # TODO: maybe try with just two layers
            nn.Linear(self.n * (self.max_possible_load + 1), self.inner1_size),
            nn.ReLU(),
            nn.Linear(self.inner1_size, self.inner2_size),
            nn.ReLU(),
            nn.Linear(self.inner2_size, self.max_threshold + 1)
        )

        self.to(self.device).double()

    def forward(self, x):
        x = F.one_hot(x.sort()[0], num_classes=self.max_possible_load + 1).view(-1, self.n * (
                self.max_possible_load + 1)).double().to(self.device)
        res = self.fc(x)
        return res


class FullTwoThinningRecurrentNet(nn.Module):

    def __init__(self, n, max_threshold, max_possible_load, hidden_size=64,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FullTwoThinningRecurrentNet, self).__init__()
        self.n = n
        self.max_possible_load = max_possible_load
        self.max_threshold = max_threshold
        self.device = device
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=self.max_possible_load + 1, hidden_size=self.hidden_size, batch_first=True)
        self.to(self.device).double()

    def forward(self, x):
        x = F.one_hot(x.sort()[0], num_classes=self.max_possible_load + 1).double().to(self.device)
        x = self.rnn(x)[0][:, -1, :].squeeze(1)
        return x


class FullTwoThinningRecurrentNetFC(nn.Module):

    def __init__(self, n, max_threshold, max_possible_load, hidden_size=64,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FullTwoThinningRecurrentNetFC, self).__init__()
        self.n = n
        self.max_possible_load = max_possible_load
        self.max_threshold = max_threshold
        self.device = device
        self.hidden_size = hidden_size  # self.max_threshold + 1
        # TODO: one extra layer converting one-hot to embedding (same for all loads)
        self.rnn = nn.RNN(input_size=self.max_possible_load + 1, hidden_size=self.hidden_size, batch_first=True)
        self.lin = nn.Linear(self.hidden_size, self.max_threshold + 1)
        self.to(self.device).double()

    def forward(self, x):
        x = F.one_hot(x.sort()[0], num_classes=self.max_possible_load + 1).double().to(self.device)
        x = self.rnn(x)[0][:, -1, :].squeeze(1)
        x = self.lin(x)
        return x


class FullTwoThinningGRUNetFC(nn.Module):

    def __init__(self, n, max_threshold, max_possible_load, hidden_size=64,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FullTwoThinningGRUNetFC, self).__init__()
        self.n = n
        self.max_possible_load = max_possible_load
        self.max_threshold = max_threshold
        self.device = device
        self.hidden_size = hidden_size  # self.max_threshold + 1
        # TODO: one extra layer converting one-hot to embedding (same for all loads)
        self.rnn = nn.GRU(input_size=self.max_possible_load + 1, hidden_size=self.hidden_size, batch_first=True)
        self.lin = nn.Linear(self.hidden_size, self.max_threshold + 1)
        self.to(self.device).double()

    def forward(self, x):
        x = F.one_hot(x.sort()[0], num_classes=self.max_possible_load + 1).double().to(self.device)
        x = self.rnn(x)[0][:, -1, :].squeeze(1)
        x = self.lin(x)
        return x


class FullTwoThinningClippedRecurrentNetFC(nn.Module):

    def __init__(self, n, max_threshold, max_possible_load, hidden_size=64,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FullTwoThinningClippedRecurrentNetFC, self).__init__()
        self.n = n
        self.max_possible_load = max_possible_load
        self.max_threshold = max_threshold
        self.device = device
        self.hidden_size = hidden_size  # self.max_threshold + 1
        # TODO: one extra layer converting one-hot to embedding (same for all loads)
        self.rnn = nn.RNN(input_size=self.max_possible_load + 1, hidden_size=self.hidden_size, batch_first=True)
        self.lin = nn.Linear(self.hidden_size, self.max_threshold + 1)
        # TODO: add softmax, pass previous threshold as argument
        self.to(self.device).double()

    def forward(self, x):
        x = x.minimum(torch.tensor(self.max_possible_load))
        x = F.one_hot(x.sort()[0], num_classes=self.max_possible_load + 1).double().to(self.device)
        x = self.rnn(x)[0][:, -1, :].squeeze(1)
        x = self.lin(x)
        return x


class GeneralNet(nn.Module):

    def __init__(self, n, max_threshold, max_possible_load, hidden_size=128, rnn_num_layers=3, num_lin_layers=2,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(GeneralNet, self).__init__()
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
        self.to(self.device).double()

    def forward(self, x):
        x = x.minimum(torch.tensor(self.max_possible_load))
        x = F.one_hot(x.sort()[0], num_classes=self.max_possible_load + 1).double().to(self.device)
        x = self.rnn(x)[0][:, -1, :].squeeze(1)
        x = self.linear_block(x)
        return x
