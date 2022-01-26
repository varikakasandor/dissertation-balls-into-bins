import torch
import torch.nn as nn
import torch.nn.functional as F


class FullTwoThinningNet(nn.Module):

    def __init__(self, n, max_threshold, inner1_size=None, inner2_size=None,
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

    def __init__(self, n, max_threshold, max_possible_load, inner1_size=None, inner2_size=None,
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

    def __init__(self, n, max_threshold, max_possible_load, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FullTwoThinningRecurrentNet, self).__init__()
        self.n = n
        self.max_possible_load = max_possible_load
        self.max_threshold = max_threshold
        self.device = device
        self.hidden_size = self.max_threshold+1
        self.rnn = nn.RNN(input_size=self.max_possible_load + 1, hidden_size=self.hidden_size, batch_first=True)
        self.to(self.device).double()

    def forward(self, x):
        x = F.one_hot(x.sort()[0], num_classes=self.max_possible_load + 1).double().to(self.device)
        x = self.rnn(x)[0][:, -1, :].squeeze(1)
        return x

class FullTwoThinningRecurrentNetFC(nn.Module):

    def __init__(self, n, max_threshold, max_possible_load, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FullTwoThinningRecurrentNetFC, self).__init__()
        self.n = n
        self.max_possible_load = max_possible_load
        self.max_threshold = max_threshold
        self.device = device
        self.hidden_size = self.max_threshold+1
        self.rnn = nn.RNN(input_size=self.max_possible_load + 1, hidden_size=self.hidden_size, batch_first=True)
        self.lin = nn.Linear(self.hidden_size, self.max_threshold + 1)
        self.to(self.device).double()

    def forward(self, x):
        x = F.one_hot(x.sort()[0], num_classes=self.max_possible_load + 1).double().to(self.device)
        x = self.rnn(x)[0][:, -1, :].squeeze(1)
        x = self.lin(x)
        return x

