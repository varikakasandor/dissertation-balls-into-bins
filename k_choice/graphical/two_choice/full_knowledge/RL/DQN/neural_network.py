import torch
import torch.nn as nn
import torch.nn.functional as F


class FullGraphicalTwoChoiceFCNet(nn.Module):

    def __init__(self, n, max_possible_load=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FullGraphicalTwoChoiceFCNet, self).__init__()
        self.n = n
        self.device = device
        self.hidden_size = self.n

        self.fc = nn.Sequential(  # TODO: maybe try with just two layers
            nn.Linear(self.n, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.n)
        )

        self.to(self.device).double()

    def forward(self, x):
        x = x.double().to(self.device)
        x = self.fc(x)
        return x


class FullGraphicalTwoChoiceOneHotFCNet(nn.Module):

    def __init__(self, n, max_possible_load, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FullGraphicalTwoChoiceOneHotFCNet, self).__init__()
        self.n = n
        self.max_possible_load = max_possible_load
        self.device = device
        self.hidden_size = self.n

        self.fc = nn.Sequential(  # TODO: maybe try with just two layers
            nn.Linear(self.n * (self.max_possible_load + 1), self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.n)
        )

        self.to(self.device).double()

    def forward(self, x):
        x = F.one_hot(x, num_classes=self.max_possible_load + 1).flatten(-2, -1).double().to(self.device)
        x = self.fc(x)
        return x


