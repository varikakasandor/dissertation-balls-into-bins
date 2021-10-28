import torch
import torch.nn as nn


class TwoThinningNet(nn.Module):

    def __init__(self, n, m, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(TwoThinningNet, self).__init__()
        self.n = n
        self.m = m
        self.device = device

        self.fc = nn.Sequential(
            nn.Linear(self.n, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.m + 1)
        )

    def forward(self, x):
        x = x.to(self.device)
        res = self.fc(x)
        return res
