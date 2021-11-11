import torch
import torch.nn as nn


class AverageTwoThinningNet(nn.Module):

    def __init__(self, m, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(AverageTwoThinningNet, self).__init__()
        self.m = m
        self.device = device

        self.fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.m + 1)
        )

        self.to(self.device).double()

    def forward(self, x):
        x = x.to(self.device).double()
        res = self.fc(x)
        return res
