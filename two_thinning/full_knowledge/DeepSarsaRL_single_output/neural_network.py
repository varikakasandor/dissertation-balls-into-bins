import torch
import torch.nn as nn


class FullTwoThinningNet(nn.Module):

    def __init__(self, n, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FullTwoThinningNet, self).__init__()
        self.n = n
        self.device = device

        self.fc = nn.Sequential(
            nn.Linear(self.n + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.to(self.device).double()

    def forward(self, x):
        x = x.to(self.device).double()
        res = self.fc(x)
        return res
