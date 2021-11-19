import torch
import torch.nn as nn


class FullTwoThinningNet(nn.Module):

    def __init__(self, n, max_threshold, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FullTwoThinningNet, self).__init__()
        self.n = n
        self.max_threshold = max_threshold
        self.device = device

        self.fc = nn.Sequential( # TODO: maybe try with just two layers
            nn.Linear(self.n, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.max_threshold + 1)
        )

        self.to(self.device).double()

    def forward(self, x):
        x = x.double().sort()[0].to(self.device)
        res = self.fc(x)
        return res
