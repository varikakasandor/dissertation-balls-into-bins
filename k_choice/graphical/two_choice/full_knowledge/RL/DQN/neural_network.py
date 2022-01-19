import torch
import torch.nn as nn
import torch.nn.functional as F


class FullGraphicalTwoChoiceRecurrentNet(nn.Module):

    def __init__(self, n, max_possible_load, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FullGraphicalTwoChoiceRecurrentNet, self).__init__()
        self.n = n
        self.max_possible_load = max_possible_load
        self.device = device
        self.hidden_size = self.max_threshold+1

        self.rnn = nn.RNN(input_size=self.max_possible_load + 1, hidden_size=self.hidden_size, batch_first=True
                          )# ,nonlinearity='relu')  # ,dropout=0.5)
        # self.relu = nn.ReLU(), TODO: check if needed or not after RNN
        self.lin = nn.Linear(self.hidden_size, 1)

        self.to(self.device).double()

    def forward(self, x):
        x = F.one_hot(x.sort()[0], num_classes=self.max_possible_load + 1).double().to(self.device)
        x = self.rnn(x)[0][:, -1, :].squeeze(1)
        x = self.lin(x).squeeze(-1)
        return x
