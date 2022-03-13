import torch
import torch.nn as nn
import torch.nn.functional as F


class FullKThinningRecurrentNet(nn.Module):

    def __init__(self, n, max_threshold, k, max_possible_load,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(FullKThinningRecurrentNet, self).__init__()
        self.n = n
        self.max_possible_load = max_possible_load
        self.k = k
        self.max_threshold = max_threshold
        self.device = device
        self.hidden_size = self.max_threshold + 1

        self.rnn = nn.RNN(input_size=self.max_possible_load + 1, hidden_size=self.hidden_size, batch_first=True)  # ,nonlinearity='relu')  # ,dropout=0.5)
        self.lin = nn.Linear(self.hidden_size + self.k - 1, self.max_threshold + 1)
        self.to(self.device).double()

    def forward(self, x):
        loads = x[:,:-1]
        choices_left = x[:,-1]
        loads = loads.minimum(torch.tensor(self.max_possible_load))
        loads_one_hot = F.one_hot(loads.sort()[0], num_classes=self.max_possible_load + 1).double().to(self.device)
        choices_left_one_hot = F.one_hot(choices_left-2, num_classes=self.k - 1).double().to(self.device)
        after_rnn = self.rnn(loads_one_hot)[0][:, -1, :].squeeze(1)
        res = self.lin(torch.cat([after_rnn,choices_left_one_hot], dim=1))
        return res
