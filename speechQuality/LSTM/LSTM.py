import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np

# fix random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

class lstm_net(nn.Module):
    def __init__(self, fft_size):
        super(lstm_net, self).__init__()
        freq_len = fft_size // 2 + 1
        self.bn = nn.BatchNorm1d(freq_len)
        self.lstm1 = nn.LSTM(input_size=freq_len, hidden_size=1024, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(in_features=1024 * 2, out_features=freq_len),
            nn.Softplus())

    def forward(self, x):
        x = self.bn(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        return x