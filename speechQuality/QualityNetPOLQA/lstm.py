import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class lstm_net(nn.Module):
    def __init__(self, fft_size):
        super(lstm_net, self).__init__()
        freq_len = fft_size // 2 + 1
        # self.ln = nn.LayerNorm(freq_len)
        # self.lstm1 = nn.LSTM(input_size=freq_len, hidden_size=512, num_layers=1, batch_first=True)
        self.input = nn.Sequential(
            Rearrange("N L C -> N C L"),

            nn.Conv1d(freq_len, 512, 1),
            nn.BatchNorm1d(512),
            Rearrange("N C L -> N L C"),
        )
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(in_features=512 * 2, out_features=freq_len),
            nn.Softplus())

    def forward(self, x):
        # x = self.ln(x)
        x = self.input(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = lstm_net(fft_size=512)
    x = torch.randn([4, 256, 257])
    y = model(x)
    print(y.shape)
