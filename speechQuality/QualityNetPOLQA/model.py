import torch
import torch.nn as nn
from einops.einops import rearrange
from einops.layers.torch import Rearrange


class timeRestirctedAttention(nn.Module):
    def __init__(self, d_model, dq, heads=8, offset=64):
        super(timeRestirctedAttention, self).__init__()
        self.wq = nn.Linear(d_model, heads * dq)
        self.wk = nn.Linear(d_model, heads * dq)
        self.wv = nn.Linear(d_model, heads * dq)
        self.arrange = Rearrange('B L (H C) -> B H L C')

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=64):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.conv2 = nn.Conv1d(mid_channels, mid_channels, 3, padding="same")
        self.conv2 = nn.Conv1d(mid_channels, out_channels, 1)
        self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.relu(x + res)


class Cnn(nn.Module):
    """
    CNN model
    input shape: (N, C, L)
    """

    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = ConvBlock(257, 128)
        self.conv2 = ConvBlock(128, 64)
        # self.conv3 = ConvBlock(64, 1)
        self.linear = nn.Linear(64, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = rearrange(x, "N L C -> N C L")
        x = self.conv1(x)
        x = self.conv2(x)

        Frame_score = self.linear(rearrange(x, "N C L -> N L C")).squeeze(-1)
        Average_score = self.avg_pool(Frame_score)
        return Frame_score, Average_score


class QualityNet(nn.Module):
    """
    QualityNet model
    input shape: (N, L, C)
    """

    def __init__(self, dropout=0.3) -> None:
        super(QualityNet, self).__init__()
        self.lstm = nn.LSTM(257, 100, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(200, 50)  # 2 * 100
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(50, 1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        l1 = self.dropout(self.elu(self.linear1(lstm_out)))
        Frame_score = self.linear2(l1).squeeze(-1)
        Average_score = self.avgpool(Frame_score)
        return Frame_score, Average_score


if __name__ == '__main__':
    model = QualityNet()
    x = torch.randn((4, 128, 257))
    y = model(x)
    print(y[0].shape, y[1].shape)
