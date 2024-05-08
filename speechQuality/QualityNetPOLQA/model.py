import torch
import torch.nn as nn
from einops.einops import rearrange


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding="same")
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        return self.relu(x)


class Cnn(nn.Module):
    """
    CNN model
    input shape: (N, C, L)
    """
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = ConvBlock(257, 64)
        self.conv2 = ConvBlock(64, 8)
        self.conv3 = ConvBlock(8, 1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = rearrange(x, "N L C -> N C L")
        x = self.conv1(x)
        x = self.conv2(x)
        Frame_score = self.conv3(x).squeeze(1)
        Average_score = self.avgpool(Frame_score)
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
