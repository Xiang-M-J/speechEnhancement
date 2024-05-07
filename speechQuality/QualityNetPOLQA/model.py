import torch
import torch.nn as nn


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
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = ConvBlock(257, 64)
        self.conv2 = ConvBlock(64, 8)
        self.conv3 = ConvBlock(8, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


if __name__ == '__main__':
    model = Cnn()
    x = torch.randn((4, 257, 128))
    y = model(x)
    print(y.shape)
