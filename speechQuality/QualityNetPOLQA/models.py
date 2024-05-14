import time

import torch
import torch.nn as nn
from einops.einops import rearrange
from einops.layers.torch import Rearrange
from keras.applications.densenet import layers

from blocks import TCNLayer


class TimeRestrictedAttention(nn.Module):
    def __init__(self, d_model, dq, heads=4, offset=64):
        super(TimeRestrictedAttention, self).__init__()
        # self.wq = nn.Linear(d_model, heads * dq)
        # self.wk = nn.Linear(d_model, heads * dq)
        # self.wv = nn.Linear(d_model, heads * dq)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.offset = offset
        self.arrange = Rearrange('B L (H C) -> B H L C', H=heads)
        self.out = nn.Sequential(
            Rearrange('B H L C -> B L (H C)', H=heads),
            nn.Linear(heads * dq, self.d_model),
        )

    def forward(self, x, mask):
        # mask = self.get_attn_mask(x.shape[1])
        # q = self.arrange(self.wq(x))
        # k = self.arrange(self.wk(x))
        # v = self.arrange(self.wv(x))
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        attn = torch.matmul(q, k.transpose(-1, -2)) / 256
        attn = torch.softmax(attn, dim=-1)
        attn = attn * mask
        context = torch.bmm(attn, v)
        # context = torch.bmm(attn, v)
        # context = self.out(context)
        return context, attn


class MyMultiheadAttention(nn.Module):
    def __init__(self, d_model=200, heads=4, offset=64):
        super(MyMultiheadAttention, self).__init__()
        # self.input = nn.Linear(d_model, 256)
        self.attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        # self.output = nn.Linear(256, d_model)

    def forward(self, x, mask):
        # x = self.input(x)
        x, attn = self.attn(x, x, x)
        # x = self.output(x)
        return x, attn


def get_attn_mask(seq_len, offset):
    mask = torch.ones([seq_len, seq_len], dtype=torch.bool)
    mask = torch.tril(mask, offset) * torch.triu(mask, -offset)
    return mask


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, dilation, feature_dim=128):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, feature_dim, 1, dilation=dilation)
        self.conv2 = nn.Conv1d(feature_dim, feature_dim, 3, dilation=dilation, padding="same")
        self.conv3 = nn.Conv1d(feature_dim, out_channels, 1, dilation=dilation)
        self.pool = nn.MaxPool1d(pool_size)
        self.rct = nn.ReLU()
        # self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual_x = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.rct((self.pool(residual_x + x)))


class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, dilation_size, feature_dim=128):
        super(ConvBlock2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, feature_dim, 1, dilation=1)
        self.conv2 = nn.Conv1d(feature_dim, feature_dim, 3, dilation=dilation_size, padding="same")
        self.conv3 = nn.Conv1d(feature_dim, out_channels, 1, dilation=1)
        self.pool = nn.MaxPool1d(pool_size)
        self.rct = nn.ReLU()
        # self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.rct((self.pool(x)))


class Cnn(nn.Module):
    """
    CNN model
    input shape: (N, C, L)
    """

    def __init__(self, filter_size, feature_dim, dropout):
        super(Cnn, self).__init__()
        # filter_size = 128
        # feature_dim = 64
        self.prepare = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.Conv1d(in_channels=257, out_channels=filter_size, kernel_size=5),
            nn.ELU(),
        )
        self.conv1 = ConvBlock(filter_size, filter_size, pool_size=4, feature_dim=feature_dim, dilation=64)
        self.conv2 = ConvBlock(filter_size, filter_size, pool_size=4, feature_dim=feature_dim, dilation=32)
        self.conv3 = ConvBlock(filter_size, filter_size, pool_size=2, feature_dim=feature_dim, dilation=16)
        self.conv4 = ConvBlock(filter_size, filter_size, pool_size=2, feature_dim=feature_dim, dilation=8)

        # self.conv2 = ConvBlock(256, 256)
        # self.conv3 = ConvBlock(64, 1)
        # self.mlp = nn.Sequential(
        #     nn.Linear(128, 64),
        #     # nn.Dropout(0.2),
        #     nn.Linear(64, 1),
        #     # nn.ELU()
        # )
        self.avg_pool = nn.Sequential(nn.Flatten(), nn.AdaptiveAvgPool1d(1))
        self.avg_linear = nn.Sequential(
            Rearrange("N C L -> N L C"),
            nn.LayerNorm(filter_size),
            Rearrange("N L C -> N C L"),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            # nn.Dropout(dropout),
            nn.Linear(filter_size, 50),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(50, 1),

        )

    def forward(self, x):
        x = self.prepare(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Frame_score = self.mlp(rearrange(x, "N C L -> N L C")).squeeze(-1)
        Average_score = self.avg_linear(x)
        # Average_score = torch.clamp(Average_score, min=1, max=5)
        return Average_score


class CnnClass(nn.Module):
    """
    CNN model
    input shape: (N, C, L)
    """

    def __init__(self, dropout=0.3, step=0.2):
        super(CnnClass, self).__init__()
        filter_size = 256
        feature_dim = 128
        self.prepare = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.Conv1d(in_channels=257, out_channels=filter_size, kernel_size=5),
            # nn.BatchNorm1d(num_features=256),
            nn.ELU(),
        )
        self.conv1 = ConvBlock2(filter_size, filter_size, pool_size=4, feature_dim=feature_dim, dilation_size=64)
        self.conv2 = ConvBlock2(filter_size, filter_size, pool_size=4, feature_dim=feature_dim, dilation_size=32)
        self.conv3 = ConvBlock2(filter_size, filter_size, pool_size=2, feature_dim=feature_dim, dilation_size=16)
        self.conv4 = ConvBlock2(filter_size, filter_size, pool_size=2, feature_dim=feature_dim, dilation_size=8)
        # self.conv2 = ConvBlock(256, 256)
        # self.conv3 = ConvBlock(64, 1)
        # self.mlp = nn.Sequential(
        #     nn.Linear(128, 64),
        #     # nn.Dropout(0.2),
        #     nn.Linear(64, 1),
        #     # nn.ELU()
        # )
        self.avg_pool = nn.Sequential(
            # nn.BatchNorm1d(filter_size),
            nn.AdaptiveAvgPool1d(1),
            Rearrange("N C L -> N (C L)"),
            nn.Linear(in_features=filter_size, out_features=128),
            nn.Dropout(dropout),
            nn.Linear(in_features=128, out_features=int(400 // int(step * 100))),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.prepare(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Frame_score = self.mlp(rearrange(x, "N C L -> N L C")).squeeze(-1)
        Average_score = self.avg_pool(x)
        # Average_score = torch.clamp(Average_score, min=1, max=5)
        return torch.tensor(0), Average_score


class TCN(nn.Module):
    def __init__(self, ):
        super(TCN, self).__init__()
        self.prepare = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.Conv1d(in_channels=257, out_channels=128, kernel_size=5),
            # nn.BatchNorm1d(num_features=256),
            nn.ELU(),
        )
        self.net = TCNLayer(128, 128)

        self.pool = nn.Sequential(nn.Flatten(), nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        x = self.prepare(x)
        x = self.net(x)
        Average_score = self.pool(x)
        return torch.tensor(0), Average_score


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
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        l1 = self.dropout(self.elu(self.linear1(lstm_out)))
        Frame_score = self.linear2(l1).squeeze(-1)
        Average_score = self.pool(Frame_score)
        return Frame_score, Average_score


class QualityNetAttn(nn.Module):
    """
    QualityNet with attention model
    input shape: (N, L, C)
    """

    def __init__(self, dropout=0.3) -> None:
        super(QualityNetAttn, self).__init__()
        self.lstm = nn.LSTM(257, 100, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(200, 50)  # 2 * 100
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(50, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.attn = TimeRestrictedAttention(200, 64, offset=64)
        # self.attn = MyMultiheadAttention()
        self.mask = None

    def forward(self, x):
        if self.mask is None:
            self.mask = get_attn_mask(x.shape[1], offset=64).to(x.device)
        lstm_out, _ = self.lstm(x)
        lstm_out, attn = self.attn(lstm_out, self.mask)
        l1 = self.dropout(self.elu(self.linear1(lstm_out)))
        # l1, attn = self.attn(l1, self.mask)
        Frame_score = self.linear2(l1).squeeze(-1)
        Average_score = self.pool(Frame_score)
        return Frame_score, Average_score


class QualityNetClassifier(nn.Module):
    """
    model name: lstmClass
    """

    def __init__(self, dropout=0.3, step=0.2) -> None:
        super(QualityNetClassifier, self).__init__()
        self.lstm = nn.LSTM(257, 100, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(200, 100)  # 2 * 100
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(100, int(400 // int(step * 100)))
        self.pool = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        l1 = self.dropout(self.elu(self.linear1(lstm_out)))
        Frame_score = self.linear2(l1)
        Average_score = self.pool(Frame_score)
        return torch.softmax(Frame_score, dim=-1), torch.softmax(Average_score, dim=-1)


class QualityNetClassifier2(nn.Module):
    """
    model name: lstmClass2
    """

    def __init__(self, dropout=0.3, step=0.2) -> None:
        super(QualityNetClassifier2, self).__init__()
        self.lstm = nn.LSTM(257, 100, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(200, 50)  # 2 * 100
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Sequential(nn.Linear(50, 1), nn.Flatten())
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        l1 = self.dropout(self.elu(self.linear1(lstm_out)))
        Frame_score = self.linear2(l1)
        Average_score = self.pool(Frame_score)
        return Frame_score, Average_score


if __name__ == '__main__':
    # model = TimeRestrictedAttention(257, 128)
    model = Cnn(128,64, 0.3)
    x = torch.randn((4, 1024, 257))
    y = model(x)
    print(y[0].shape, y[1].shape)
