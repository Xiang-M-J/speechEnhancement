import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size, dilation, feature_dim=128):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, feature_dim, 1, dilation=dilation)
        self.conv2 = nn.Conv1d(feature_dim, feature_dim, 2, dilation=dilation, padding="same")
        self.conv3 = nn.Conv1d(feature_dim, out_channels, 1, dilation=dilation)
        self.pool = nn.MaxPool1d(pool_size)
        self.rct = nn.ReLU()
        # self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual_x = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return self.rct(self.bn(self.pool(residual_x + x)))


class Cnn(nn.Module):
    """
    CNN model
    inp shape: (N, C, L)
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
        self.avg_pool = nn.Sequential(nn.Flatten(), nn.AdaptiveAvgPool1d(1))
        self.avg_linear = nn.Sequential(
            # Rearrange("N C L -> N L C"),
            # nn.LayerNorm(filter_size),
            # Rearrange("N L C -> N C L"),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(filter_size, 50),
            nn.Dropout(dropout),
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


class Cnn2d(nn.Module):
    def __init__(self):
        super(Cnn2d, self).__init__()
        self.prepare = Rearrange("N L (H C) -> N H C L", H=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, (3, 3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d((2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, (3, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d((2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )
        self.avg = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d((2, 2)),
            Rearrange("N H C L -> N (H C) L"),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.avg2 = nn.Sequential(
            nn.Linear(128 * 12, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.prepare(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avg(x)
        x = self.avg2(x)
        return x


class CnnMAttnStack(nn.Module):
    def __init__(self, ):
        super(CnnMAttnStack, self).__init__()
        self.prepare = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, ),
            Rearrange("N H L C -> N (H C) L"),
            nn.Conv1d(in_channels=257, out_channels=128, kernel_size=4, dilation=2),
            nn.ELU(),
        )
        self.conv1 = ConvBlock(128, 128, pool_size=2, feature_dim=64, dilation=64)
        self.conv2 = ConvBlock(128, 128, pool_size=2, feature_dim=64, dilation=32)
        self.conv3 = ConvBlock(128, 128, pool_size=2, feature_dim=64, dilation=16)
        self.conv4 = ConvBlock(128, 128, pool_size=2, feature_dim=64, dilation=8)

        self.attn = nn.MultiheadAttention(128, 8, 0.1, batch_first=True)
        self.avg_linear = nn.Sequential(
            nn.LayerNorm(128),
            Rearrange("N L C -> N C L"),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.prepare(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.permute([0, 2, 1])
        x, _ = self.attn(x, x, x)
        x = self.avg_linear(x)
        return x


class CnnMAttn(nn.Module):
    def __init__(self, ):
        super(CnnMAttn, self).__init__()
        self.prepare = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.Conv1d(in_channels=257, out_channels=128, kernel_size=4),
            nn.ELU(),
        )
        self.conv1 = ConvBlock(128, 128, pool_size=2, feature_dim=64, dilation=64)
        self.conv2 = ConvBlock(128, 128, pool_size=2, feature_dim=64, dilation=32)
        self.conv3 = ConvBlock(128, 128, pool_size=2, feature_dim=64, dilation=16)
        self.conv4 = ConvBlock(128, 128, pool_size=2, feature_dim=64, dilation=8)

        self.attn = nn.MultiheadAttention(128, 8, 0.2, batch_first=True)
        self.avg_linear = nn.Sequential(
            nn.LayerNorm(128),
            Rearrange("N L C -> N C L"),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.prepare(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.permute([0, 2, 1])
        x, _ = self.attn(x, x, x)
        x = self.avg_linear(x)
        return x


class CAN2dClass(nn.Module):
    def __init__(self, n_class):
        super(CAN2dClass, self).__init__()
        self.prepare = Rearrange("N L (H C) -> N H C L", H=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 16, (3, 3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d((2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, (3, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d((2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )
        self.avg = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d((2, 2)),
            Rearrange("N H C L -> N (H C) L"),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128 * 12, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        self.cls = nn.Sequential(
            Rearrange("N H C L -> N (H C) L"),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64 * 27, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, n_class)
        )

    def forward(self, x):
        x = self.prepare(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        avg = self.avg(x)
        c = self.cls(x)
        return avg, c


class CnnClass(nn.Module):
    """
    CNN model
    inp shape: (N, C, L)
    """

    def __init__(self, num_class):
        super(CnnClass, self).__init__()
        filter_size = 128
        feature_dim = 64
        self.prepare = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.Conv1d(in_channels=257, out_channels=filter_size, kernel_size=5),
            nn.ELU(),
        )
        self.conv1 = ConvBlock(filter_size, filter_size, pool_size=2, feature_dim=feature_dim, dilation=64)
        self.conv2 = ConvBlock(filter_size, filter_size, pool_size=2, feature_dim=feature_dim, dilation=32)
        self.conv3 = ConvBlock(filter_size, filter_size, pool_size=2, feature_dim=feature_dim, dilation=16)
        self.conv4 = ConvBlock(filter_size, filter_size, pool_size=2, feature_dim=feature_dim, dilation=8)
        self.score = nn.Sequential(
            # nn.Conv1d(128, 128, kernel_size=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(filter_size, 1)
        )
        self.cls = nn.Sequential(
            # nn.Conv1d(128, 128, kernel_size=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(filter_size, num_class),
        )

    def forward(self, x):
        x = self.prepare(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        score = self.score(x)
        c = self.cls(x)
        # Average_score = torch.clamp(Average_score, min=1, max=5)
        return score, c


class LstmCANClass(nn.Module):
    def __init__(self, dropout, num_class) -> None:
        super(LstmCANClass, self).__init__()
        self.lstm = nn.LSTM(257, 100, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)
        self.linear1 = nn.Sequential(
            nn.Linear(200, 100),  # 2 * 100
            nn.ELU(),
            nn.Dropout(dropout)
        )
        self.avg = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.AdaptiveAvgPool1d(128),
            nn.Linear(128, 1),
            nn.Flatten(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 1),
        )
        self.cls = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.AdaptiveAvgPool1d(128),
            nn.Linear(128, 1),
            nn.Flatten(),
            nn.BatchNorm1d(100),
            nn.Linear(100, num_class),
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear1(x)
        avg = self.avg(x)
        c = self.cls(x)
        return avg, c


class LstmClassifier(nn.Module):
    """
    model name: lstmClass
    """

    def __init__(self, dropout, num_class) -> None:
        super(LstmClassifier, self).__init__()
        self.lstm = nn.LSTM(257, 100, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)
        self.linear1 = nn.Sequential(
            nn.Linear(200, 100),  # 2 * 100
            nn.ELU(),
            nn.Dropout(dropout)
        )

        self.linear2 = nn.Sequential(
            nn.LayerNorm(100),
            nn.Linear(100, num_class)
        )
        self.pool = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.AdaptiveAvgPool1d(1),
            # nn.Linear(128, 1),
            nn.Flatten()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        l1 = self.linear1(lstm_out)
        Frame_score = self.linear2(l1)
        Average_score = self.pool(Frame_score)
        return Frame_score, Average_score


class HASANetStack(nn.Module):
    """
    input_size: 257
    hidden_size: 100
    num_layers: 1
    dropout: 0
    linear_output: 128
    act_fn: 'relu'
    """

    def __init__(self):
        super(HASANetStack, self).__init__()
        hidden_size = 100
        num_layers = 1
        dropout = 0.
        linear_output = 128
        self.blstm1 = nn.LSTM(input_size=257,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=True,
                              batch_first=True)
        self.blstm2 = nn.LSTM(input_size=257,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=True,
                              batch_first=True)
        self.linear1 = nn.Linear(hidden_size * 2, linear_output, bias=True)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.hasqiAtt_layer = nn.MultiheadAttention(linear_output, num_heads=8)
        self.ln = nn.LayerNorm(linear_output)
        self.hasqiframe_score = nn.Linear(linear_output, 1, bias=True)
        # self.act = nn.LeakyReLU()
        self.hasqiaverage_score = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # hl:(B,6)

        out1, _ = self.blstm1(x)  # (B,T, 2*hidden)
        out2, _ = self.blstm2(x)
        out = out1 + out2
        out = self.dropout(self.act_fn(self.linear1(out))).transpose(0, 1)  #(T_length, B,  128)
        hasqi, _ = self.hasqiAtt_layer(out, out, out)
        hasqi = hasqi.transpose(0, 1)  # (B, T_length, 128)
        hasqi = self.ln(hasqi)
        hasqi = self.hasqiframe_score(hasqi)  # (B, T_length, 1)
        # hasqi = self.act(hasqi)  # pass a sigmoid
        hasqi_fram = hasqi.permute(0, 2, 1)  # (B, 1, T_length)
        hasqi_avg = self.hasqiaverage_score(hasqi_fram)  # (B,1,1)

        return hasqi_fram, hasqi_avg.squeeze(1)  # (B, 1, T_length) (B,1)


class HASAClassifier(nn.Module):
    """
    input_size: 257
    hidden_size: 100
    num_layers: 1
    dropout: 0
    linear_output: 128
    act_fn: 'relu'
    """

    def __init__(self, num_class):
        super(HASAClassifier, self).__init__()
        hidden_size = 100
        num_layers = 1
        dropout = 0.
        linear_output = 128
        self.biLstm = nn.LSTM(input_size=257,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=True,
                              batch_first=True)
        self.linear1 = nn.Linear(hidden_size * 2, linear_output, bias=True)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.attn = nn.MultiheadAttention(linear_output, num_heads=8, dropout=0.1, batch_first=True)
        self.ln = nn.LayerNorm(linear_output)
        self.cls = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(linear_output, num_class),
        )
        self.score = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(linear_output, 1),
        )

    def forward(self, x):
        x, _ = self.biLstm(x)
        x = self.dropout(self.act_fn(self.linear1(x)))
        x, _ = self.attn(x, x, x)
        x = self.ln(x)
        score = self.score(x)
        c = self.cls(x)

        return score, c
        return c, c


class HASANet(nn.Module):
    """
    input_size: 257
    hidden_size: 100
    num_layers: 1
    dropout: 0
    linear_output: 128
    act_fn: 'relu'
    """

    def __init__(self):
        super(HASANet, self).__init__()
        hidden_size = 100
        num_layers = 1
        dropout = 0.
        linear_output = 128
        self.biLstm = nn.LSTM(input_size=257,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=True,
                              batch_first=True)
        self.linear1 = nn.Linear(hidden_size * 2, linear_output, bias=True)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.attn = nn.MultiheadAttention(linear_output, num_heads=8)
        self.ln = nn.LayerNorm(linear_output)
        self.frame = nn.Linear(linear_output, 1, bias=True)
        # self.act = nn.LeakyReLU()
        self.average = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # hl:(B,6)
        # x = self.prepare(x)
        x, _ = self.biLstm(x)  # (B,T, 2*hidden)
        x = self.dropout(self.act_fn(self.linear1(x))).transpose(0, 1)  # (T_length, B,  128)
        x, _ = self.attn(x, x, x)
        x = x.transpose(0, 1)  # (B, T_length, 128)
        x = self.ln(x)

        x = self.frame(x)  # (B, T_length, 1)
        # x = self.act(x)  # pass a sigmoid
        frame_score = x.permute(0, 2, 1)  # (B, 1, T_length)
        average_score = self.average(frame_score)  # (B,1,1)

        return frame_score.squeeze(1), average_score.squeeze(1)  # (B, 1, T_length) (B,1)


if __name__ == '__main__':
    model = LstmCANClass(0.1, 20)
    x = torch.randn((4, 1024, 257))
    import time

    t1 = time.time()
    for i in range(100):
        y = model(x)
    t2 = time.time()
    print(t2 - t1)
    print(y.shape)
