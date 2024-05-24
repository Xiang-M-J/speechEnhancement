import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


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
        # mask_target = self.get_attn_mask(x.shape[1])
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
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual_x = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.rct(self.bn(self.pool(residual_x + x)))


class Cnn2(nn.Module):
    def __init__(self):
        super(Cnn2, self).__init__()
        self.conv0 = nn.Conv1d(257, 512, 3, 2)
        self.norm0 = nn.GroupNorm(512, 512)
        self.conv1 = nn.Conv1d(512, 512, 3, 2)
        self.conv2 = nn.Conv1d(512, 512, 3, 2)
        self.conv3 = nn.Conv1d(512, 512, 2, 2)
        self.conv4 = nn.Conv1d(512, 512, 2, 2)
        self.conv5 = nn.Conv1d(512, 512, 2, 2)
        self.gelu = nn.GELU()
        self.linear = nn.Sequential(
            Rearrange("N C L -> N L C"),
            nn.Linear(512, 64),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Flatten()
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.gelu(self.norm0(self.conv0(x)))
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))
        x = self.gelu(self.conv4(x))
        x = self.gelu(self.conv5(x))
        Frame = self.linear(x)
        Avg = self.pool(Frame)
        return Frame, Avg


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
            Rearrange("N C L -> N L C"),
            nn.LayerNorm(filter_size),
            Rearrange("N L C -> N C L"),
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
            nn.Conv2d(32, 64, (3,3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )
        self.avg = nn.Sequential(
            nn.Conv2d(64, 128, (3,3)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d((2,2)),
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


class SEBlock(nn.Module):
    """
    inp: N C L
    """

    def __init__(self, filter_size, feature_dim, pool, dilation):
        super(SEBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(filter_size, feature_dim, kernel_size=1, dilation=dilation),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, dilation=dilation, padding="same"),
            nn.Conv1d(feature_dim, filter_size, kernel_size=1, dilation=dilation),
            nn.ReLU()
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(filter_size, filter_size // 32),
            nn.Linear(filter_size // 32, filter_size),
            nn.Sigmoid(),
            Rearrange("N (C H) -> N C H", H=1)
        )
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x):
        y = self.conv(x)
        attn = self.se(y)
        y = y * attn
        return self.pool(self.act(x + y))


class CnnAttn(nn.Module):
    def __init__(self, filter_size, feature_dim, dropout):
        super(CnnAttn, self).__init__()
        self.prepare = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.Conv1d(in_channels=257, out_channels=filter_size, kernel_size=5),
            nn.ELU(),
        )
        self.conv1 = SEBlock(filter_size, pool=4, feature_dim=feature_dim, dilation=64)
        self.conv2 = SEBlock(filter_size, pool=4, feature_dim=feature_dim, dilation=32)
        self.conv3 = SEBlock(filter_size, pool=2, feature_dim=feature_dim, dilation=16)
        self.conv4 = SEBlock(filter_size, pool=2, feature_dim=feature_dim, dilation=8)

        self.avg_pool = nn.Sequential(nn.Flatten(), nn.AdaptiveAvgPool1d(1))
        self.avg_linear = nn.Sequential(
            Rearrange("N C L -> N L C"),
            nn.LayerNorm(filter_size),
            Rearrange("N L C -> N C L"),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(filter_size, 50),
            nn.Dropout(dropout),
            nn.Linear(50, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.prepare(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_linear(x)
        return x


class CANBlock(nn.Module):
    def __init__(self, channel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv1d(channel, channel, kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size=2, stride=2)
        self.pool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(channel)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        r_x = x
        x = self.conv1(x)
        x = self.conv2(x)
        r_x = self.pool(r_x)
        return self.dropout(self.relu(r_x + x))


class CANClass(nn.Module):
    def __init__(self, channels,feature_dim, step):
        super(CANClass, self).__init__()
        self.input = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.Conv1d(in_channels=257, out_channels=channels, kernel_size=4, stride=2),
            nn.GroupNorm(channels, channels),
            nn.PReLU()
        )

        self.conv1 = CANBlock(channels)
        self.conv2 = CANBlock(channels)
        self.middle1 = nn.Conv1d(channels, feature_dim, kernel_size=1)
        self.conv3 = CANBlock(feature_dim)

        self.avg = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, int(400 // int(step * 100))),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.middle1(x)
        x = self.conv3(x)
        avg = self.avg(x)
        c = self.classifier(x)
        return avg, c

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
            nn.Conv2d(32, 64, (3,3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )
        self.avg = nn.Sequential(
            nn.Conv2d(64, 128, (3,3)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d((2,2)),
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

    def __init__(self, dropout=0.3, step=0.2):
        super(CnnClass, self).__init__()
        filter_size = 128
        feature_dim = 64
        self.prepare = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.Conv1d(in_channels=257, out_channels=filter_size, kernel_size=5),
            nn.BatchNorm1d(num_features=filter_size),
            nn.LeakyReLU(),
        )
        self.conv1 = ConvBlock(filter_size, filter_size, pool_size=4, feature_dim=feature_dim, dilation=64)
        self.conv2 = ConvBlock(filter_size, filter_size, pool_size=4, feature_dim=feature_dim, dilation=32)
        self.conv3 = ConvBlock(filter_size, filter_size, pool_size=2, feature_dim=feature_dim, dilation=16)
        self.conv4 = ConvBlock(filter_size, filter_size, pool_size=2, feature_dim=feature_dim, dilation=8)
        self.avg = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_features=filter_size, out_features=int(400 // int(step * 100))),
        )

    def forward(self, x):
        x = self.prepare(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Frame_score = self.mlp(rearrange(x, "N C L -> N L C")).squeeze(-1)
        Average_score = self.avg(x)
        # Average_score = torch.clamp(Average_score, min=1, max=5)
        return torch.tensor(0), Average_score


class QualityNet(nn.Module):
    """
    QualityNet model
    inp shape: (N, L, C)
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
    inp shape: (N, L, C)
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
        # l1, attn = self.attn(l1, self.mask_target)
        Frame_score = self.linear2(l1).squeeze(-1)
        Average_score = self.pool(Frame_score)
        return Frame_score, Average_score

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



class QualityNetClassifier(nn.Module):
    """
    model name: lstmClass
    """

    def __init__(self, dropout=0.3, step=0.2) -> None:
        super(QualityNetClassifier, self).__init__()
        self.lstm = nn.LSTM(257, 100, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)
        self.linear1 = nn.Sequential(
            nn.Linear(200, 100),  # 2 * 100
            nn.ELU(),
            nn.Dropout(dropout)
        )

        self.linear2 = nn.Sequential(
            nn.LayerNorm(100),
            nn.Linear(100, int(400 // int(step * 100)))
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
        self.blstm = nn.LSTM(input_size=257,
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

        out, _ = self.blstm(x)  # (B,T, 2*hidden)
        out = self.dropout(self.act_fn(self.linear1(out))).transpose(0, 1)  #(T_length, B,  128)
        hasqi, _ = self.hasqiAtt_layer(out, out, out)
        hasqi = hasqi.transpose(0, 1)  # (B, T_length, 128)
        hasqi = self.ln(hasqi)
        hasqi = self.hasqiframe_score(hasqi)  # (B, T_length, 1)
        # hasqi = self.act(hasqi)  # pass a sigmoid
        hasqi_fram = hasqi.permute(0, 2, 1)  # (B, 1, T_length)
        hasqi_avg = self.hasqiaverage_score(hasqi_fram)  # (B,1,1)

        return hasqi_fram, hasqi_avg.squeeze(1)  # (B, 1, T_length) (B,1)


if __name__ == '__main__':
    # model = TimeRestrictedAttention(257, 128)
    model = LstmCANClass(0.1, 20)
    x = torch.randn((4, 1024, 257))
    import time

    t1 = time.time()
    for i in range(100):
        y = model(x)
    t2 = time.time()
    print(t2 - t1)
    print(y.shape)
