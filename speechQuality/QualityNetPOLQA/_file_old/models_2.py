import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from _file_old.blocks import TCNLayer


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


class Cnn(nn.Module):
    """
    CNN model
    input shape: (N, C, L)
    """

    def __init__(self, filter_size, feature_dim, dropout, n_o=False):
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
            nn.Sigmoid(),
        )
        self.n_o = n_o
        self.o = nn.Sigmoid()

    def forward(self, x):
        x = self.prepare(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Frame_score = self.mlp(rearrange(x, "N C L -> N L C")).squeeze(-1)
        Average_score = self.avg_linear(x)
        # Average_score = torch.clamp(Average_score, min=1, max=5)
        if self.n_o:
            return self.o(Average_score)
        else:
            return Average_score


class SEBlock(nn.Module):
    """
    input: N C L
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
    def __init__(self, filter_size, feature_dim, dropout, n_o=False):
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

        )
        self.n_o = n_o
        self.o = nn.Sigmoid()

    def forward(self, x):
        x = self.prepare(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_linear(x)
        if self.n_o:
            return self.o(x)
        else:
            return x


class CnnClass(nn.Module):
    """
    CNN model
    input shape: (N, C, L)
    """

    def __init__(self, dropout=0.3, step=0.2):
        super(CnnClass, self).__init__()
        filter_size = 128
        feature_dim = 64
        self.prepare = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.Conv1d(in_channels=257, out_channels=filter_size, kernel_size=5),
            # nn.BatchNorm1d(num_features=256),
            nn.ELU(),
        )
        self.conv1 = ConvBlock(filter_size, filter_size, pool_size=4, feature_dim=feature_dim, dilation=64)
        self.conv2 = ConvBlock(filter_size, filter_size, pool_size=4, feature_dim=feature_dim, dilation=32)
        self.conv3 = ConvBlock(filter_size, filter_size, pool_size=2, feature_dim=feature_dim, dilation=16)
        self.conv4 = ConvBlock(filter_size, filter_size, pool_size=2, feature_dim=feature_dim, dilation=8)
        self.avg_pool = nn.Sequential(
            # nn.BatchNorm1d(filter_size),
            nn.AdaptiveAvgPool1d(1),
            Rearrange("N C L -> N (C L)"),
            # nn.Linear(in_features=filter_size, out_features=128),
            # nn.Dropout(dropout),
            nn.Linear(in_features=filter_size, out_features=int(400 // int(step * 100))),
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
    def __init__(self, n_o=False):
        super(TCN, self).__init__()
        self.prepare = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.Conv1d(in_channels=257, out_channels=128, kernel_size=5),
            # nn.BatchNorm1d(num_features=256),
            nn.ELU(),
        )
        self.net = TCNLayer(128, 128)

        self.pool = nn.Sequential(nn.Flatten(), nn.AdaptiveAvgPool1d(1))
        self.n_o = n_o
        self.o = nn.Sigmoid()

    def forward(self, x):
        x = self.prepare(x)
        x = self.net(x)
        Average_score = self.pool(x)
        if self.n_o:
            return self.o(Average_score)
        else:
            return Average_score


class QualityNet(nn.Module):
    """
    QualityNet model
    input shape: (N, L, C)
    """

    def __init__(self, dropout=0.3, n_o=False) -> None:
        super(QualityNet, self).__init__()
        self.lstm = nn.LSTM(257, 100, num_layers=2, bidirectional=True, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(200, 50)  # 2 * 100
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(50, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.n_o = n_o
        self.o = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        l1 = self.dropout(self.elu(self.linear1(lstm_out)))
        Frame_score = self.linear2(l1).squeeze(-1)
        Average_score = self.pool(Frame_score)
        if self.n_o:
            return self.o(Frame_score), self.o(Average_score)
        else:
            return Frame_score, Average_score


class QualityNetAttn(nn.Module):
    """
    QualityNet with attention model
    input shape: (N, L, C)
    """

    def __init__(self, dropout=0.3, n_o=False) -> None:
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
        self.n_o = n_o
        self.o = nn.Sigmoid()

    def forward(self, x):
        if self.mask is None:
            self.mask = get_attn_mask(x.shape[1], offset=64).to(x.device)
        lstm_out, _ = self.lstm(x)
        lstm_out, attn = self.attn(lstm_out, self.mask)
        l1 = self.dropout(self.elu(self.linear1(lstm_out)))
        # l1, attn = self.attn(l1, self.mask)
        Frame_score = self.linear2(l1).squeeze(-1)
        Average_score = self.pool(Frame_score)
        if self.n_o:
            return self.o(Frame_score), self.o(Average_score)
        else:
            return Frame_score, Average_score


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
        return torch.softmax(Frame_score, dim=-1), torch.softmax(Average_score, dim=-1)


class HASANet(nn.Module):
    """
    input_size: 257
    hidden_size: 100
    num_layers: 1
    dropout: 0
    linear_output: 128
    act_fn: 'relu'
    """
    def __init__(self, n_o=False):
        super(HASANet, self).__init__()
        hidden_size = 100
        num_layers = 1
        dropout = 0.
        linear_output = 128
        self.n_o = n_o
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
        self.ln = nn.LayerNorm(128)
        self.hasqiframe_score = nn.Linear(128, 1, bias=True)
        # self.act = nn.LeakyReLU()
        self.hasqiaverage_score = nn.AdaptiveAvgPool1d(1)
        self.o = nn.Sigmoid()

    def forward(self, x):  # hl:(B,6)
        B, T, Freq = x.size()

        out, _ = self.blstm(x)  # (B,T, 2*hidden)
        out = self.dropout(self.act_fn(self.linear1(out))).transpose(0, 1)  #(T_length, B,  128)
        hasqi, _ = self.hasqiAtt_layer(out, out, out)
        hasqi = hasqi.transpose(0, 1)  # (B, T_length, 128)
        hasqi = self.ln(hasqi)
        hasqi = self.hasqiframe_score(hasqi)  # (B, T_length, 1)
        # hasqi = self.act(hasqi)  # pass a sigmoid
        hasqi_fram = hasqi.permute(0, 2, 1)  # (B, 1, T_length)
        hasqi_avg = self.hasqiaverage_score(hasqi_fram)  # (B,1,1)
        if self.n_o:
            return self.o(hasqi_fram), self.o(hasqi_avg.squeeze(1))  # (B, 1, T_length) (B,1)
        else:
            return hasqi_fram, hasqi_avg.squeeze(1)  # (B, 1, T_length) (B,1)


if __name__ == '__main__':
    # model = TimeRestrictedAttention(257, 128)
    model = Cnn(128, 64, 0.3)
    x = torch.randn((4, 1024, 257))
    y = model(x)
    print(y[0].shape, y[1].shape)
