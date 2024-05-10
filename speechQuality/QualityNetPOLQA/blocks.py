from itertools import repeat

import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class SpatialDropout(nn.Module):
    """
    空间dropout，即在指定轴方向上进行dropout，常用于Embedding层和CNN层后
    如对于(batch, timestamps, embedding)的输入，若沿着axis=1则可对embedding的若干channel进行整体dropout
    若沿着axis=2则可对某些token进行整体dropout
    """

    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop

    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple, 应当与inputs的shape一致，其中值为1的即沿着drop的轴
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim() - 2), inputs.shape[-1])  # 默认沿着中间所有的shape
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)
            outputs.mul_(noises)
            return outputs

    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)


class Temporal_Aware_Block(nn.Module):
    def __init__(self, feature_dim, filters, dilation, kernel_size, dropout=0.):
        super(Temporal_Aware_Block, self).__init__()
        padding = (kernel_size - 1) * dilation
        conv1 = (nn.Conv1d(in_channels=feature_dim, out_channels=filters,
                           kernel_size=kernel_size, padding=padding, dilation=dilation))
        chomp1 = Chomp1d(padding)  # 截断输出
        bn1 = nn.BatchNorm1d(filters)
        act1 = nn.ReLU()
        dropout1 = SpatialDropout(dropout)
        conv2 = (nn.Conv1d(in_channels=filters, out_channels=filters,
                           kernel_size=kernel_size, padding=padding, dilation=dilation))
        chomp2 = Chomp1d(padding)
        bn2 = nn.BatchNorm1d(filters)

        act2 = nn.ReLU()
        dropout2 = SpatialDropout(dropout)

        self.casual = nn.Sequential(
            conv1, chomp1, bn1, act1, dropout1, conv2, chomp2, bn2, act2, dropout2
        )
        self.act = nn.ReLU()
        self.resample = nn.Conv1d(in_channels=feature_dim, out_channels=filters, kernel_size=1, padding="same")

    def forward(self, x):  # input_shape: [batch_size, feature_dim, seq_len]
        identity_x = x
        x = self.casual(x)
        if identity_x.shape[1] != x.shape[1]:
            identity_x = self.resample(identity_x)
        x = torch.add(x, identity_x)
        return self.act(x)


class WeightLayer(nn.Module):
    def __init__(self, dilation=8):
        super().__init__()
        self.param = nn.parameter.Parameter(torch.randn(dilation, 1), requires_grad=True)

    def forward(self, x):
        x = torch.matmul(x, self.param)
        return x


class TCNLayer(nn.Module):  # Conv1d Input:[batch_size, feature_dim, seq_len]
    """
    Input: [batch_size, feature_dim, seq_len] \n
    Output: [batch_size, filters, seq_len]
    """

    def __init__(self, in_channel, channel=256, dilation=8, dropout=0.3):
        super(TCNLayer, self).__init__()

        self.dilation_layer = nn.ModuleList([])  # ModuleList存放Module，方便后面add_graph

        for i in [2 ** i for i in range(dilation)]:
            self.dilation_layer.append(
                Temporal_Aware_Block(feature_dim=channel, filters=channel, kernel_size=3,
                                     dilation=i, dropout=dropout)
            )


    def forward(self, x):
        for layer in self.dilation_layer:
            x = layer(x)
        return x  # o


class TIM(nn.Module):  # Conv1d Input:[batch_size, feature_dim, seq_len]
    """
    Input: [batch_size, feature_dim, seq_len] \n
    Output: [batch_size, filters, seq_len]
    """

    def __init__(self, in_channel, channel=256, dilation=8, dropout=0.3):
        super(TIM, self).__init__()

        self.dilation_layer = nn.ModuleList([])  # ModuleList存放Module，方便后面add_graph
        # padding = 0  # 下面的因果卷积的dilation=1 kernel_size=1
        # self.conv = nn.Sequential(
        #     nn.Conv1d(in_channels=in_channel, out_channels=channel, kernel_size=1),
        #     nn.BatchNorm1d(channel),
        #     nn.ReLU()
        # )
        # chomp = Chomp1d(padding)
        # self.causalConv = nn.Sequential(conv)

        for i in [2 ** i for i in range(dilation)]:
            self.dilation_layer.append(
                Temporal_Aware_Block(feature_dim=channel, filters=channel, kernel_size=3,
                                     dilation=i, dropout=dropout)
            )
        self.drop = nn.Dropout(p=dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.weight = WeightLayer(dilation)

    def forward(self, x):
        x_f = x
        x_b = torch.flip(x, dims=[-1])
        # x_f = self.conv(x_f)
        # x_b = self.conv(x_b)
        skip_stack = None
        skip_out_f = x_f
        skip_out_b = x_b

        for layer in self.dilation_layer:
            skip_out_f = layer(skip_out_f)
            skip_out_b = layer(skip_out_b)
            skip_temp = torch.add(skip_out_f, skip_out_b)
            skip_temp = skip_temp.unsqueeze(-1)
            if skip_stack is None:
                skip_stack = skip_temp
            else:
                skip_stack = torch.cat((skip_stack, skip_temp), dim=-1)
        skip_stack = self.weight(skip_stack)
        skip_stack = skip_stack.squeeze(-1)
        return skip_stack  # output shape: [batch_size, feature_dim, seq_len]


if __name__ == "__main__":
    pass
