import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class Hubert(nn.Module):
    def __init__(self, d_model: int = 768, dim_feedforward: int = 1024,
                 n_heads: int = 8, num_layer: int = 8):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.feature_projection = FeatureProjection(d_model)
        self.positional_embedding = PositionalConvEmbedding(d_model, 64, 8)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.encoder = TransformerEncoder(d_model, n_heads, dim_feedforward, num_layer)

        self.output = nn.Sequential(
            Rearrange("N L C -> N C L"),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 1),
        )

    def encode(self, x: torch.Tensor):
        x = self.feature_extractor(x)
        x = self.feature_projection(x.transpose(1, 2))
        x = x + self.positional_embedding(x)
        x = self.dropout(self.norm(x))
        x = self.encoder(x)
        return x

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        output = self.output(x)
        return output


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv1d(1, 512, 10, 8, bias=False)
        self.norm0 = nn.GroupNorm(512, 512)
        self.conv1 = nn.Conv1d(512, 512, 3, 3, bias=False)
        self.conv2 = nn.Conv1d(512, 512, 3, 3, bias=False)
        self.conv3 = nn.Conv1d(512, 512, 3, 3, bias=False)
        self.conv4 = nn.Conv1d(512, 512, 2, 2, bias=False)
        self.conv5 = nn.Conv1d(512, 512, 2, 2, bias=False)
        self.conv6 = nn.Conv1d(512, 512, 2, 2, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = self.gelu(self.norm0(self.conv0(x)))
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))
        x = self.gelu(self.conv4(x))
        x = self.gelu(self.conv5(x))
        x = self.gelu(self.conv6(x))
        return x


class FeatureProjection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class PositionalConvEmbedding(nn.Module):
    def __init__(self, d_model, kernel_size, groups):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x: torch.Tensor):
        x = self.conv(x.transpose(1, 2))
        x = F.gelu(x[:, :, :-1])
        return x.transpose(1, 2)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, num_layers: int):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, activation="gelu",
                                                          batch_first=True))
        self.num_layers = num_layers

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer( output, src_mask=None, src_key_padding_mask=None)
        return output


if __name__ == "__main__":
    model = Hubert()
    import time

    x = torch.randn((4, 1, 48000))
    t1 = time.time()
    for i in range(100):
        y = model(x)
    t2 = time.time()
    print(t2 - t1)
    # print(y.shape)
