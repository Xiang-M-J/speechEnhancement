# modelInfo

## LSTM

### 2024/05/09 15:25

**模型代码**
```Python
import torch.nn as nn
class QualityNet(nn.Module):
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
```

**参数设置**

仅显示部分参数
```yaml
batch_size: 64
delta_loss: 0.0001
dropout: 0.1
epochs: 50
gamma: 0.3
load_weight: false
lr: 0.001
model_name: QN20240508_174129
model_type: ''
optimizer_type: 3
patience: 6
shuffle: true
step_size: 10
weight_decay: 0
```



### 2024/05/09 23:02:05

与上一个相比，将原本 Frame_loss 中的

```python
torch.mean((10 ** (true_pesq - 4.5)) * torch.mean((input - target) ** 2, dim=1))
```

改为

```python
torch.mean((10 ** (true_pesq - 5)) * torch.mean((input - target) ** 2, dim=1))
```





## LSTMA



```python
class TimeRestrictedAttention(nn.Module):
    def __init__(self, d_model, dq, heads=4, offset=64):
        super(TimeRestrictedAttention, self).__init__()
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
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        attn = torch.matmul(q, k.transpose(-1, -2)) / 256
        attn = torch.softmax(attn, dim=-1)
        attn = attn * mask
        context = torch.bmm(attn, v)
        return context, attn
   
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
        self.mask = None

    def forward(self, x):
        if self.mask is None:
            self.mask = get_attn_mask(x.shape[1], offset=64).to(x.device)
        lstm_out, _ = self.lstm(x)
        lstm_out, attn = self.attn(lstm_out, self.mask)
        l1 = self.dropout(self.elu(self.linear1(lstm_out)))
        Frame_score = self.linear2(l1).squeeze(-1)
        Average_score = self.pool(Frame_score)
        return Frame_score, Average_score
```

### lstmA20240515_003916

使用一半数据进行训练，训练耗时：398.81


| test loss | mse    | lcc    | srcc   |
| --------- | ------ | ------ | ------ |
| 0.1378    | 0.1300 | 0.9595 | 0.9287 |

epochs: 35 dropout: 0.3 random_seed: 34 model_type: lstmA save: True save_model_epoch: 5 scheduler_type: 1 load_weight: False enableFrame: True smooth: True score_step: 0.2 cnn_filter: 128 cnn_feature: 64 shuffle: True batch_size: 128 spilt_rate: [0.8, 0.1, 0.1] fft_size: 512 hop_size: 256 lr: 0.001 weight_decay: 0 optimizer_type: 3 beta1: 0.99 beta2: 0.999 step_size: 10 gamma: 0.3 patience: 5 delta_loss: 0.001



## CNN







## CNNA

### cnnA20240515_004030

```python
class SEBlock(nn.Module):
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
        x = self.avg_linear(x)
        return x
```

使用一半数据进行训练，训练耗时：397.53

| test loss | mse    | lcc    | srcc   |
| --------- | ------ | ------ | ------ |
| 0.1363    | 0.1368 | 0.9570 | 0.9333 |
