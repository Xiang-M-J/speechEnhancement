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




