语音信号高度非平稳，二阶统计量（功率谱）随着时间改变，在短时间内（10-30 ms），谱特征是平稳的，并且呈现准周期的性质



语音质量本质上是高度主观的，很难可靠地评估。可理解性评估说话者说了什么，即讲述单词的内容，并不主观。两者并没有很大的关联。



在多条不等长语音进行训练时，需要将其补零至等长的情况，下面分析先补零后做stft，先做stft后补零两种情况

假设以先补零再做stft为标准，hop_length 为 win_length 的一半，那么这两种情况只会影响最后一帧，如果两种情况需要完全等同，需要进行下面的操作

先将所有的音频补零至为 win_length 的整数倍，对于多个音频的处理，可以采用下面的思路，先找出最长的stft结果，对于剩余的所有音频，采取下面的做法

1. 先丢弃最后一帧的数据
2. 计算加零时最后一帧的数据
3. 补零至最长

```python
import torch
data = torch.randn([1, 25088])

x1 = torch.concat([data, torch.zeros([1, 512])], dim=1)   # 补零
x2 = data													

y1 = torch.stft(x1, 512, hop_length=256, return_complex=True)
y2 = torch.stft(x2, 512, hop_length=256, return_complex=True)
y_ = torch.stft(torch.concat([x2[:, 24832:], torch.zeros([1, 256])], dim=1), 512, hop_length=256, return_complex=True, center=False)        # 注意 center=False时，stft不会默认补零
y2 = torch.concat([y2[:,:, :-1], y_, torch.zeros([1, 257, y1.shape[2]-y2.shape[2]])], dim=2)
```

或者在计算 y1 和 y2 时便设置 center = False

```python
x1 = torch.concat([data, torch.zeros([1, 512])], dim=1)
x2 = data

y1 = torch.stft(x1, 512, hop_length=256, return_complex=True, center=False)
y2 = torch.stft(x2, 512, hop_length=256, return_complex=True, center=False)
y_ = torch.stft(torch.concat([x2[:, 24832:], torch.zeros([1, 256])], dim=1), 512, hop_length=256, return_complex=True, center=False)
y2 = torch.concat([y2, y_, torch.zeros([1, 257, y1.shape[2]-y2.shape[2]-1])], dim=2)
```



