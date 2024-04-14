stft 的计算为 torch.stft()，默认情况下 window 为矩形窗，可以传递汉宁窗

```python
y = torch.stft(x, n_fft=256, hop_length=128, win_length=256,  window=torch.hann_window(window_length=256), return_complex=True)

```



## 能量分析

stft 后的能量与原始信号的能量存在一些关系

> 下面的分析仅针对符合均值为 0 的正态分布的变量

以矩形窗，n_fft=256，hop_length=128 为例

```python
x = torch.rand([102400,])
y = torch.stft(x, n_fft=256, hop_length=128, win_length=256, return_complex=True)
```

可以计算 y 的能量大约是 x 的能量的 7/4 * win_length 倍

```python
p1 = torch.sum(torch.pow(x, 2))
p2 = torch.sum(torch.pow(torch.abs(y), 2))
print(p2/p1)   # 约为 448
```

如果 hop_length 为 win_length 的 1/4 倍，此时

```python
p1 = torch.sum(torch.pow(x, 2))
p2 = torch.sum(torch.pow(torch.abs(y), 2))
print(p2/p1)   # 约为 7/2 * win_length 倍
```

换成 hanning 窗，再进行实验

```python
y = torch.stft(x, n_fft=256, hop_length=128, win_length=256,  window=torch.hann_window(window_length=256), return_complex=True)
```

当 hop_length 为 128 时

```python
p1 = torch.sum(torch.pow(x, 2))
p2 = torch.sum(torch.pow(torch.abs(y), 2))
print(p2/p1)   # 9/16 * win_length
```

当 hop_length 为 64 时

```python
p1 = torch.sum(torch.pow(x, 2))
p2 = torch.sum(torch.pow(torch.abs(y), 2))
print(p2/p1)   # 9/8 * win_length
```

