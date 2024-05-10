import os
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Sp_and_phase(path, fft_size=512, hop_size=256, Noisy=False, compress=False):
    signal, rate = torchaudio.load(path)
    signal = signal / torch.max(torch.abs(signal))

    F = torch.stft(signal, n_fft=fft_size, hop_length=hop_size, win_length=fft_size,
                   window=torch.hamming_window(fft_size), return_complex=True)
    F = F.squeeze(0).t()
    Lp = torch.abs(F)
    if compress:
        Lp = torch.sqrt(Lp)
    phase = torch.angle(F)
    if Noisy:
        meanR = torch.mean(Lp, dim=1).reshape((257, 1))
        stdR = torch.std(Lp, dim=1).reshape((257, 1)) + 1e-12
        NLp = (Lp - meanR) / stdR
    else:
        NLp = Lp

    return NLp, phase


class DNSPOLQADataset(Dataset):
    def __init__(self, wav_files: list[str], fft_size: int = 512, hop_size: int = 256, ):
        super(DNSPOLQADataset, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.wav = []
        self.polqa = []
        for wav_file in wav_files:
            file, score = wav_file.split(",")
            self.wav.append(file)
            self.polqa.append(score)

    def __getitem__(self, index):
        wav = self.wav[index]
        polqa = torch.tensor(float(self.polqa[index]), dtype=torch.float32, device=device)
        noisy_LP, _ = Sp_and_phase(wav)
        noisy_LP = noisy_LP.to(device=device)
        return noisy_LP, [polqa, polqa * torch.ones([noisy_LP.shape[0]], dtype=torch.float32, device=device)]

    def __len__(self):
        return len(self.wav)


class FrameMse(nn.Module):
    def __init__(self, enable=True) -> None:
        super(FrameMse, self).__init__()
        self.enable = enable
        if not enable:
            print("warning! FrameMse 损失被设置为永远返回0")

    def forward(self, input, target):
        true_pesq = target[:, 0]

        if self.enable:
            return torch.mean((10 ** (true_pesq - 5)) * torch.mean((input - target) ** 2, dim=1))
        else:
            return 0


class FrameEDMLoss(nn.Module):
    """
    input: N L C, C = 4 // step
    target: N L 1
    """

    def __init__(self, enable=True) -> None:
        super(FrameEDMLoss, self).__init__()
        self.enable = enable
        if not enable:
            print("warning! FrameEDMLoss 损失被设置为永远返回0")

    def forward(self, input, target, step):
        if self.enable:
            target = floatTensorToOnehot(target, step)
            assert input.shape[0] == target.shape[0]
            cdf_target = torch.cumsum(target, dim=-1)
            cdf_input = torch.cumsum(input, dim=-1)
            cdf_diff = cdf_input - cdf_target
            samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_diff, 2)))
            return samplewise_emd
        else:
            return 0


class AvgCrossEntropyLoss(nn.Module):
    def __init__(self, enable=True, step=0.2) -> None:
        super(AvgCrossEntropyLoss, self).__init__()
        self.enable = enable
        self.loss = nn.CrossEntropyLoss()
        self.step = step
        if not enable:
            print("warning! FrameCrossEntropyLoss 损失被设置为永远返回0")

    def forward(self, input, target):
        if self.enable:
            input_class = torch.argmax(input, dim=-1).float()
            target_class = floatTensorToClass(target, self.step)
            return self.loss(input_class, target_class)
        else:
            return 0


class FrameCrossEntropyLoss(nn.Module):
    def __init__(self, enable=True) -> None:
        super(FrameCrossEntropyLoss, self).__init__()
        self.enable = enable
        self.loss = nn.CrossEntropyLoss()
        if not enable:
            print("warning! FrameCrossEntropyLoss 损失被设置为永远返回0")

    def forward(self, input, target):
        if self.enable:
            input_class = torch.argmax(input, dim=-1)
            target_class = torch.argmax(target, dim=-1)
            return self.loss(input_class, target_class)
            pass
        else:
            return 0


class EDMLoss(nn.Module):
    """
    input: N
    forward(estimate, target, step)
    estimate: N C, C=4//step
    target: N 1
    """

    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_estimate: torch.Tensor, p_target: torch.Tensor, step):
        """
        p_target: [B, N]
        p_estimate: [B, N]
        B 为批次大小，N 为类别数
        """
        p_target = floatTensorToOnehot(p_target, step)
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()


def floatTensorToOnehot(x, step):
    """
    出于精度考虑，全部转为整数进行计算
    """
    x = (x * 100 - 100).int()
    x[x == 400] = 399
    x = x // int(step * 100)
    x = torch.nn.functional.one_hot(x.long(), int(4 / step)).squeeze(-2)
    return x


def floatTensorToClass(x, step):
    x = (x * 100 - 100).int()
    x[x == 400] = 399
    x = x // int(step * 100)
    return x.float()


def ListRead(path):
    with open(path, 'r') as f:
        file_list = f.read().splitlines()
    return file_list


if __name__ == '__main__':
    p_est = torch.randn([4, 20])
    p_tgt = torch.abs(torch.randn([4, 1])) * 2 + 1.0
    p_tgt[p_tgt > 5.0] = 4.99
    loss = AvgCrossEntropyLoss()
    print(loss(p_est, p_tgt))
