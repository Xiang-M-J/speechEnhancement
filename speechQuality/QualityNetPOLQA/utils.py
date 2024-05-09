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
    def __init__(self) -> None:
        super(FrameMse, self).__init__()

    def forward(self, input, target):
        true_pesq = target[:, 0]
        return torch.mean((10 ** (true_pesq - 4.5)) * torch.mean((input - target) ** 2, dim=1))


class EDMLoss(nn.Module):
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target: torch.Tensor, p_estimate: torch.Tensor):
        """
        p_target: [B, N]
        p_estimate: [B, N]
        B 为批次大小，N 为类别数
        """
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()


def ListRead(path):
    with open(path, 'r') as f:
        file_list = f.read().splitlines()
    return file_list
