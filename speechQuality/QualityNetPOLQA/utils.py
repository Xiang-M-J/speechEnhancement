import numpy as np
import torch
import torchaudio
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


def accurate_num_cal(y_pred, y_true, step):
    """
    计算正确个数
    y_pred: N C
    y_true: N
    """
    if y_pred.shape[1] == 1:
        predict = floatTensorToClass(y_pred.squeeze(1), step)
    else:
        predict = torch.max(y_pred.data, 1)[1]  # torch.max()返回[values, indices]，torch.max()[1]返回indices
    target = floatTensorToClass(y_true, step)

    true_num = torch.eq(predict, target).cpu().sum().numpy()
    return true_num


def floatTensorToOnehot(x, step, s=False):
    """
    出于精度考虑，全部转为整数进行计算
    """

    x_i = (x * 100 - 100).int()
    x_i[x_i >= 400] = 399
    x_i[x_i <= 0] = 0
    x_i = x_i // int(step * 100)

    x_onehot = torch.nn.functional.one_hot(x_i.long(), int(4 / step)).squeeze(-2)
    if s:
        if len(x_onehot.shape) == 2:
            return smoothLabel(x_onehot)
        else:
            return smoothLabel3(x_onehot)
    else:
        return x_onehot


def smoothLabel(x):
    max_num = x.shape[1] + 2
    idx = torch.max(x, dim=-1)[1]
    b_idx = torch.arange(x.shape[0], dtype=torch.int64)
    smooth_dis = torch.zeros([x.shape[0], x.shape[1] + 4])
    smooth_dis.index_put_((b_idx, idx + 2), 0.8426 * torch.ones([x.shape[0]]))
    smooth_dis.index_put_((b_idx, idx + 3), 0.0763 * torch.ones([x.shape[0]]))
    smooth_dis.index_put_((b_idx, idx + 1), 0.0763 * torch.ones([x.shape[0]]))
    smooth_dis.index_put_((b_idx, idx + 4), 0.0024 * torch.ones([x.shape[0]]))
    smooth_dis.index_put_((b_idx, idx + 0), 0.0024 * torch.ones([x.shape[0]]))
    smooth_dis = smooth_dis[:, 2:max_num]
    return torch.div(smooth_dis, torch.sum(smooth_dis, dim=-1, keepdim=True)).to(x.device)


def smoothLabel3(x):
    smooth_dis = []
    for i in range(x.shape[0]):
        smooth_dis.append(smoothLabel(x[i, :, :]))

    return torch.stack(smooth_dis, dim=0)


def floatTensorToClass(x, step):
    x_i = (x * 100 - 100).int()
    x_i[x_i >= 400] = 399
    x_i[x_i <= 0] = 0
    x_i = x_i // int(step * 100)
    return x_i.long()


def oneHotToFloat(x: np.ndarray, step):
    """
    x : N C, C=4//step
    """
    if x.shape[1] == 1:
        value = x.squeeze(-1)
        value[value >= 5.0] = 5.0
        value[value <= 1.0] = 1.0
        return value
    value = np.arange(1.0, 5.0, step)
    return np.dot(x, value)


def oneHotToFloatTorch(x: torch.Tensor, step):
    """
    x : N C, C=4//step
    """
    value = torch.arange(1.0, 5.0, step).to(x.device).unsqueeze(1)
    return x @ value


def ListRead(path):
    with open(path, 'r') as f:
        file_list = f.read().splitlines()
    return file_list


if __name__ == '__main__':
    pass
    #
    # predict = torch.abs(torch.randn([64, 20])) * 2 + 1
    # target = torch.randn([64]) * 2 + 1
    # print(accurate_num_cal(predict, target, 0.2))

    # x = np.random.randn(4, 20)
    # print(oneHotToFloat(x, 0.2))
    # x = torch.randn([16, 128, 20])
    # y = smooth3(x)
    # x = torch.randint(1, 5, [4])
    # y = floatTensorToOnehot(x, 0.2)
    # print(y)

    # x = np.random.randn(4, 20)
    # y = oneHotToFloat(x, 0.2)
    # print(y)
