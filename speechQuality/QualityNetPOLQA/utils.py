import numpy as np
import torch
import torch.nn as nn
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


class FrameEDMLoss(nn.Module):
    """
    input: N L C, C = 4 // step
    target: N L 1
    """

    def __init__(self, smooth, enable, step=0.2) -> None:
        super(FrameEDMLoss, self).__init__()
        self.enable = enable
        self.step = step
        self.smooth = smooth
        if not enable:
            print("warning! FrameEDMLoss 损失被设置为永远返回0")

    def forward(self, input, target):
        if self.enable:
            if len(input.shape) == 2:
                input_ = floatTensorToOnehot(input, self.step, self.smooth)
            else:
                input_ = input
            target_ = floatTensorToOnehot(target, self.step, self.smooth)
            assert input_.shape[0] == target_.shape[0]
            cdf_input = torch.cumsum(input_, dim=-1)
            cdf_target = torch.cumsum(target_, dim=-1)
            cdf_diff = cdf_input - cdf_target
            samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_diff, 2), dim=-1)+1e-6)
            return samplewise_emd.mean()
        else:
            return 0


class AvgCrossEntropyLoss(nn.Module):
    """
    input: N C, C = 4 // step
    target: N
    """

    def __init__(self, enable=True, step=0.2) -> None:
        super(AvgCrossEntropyLoss, self).__init__()
        self.enable = enable
        self.loss = nn.CrossEntropyLoss()
        self.step = step
        if not enable:
            print("warning! FrameCrossEntropyLoss 损失被设置为永远返回0")

    def forward(self, input, target):
        if self.enable:
            # input_class = torch.argmax(input, dim=-1).float()
            target_class = floatTensorToClass(target, self.step).squeeze(-1)
            return self.loss(input, target_class)
        else:
            return 0


class FrameCrossEntropyLoss(nn.Module):
    def __init__(self, enable=True, step=0.2) -> None:
        super(FrameCrossEntropyLoss, self).__init__()
        self.enable = enable
        self.loss = nn.CrossEntropyLoss()
        self.step = step
        if not enable:
            print("warning! FrameCrossEntropyLoss 损失被设置为永远返回0")

    def forward(self, input, target):
        if self.enable:
            target_class = floatTensorToClass(target, self.step)
            return self.loss(torch.permute(input, [0, 2, 1]), target_class)
        else:
            return 0


class EDMLoss(nn.Module):
    """
    input: N
    forward(estimate, target, step)
    estimate: N C, C=4//step
    target: N 1
    """

    def __init__(self, step, smooth):
        super(EDMLoss, self).__init__()
        self.step = step
        self.smooth = smooth

    def forward(self, p_estimate: torch.Tensor, p_target: torch.Tensor):
        """
        p_target: [B, N]
        p_estimate: [B,]
        B 为批次大小，N 为类别数
        """
        if len(p_estimate.shape) == 1:
            p_estimate_ = floatTensorToOnehot(p_estimate, self.step, self.smooth)
        else:
            p_estimate_ = p_estimate
        p_target_ = floatTensorToOnehot(p_target, self.step, self.smooth)
        assert p_target_.shape == p_estimate_.shape
        cdf_target = torch.cumsum(p_target_, dim=1)
        cdf_estimate = torch.cumsum(p_estimate_, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2), dim=-1) + 1e-6)
        return samplewise_emd.mean()


class topKError(nn.Module):
    def __init__(self, topK, step):
        super(topKError, self).__init__()
        self.topK = topK
        self.step = step

    def forward(self, input, target):
        value, index = torch.topk(input, self.topK, dim=-1, out=None)
        pred = value @ (index * self.step + 1.0).t()
        pred = torch.diag(pred)
        return torch.mean(torch.pow(target - pred, 2))


class disError(nn.Module):
    def __init__(self, step):
        super(disError, self).__init__()
        self.step = step

    def forward(self, input, target):
        pred = oneHotToFloatTorch(input, self.step).squeeze(1)
        return torch.mean(torch.pow(target - pred, 2))


class shiftError(nn.Module):
    def __init__(self, step, topk):
        super(shiftError, self).__init__()
        self.step = step
        self.topk = topk

    def forward(self, input):
        value, index = torch.topk(input, self.topk, dim=-1, out=None)
        non_pLoss = torch.mean(torch.pow(1 - torch.sum(value, dim=-1), 2))
        # non_cIndex = index - index[:, 0].unsqueeze(1).repeat(1, self.topk)
        # non_cIndex = non_cIndex[:, 1:]
        # non_cLoss = torch.mean(torch.sum(torch.abs(non_cIndex), dim=-1))
        return non_pLoss


class shiftErrorWithTarget(nn.Module):
    def __init__(self, step, topk):
        super(shiftErrorWithTarget, self).__init__()
        self.step = step
        self.topk = topk
        self.left = (topk - 1) // 2

    def forward(self, input, target):
        # true_index = torch.argmax(target, dim=-1)
        true_index = (((target - 1.0) * 100).int() // int(self.step * 100)).long().unsqueeze(1)
        topk_p = torch.zeros([input.shape[0], 1]).to(input.device)
        input_extend = torch.cat([torch.zeros([input.shape[0], self.left]).to(input.device), input,
                                  torch.zeros([input.shape[0], self.left]).to(input.device)], dim=-1)

        for i in range(0, self.topk):
            topk_p = topk_p + torch.gather(input_extend, dim=-1, index=true_index + i)

        non_pLoss = torch.mean(torch.pow(1 - topk_p, 2))

        # pred_index = torch.argmax(input, dim=-1).unsqueeze(-1)
        # non_cLoss = torch.mean(torch.sum(torch.abs(pred_index - true_index), dim=-1))
        return non_pLoss


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
    idx = torch.max(x, dim=-1)[1]
    b_idx = torch.arange(x.shape[0], dtype=torch.int64)
    smooth_dis = torch.zeros([x.shape[0], x.shape[1] + 4])
    smooth_dis.index_put_((b_idx, idx + 2), 0.8426 * torch.ones([x.shape[0]]))
    smooth_dis.index_put_((b_idx, idx + 3), 0.0763 * torch.ones([x.shape[0]]))
    smooth_dis.index_put_((b_idx, idx + 1), 0.0763 * torch.ones([x.shape[0]]))
    smooth_dis.index_put_((b_idx, idx + 4), 0.0024 * torch.ones([x.shape[0]]))
    smooth_dis.index_put_((b_idx, idx + 0), 0.0024 * torch.ones([x.shape[0]]))
    smooth_dis = smooth_dis[:, 2:22]
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
    p_est = torch.randn([4, 20])
    p_tgt = torch.abs(torch.randn([4, ])) * 2 + 1.0
    p_tgt[p_tgt > 5.0] = 4.99
    # loss = FrameEDMLoss()
    # loss = topKError(topK=5, step=0.2)
    loss = shiftError(step=0.2, topk=5)
    print(loss(p_est))
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
