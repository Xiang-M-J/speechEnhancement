import numpy as np
import pystoi
import torch
import torchaudio
from torch.utils.data import Dataset
import pesq
from torchaudio.transforms import Resample

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


class DNSDataset(Dataset):
    def __init__(self, files, fft_num, win_shift, win_size, input_type=2) -> None:
        super().__init__()
        self.noise_path = []
        self.clean_path = []
        for file in files:
            noise, clean = file.split(",")
            self.noise_path.append(noise)
            self.clean_path.append(clean)
        self.fft_num = fft_num
        self.win_shift = win_shift
        self.win_size = win_size
        self.input_type = input_type

    def __getitem__(self, index):
        noisy_wav = self.noise_path[index]
        clean_wav = self.clean_path[index]

        noise, _ = preprocess(noisy_wav)
        clean, _ = preprocess(clean_wav)

        pn_feat, pn_phase = getStftSpec(noise, self.fft_num, self.win_shift, self.win_size, self.input_type)
        pc_feat, pc_phase = getStftSpec(clean, self.fft_num, self.win_shift, self.win_size, self.input_type)

        return pn_feat, pn_phase, pc_feat, clean

    def __len__(self):
        return len(self.noise_path)


class CalQuality:
    def __init__(self, fs=48000, batch=True):
        super().__init__()
        self.resample = Resample(fs, 16000)
        self.batch = batch
        self.fs = fs

    def cal_pesq(self, deg, ref):
        if deg.shape != ref.shape:
            raise ValueError("deg audio and ref audio must be same shape")
        deg_16k = self.resample(deg)
        ref_16k = self.resample(ref)

        if self.batch:
            p = pesq.pesq_batch(16000, ref_16k.numpy(), deg_16k.numpy(), mode="wb", n_processor=4)
            return p
        else:
            p = pesq.pesq(16000, ref_16k.numpy(), deg_16k.numpy())
            return p

    def cal_stoi(self, deg, ref):
        if deg.shape != ref.shape:
            raise ValueError("deg audio and ref audio must be same shape")
        if self.batch:
            s = []
            for i in range(deg.shape[0]):
                s_ = pystoi.stoi(ref[i, :], deg[i, :], self.fs)
                s.append(s_)
            return s
        else:
            s = pystoi.stoi(ref, deg, self.fs)
            return s

    def cal_polqa(self, deg, ref):
        raise NotImplementedError

    def __call__(self, deg: torch.Tensor, ref: torch.Tensor):
        """
        deg: (N, L) or (L,)
        ref: (N, L) or (L,)
        return: pesq, stoi
        """
        p = self.cal_pesq(deg, ref)
        s = self.cal_stoi(deg.numpy(), ref.numpy())
        return p, s


def getStftSpec(feat_wav, fft_num, win_shift, win_size, input_type=2):
    """
    处理数据
    """
    feat_x = torch.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
                        window=torch.hann_window(win_size), return_complex=True).T
    feat_x, phase_x = torch.abs(feat_x), torch.angle(feat_x)
    feat_x = torch.sqrt(feat_x)  # 压缩幅度
    if input_type == 2:
        # 用于 dpcrn
        feat_x = torch.stack((feat_x * torch.cos(phase_x), feat_x * torch.sin(phase_x)), dim=0)
    return feat_x, phase_x


def preprocess(wav_path):
    """
    预处理音频，约束波形幅度
    """
    wav, fs = torchaudio.load(wav_path)
    wav = wav / torch.max(torch.abs(wav))
    return wav.squeeze(0), fs


def spec2wav(feat, phase, fft_size, hop_size, win_size, input_type=2):
    """
    feat:
    input type == 1:   [N L C]
    input type == 2:   [N 2 L C], phase = None
    """
    if input_type == 1:
        feat = torch.pow(feat, 2)
        comp = torch.complex(feat * torch.cos(phase), feat * torch.sin(phase))
    elif input_type == 2:
        if len(feat.shape) != 4:
            raise ValueError("feat's dimension is not 4")
        feat_mag = torch.norm(feat, dim=1)
        feat_phase = torch.atan2(feat[:, 1, :, :], feat[:, 0, :, :])
        feat_mag = torch.pow(feat_mag, 2)

        comp = torch.complex(feat_mag * torch.cos(feat_phase), feat_mag * torch.sin(feat_phase)).permute([0, 2, 1])
    else:
        raise ValueError("type error")
    wav = torch.istft(comp, n_fft=fft_size, hop_length=hop_size, win_length=win_size,
                      window=torch.hann_window(win_size), return_complex=False)
    return wav


def predict_pesq_batch(deg_feat, deg_phase, ref_feat, ref_phase, fft_size, hop_size, win_size):
    """
    feat or phase: B C L
    """
    deg_wav = spec2wav(deg_feat, deg_phase, fft_size, hop_size, win_size)
    ref_wav = spec2wav(ref_feat, ref_phase, fft_size, hop_size, win_size)
    mos = pesq.pesq_batch(48000, ref_wav, deg_wav, mode="wb")
    return mos


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
