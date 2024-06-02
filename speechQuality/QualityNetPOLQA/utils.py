import random
import logging
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from scipy.special import softmax

from sigmos.sigmos import SigMOS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None


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


def path2Spec(path, fft_size=512, hop_size=256, input_type=1):
    wav, fs = preprocess(path)
    feat_x = torch.stft(wav, n_fft=fft_size, hop_length=hop_size, win_length=fft_size,
                        window=torch.hann_window(fft_size), return_complex=True).T
    feat_x, phase_x = torch.abs(feat_x), torch.angle(feat_x)
    if input_type == 1:
        feat_x = torch.sqrt(feat_x)  # 压缩幅度
    elif input_type == 2:
        # 用于 dpcrn
        feat_x = torch.sqrt(feat_x)
        feat_x = torch.stack((feat_x * torch.cos(phase_x), feat_x * torch.sin(phase_x)), dim=0)
    return feat_x, phase_x


class DNSPOLQADataset(Dataset):
    def __init__(self, wav_files: list[str], fft_size: int = 512, hop_size: int = 256, return_wav=False, input_type=1):
        super(DNSPOLQADataset, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.return_wav = return_wav
        self.input_type = input_type

        self.wav = []
        self.polqa = []
        for wav_file in wav_files:
            file, score = wav_file.split(",")
            self.wav.append(file)
            self.polqa.append(score)

    def __getitem__(self, index):
        wav = self.wav[index]
        polqa = torch.tensor(float(self.polqa[index]), dtype=torch.float32, device=device)
        if self.return_wav:
            noisy_LP, fs = torchaudio.load(wav)
            noisy_LP = noisy_LP / torch.max(torch.abs(noisy_LP))
        else:
            noisy_LP, _ = path2Spec(wav, self.fft_size, self.hop_size, self.input_type)
        noisy_LP = noisy_LP.to(device)
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


class CalSigmos:
    def __init__(self, fs=48000, batch=True, model_dir="sigmos"):
        super().__init__()
        self.estimator = SigMOS(model_dir=model_dir)
        self.batch = batch
        self.fs = fs

    def __call__(self, wav):
        if self.batch:
            results = [[] for _ in range(7)]
            batch = wav.shape[0]
            for i in range(batch):
                wav_ = wav[i, :]
                result = self.estimator.run(wav_, self.fs)
                for j in range(7):
                    results[j].append(result[j])
            return results
        else:
            if len(wav.shape) == 2:
                wav = wav.squeeze(0)
            return self.estimator.run(wav, sr=self.fs)


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


def spec2wav(feat, phase, mask, fft_size, hop_size, win_size, input_type=2, mask_target=None):
    """
    Args:

        feat: 原始的带噪声的频谱, input_type == 1: [N L C],  input_type == 2:  [N 2 L C]
        phase: 相位谱，当input_type==2时，仅需传入None即可
        mask: 模型预测的值，**当 mask_target 为 None，mask 就是增强后的频谱**
    """
    if input_type == 1:
        mag = back_magnitude_spectrum(feat, mask, mask_target, input_type)
        comp = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase)).permute([0, 2, 1])
    elif input_type == 2:
        if len(feat.shape) != 4:
            raise ValueError("feat's dimension is not 4")
        mag = back_magnitude_spectrum(feat, mask, mask_target, input_type)
        pha = torch.atan2(feat[:, 1, :, :], feat[:, 0, :, :])
        comp = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha)).permute([0, 2, 1])
    else:
        raise ValueError("type error")
    wav = torch.istft(comp, n_fft=fft_size, hop_length=hop_size, win_length=win_size,
                      window=torch.hann_window(win_size), return_complex=False)
    return wav


def back_magnitude_spectrum(x, mask, mask_target, input_type):
    if input_type == 1:
        if mask_target is None:
            return torch.pow(mask, 2)
        elif mask_target == "IAM":
            return torch.pow(x, 2) * mask
        elif mask_target == "IRM":
            return torch.pow(x, 2) * mask
        else:
            raise NotImplementedError

    elif input_type == 2:
        if mask_target is not None:
            raise NotImplementedError
        mag = torch.pow(torch.norm(mask, dim=1), 2)
        return mag


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

    x_onehot = torch.nn.functional.one_hot(x_i.long(), int(400 / int(step*100)))

    
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


def floatNumpyToClass(x, step):
    x_i = (x * 100 - 100).astype(np.int32)
    x_i[x_i >= 400] = 399
    x_i[x_i <= 0] = 0
    x_i = x_i // int(step * 100)
    return x_i.astype(np.int64)


def oneHotToFloat(x: np.ndarray, step):
    """
    x : N C, C=4//step
    """
    x_ = softmax(x, -1)
    value = np.arange(1.0, 4.99, step) + (step / 2)
    return np.dot(x_, value)


def oneHotToFloatTorch(x: torch.Tensor, step):
    """
    x : N C, C=4//step
    """
    value = torch.arange(1.0, 5.0, step).to(x.device).unsqueeze(1)
    return torch.softmax(x, dim=-1) @ value


def ListRead(path):
    with open(path, 'r') as f:
        file_list = f.read().splitlines()
    return file_list


def normalize(y):
    return (y - 1.0) / 4.0


def denormalize(y):
    return y * 4.0 + 1.0


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_graph(model, dummy_input, save_path):
    writer = SummaryWriter(save_path)
    try:
        writer.add_graph(model, dummy_input)
        writer.close()
    except Exception as e:
        print(e)
        writer.close()


def get_logging(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    formatter1 = logging.Formatter('%(asctime)s %(filename)s %(message)s')
    handler.setFormatter(formatter1)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter2 = logging.Formatter('%(asctime)s %(message)s')
    console.setFormatter(formatter2)

    logger.addHandler(handler)
    # logger.addHandler(console)

    return logger


def apply_SE_Loss(x, y, y_pred, loss_fn, mask_target, input_type):
    """
    根据 mask_target 和 input_type 计算损失
    """
    if mask_target is None:
        return loss_fn(y_pred, y)
    elif mask_target == "IAM":
        if input_type == 1:
            return loss_fn(y_pred * torch.pow(x, 2), torch.pow(y, 2))
        elif input_type == 2:
            raise NotImplementedError
    elif mask_target == "IRM":
        if input_type == 1:
            x_mag = get_mag(x, 1)
            y_mag = get_mag(y, 1)
            n_mag = x_mag - y_mag
            return loss_fn(y_pred * torch.sqrt(torch.add(torch.pow(y_mag, 2), torch.pow(n_mag, 2)) + 1e-6), y_mag)
        elif input_type == 2:
            raise NotImplementedError
    else:
        raise NotImplementedError


def cal_QN_input(x, y, y_pred, mask_target, input_type):
    """
    根据 mask_target 计算幅度谱，用于QualityNet的输入
    返回：增强后的幅度谱，干净的幅度谱
    """
    if mask_target is None:
        if input_type == 1:
            return torch.pow(y_pred, 2), torch.pow(y, 2)
        else:
            return torch.pow(torch.norm(y_pred, dim=1), 2), torch.pow(torch.norm(y, dim=1), 2)
    elif mask_target == "IAM":
        if input_type == 1:
            return y_pred * torch.pow(x, 2), torch.pow(y, 2)
        elif input_type == 2:
            raise NotImplementedError
    elif mask_target == "IRM":
        if input_type == 1:
            x_mag = get_mag(x, 1)
            y_mag = get_mag(y, 1)
            n_mag = x_mag - y_mag
            return y_pred * torch.sqrt(torch.add(torch.pow(y_mag, 2), torch.pow(n_mag, 2))), y_mag
        elif input_type == 2:
            raise NotImplementedError
    else:
        raise NotImplementedError


def cal_QN_input_compress(x, y, y_pred, mask_target, input_type):
    """
    根据 mask_target 计算幅度谱，用于QualityNet的输入
    返回：增强后的幅度谱，干净的幅度谱
    """
    if mask_target is None:
        if input_type == 1:
            return y_pred, y
        else:
            return torch.norm(y_pred, dim=1), torch.norm(y, dim=1)
    elif mask_target == "IAM":
        if input_type == 1:
            return torch.sqrt(y_pred + 1e-6) * x, y
        elif input_type == 2:
            raise NotImplementedError
    elif mask_target == "IRM":
        if input_type == 1:
            x_mag = get_mag(x, 1)
            y_mag = get_mag(y, 1)
            n_mag = x_mag - y_mag
            return torch.sqrt(y_pred * torch.sqrt(torch.add(torch.pow(y_mag, 2), torch.pow(n_mag, 2)) + 1e-6) + 1e-6), y
        elif input_type == 2:
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_mag(x, input_type):
    if input_type == 1:
        return torch.pow(x, 2)
    elif input_type == 2:
        return torch.pow(x[:, 0, :, :], 2) + torch.pow(x[:, 1, :, :], 2)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    pass
    #
    # predict = torch.abs(torch.randn([64, 20])) * 2 + 1
    # tgt = torch.randn([64]) * 2 + 1
    # print(accurate_num_cal(predict, tgt, 0.2))

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
