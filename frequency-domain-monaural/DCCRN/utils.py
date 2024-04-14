import numpy as np
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import glob
import soundfile as sf
import librosa
import torchaudio
from config import *
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import math


class VoiceBankDemand(Dataset):
    def __init__(self, scp, noisy_path, clean_path) -> None:
        super().__init__()
        with open(scp, 'r') as f:
            self.files = f.readlines()

        while not self.files[-1].startswith("p"):
            self.files.pop()
        for i in range(len(self.files)):
            self.files[i] = self.files[i][:-1]

        self.noisy_path = noisy_path
        self.clean_path = clean_path

    def __getitem__(self, index):
        noisy_wav = os.path.join(self.noisy_path, f"{self.files[index]}.wav")
        clean_wav = os.path.join(self.clean_path, f"{self.files[index]}.wav")

        feat_noisy, _ = getStftSpec_t(noisy_wav)
        feat_clean, _ = getStftSpec_t(clean_wav)

        return feat_noisy, feat_clean

    def __len__(self):
        return len(self.files)


class VoiceBankDemandBatch:
    def __init__(self, scp, noisy_path, clean_path, batch_size=16, files=None) -> None:
        if files is None:
            with open(scp, 'r') as f:
                self.files = f.readlines()

            while not self.files[-1].startswith("p"):
                self.files.pop()
            for i in range(len(self.files)):
                self.files[i] = self.files[i][:-1]

            np.random.shuffle(self.files)
        else:
            self.files = files

        self.idx = 0
        self.max_iter = math.ceil(len(self.files) / batch_size)
        self.noisy_path = noisy_path
        self.clean_path = clean_path
        self.batch_size = batch_size

    def train_valid_spilt(self, split_size: list):
        """
        分割验证集，注意使用返回的files创建新的类，需要指定 files，

        dataset = VoiceBankDemandBatch(" ", noisy_path, clean_path, batch_size=16, files=files)
        """
        train_num = math.ceil(split_size[0] * len(self.files))
        train_files = self.files[:train_num]
        valid_files = self.files[train_num:]
        return train_files, valid_files

    def batch(self):
        if self.idx < self.max_iter - 1:
            clean_audios = []
            noisy_audios = []

            for file in self.files[self.idx * self.batch_size: (self.idx + 1) * self.batch_size]:
                noisy_wav = os.path.join(self.noisy_path, f"{file}.wav")
                clean_wav = os.path.join(self.clean_path, f"{file}.wav")
                noisy_audios.append(preprocess(noisy_wav))
                clean_audios.append(preprocess(clean_wav))
            noisy_feat, _ = getStftSpec_tb(noisy_audios)
            clean_feat, _ = getStftSpec_tb(clean_audios)
            self.idx += 1
            return noisy_feat, clean_feat

        elif self.idx == self.max_iter - 1:
            clean_audios = []
            noisy_audios = []
            for file in self.files[self.idx * self.batch_size:]:
                noisy_wav = os.path.join(self.noisy_path, f"{file}.wav")
                clean_wav = os.path.join(self.clean_path, f"{file}.wav")
                noisy_audios.append(preprocess(noisy_wav))
                clean_audios.append(preprocess(clean_wav))

            self.idx = 0
            noisy_feat, _ = getStftSpec_tb(noisy_audios)
            clean_feat, _ = getStftSpec_tb(clean_audios)
            return noisy_feat, clean_feat
        else:
            raise f"{self.idx} is out of bound"

    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)


class VoiceBankDemandIter:
    """
    数据集迭代器（如果需要批量生成数据，推荐使用该方法），使用方法如下

    >>> train_files = []
    >>> train_loader = VoiceBankDemandIter(" ", train_noisy_path, train_clean_path, batch_size=batch_size, files=train_files, shuffle=True)
    >>> batch = next(train_loader)

    如果迭代完，重新创建迭代器

    >>> t_files = train_loader.files
    >>> del train_loader
    >>> train_loader = VoiceBankDemandIter(" ", train_noisy_path, train_clean_path, batch_size=batch_size, files=t_files, shuffle=True)
    """

    def __init__(self, scp, noisy_path, clean_path, batch_size=16, files=None, shuffle=False, seed=42) -> None:
        if files is None:
            with open(scp, 'r') as f:
                self.files = f.readlines()

            while not self.files[-1].startswith("p"):
                self.files.pop()
            for i in range(len(self.files)):
                self.files[i] = self.files[i][:-1]
            np.random.seed(seed)
            np.random.shuffle(self.files)
        else:
            self.files = files
            if shuffle is True:
                np.random.shuffle(self.files)

        self.idx = 0
        self.max_iter = math.ceil(len(self.files) / batch_size)

        self.noisy_path = noisy_path
        self.clean_path = clean_path
        self.batch_size = batch_size
        self.shuffle = shuffle

    def train_valid_spilt(self, split_size: list):
        """
        分割验证集，使用返回的files创建新的类，需要指定 files，

        dataset = VoiceBankDemandBatch(" ", noisy_path, clean_path, batch_size=16, files=files)
        """
        train_num = math.ceil(split_size[0] * len(self.files))
        train_files = self.files[:train_num]
        valid_files = self.files[train_num:]
        return train_files, valid_files

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.max_iter:
            raise StopIteration
        clean_audios = []
        noisy_audios = []
        if self.idx == self.max_iter - 1:
            for file in self.files[self.idx * self.batch_size:]:
                noisy_wav = os.path.join(self.noisy_path, f"{file}.wav")
                clean_wav = os.path.join(self.clean_path, f"{file}.wav")
                noisy_audios.append(preprocess(noisy_wav))
                clean_audios.append(preprocess(clean_wav))
        elif self.idx < self.max_iter - 1:
            for file in self.files[self.idx * self.batch_size: (self.idx + 1) * self.batch_size]:
                noisy_wav = os.path.join(self.noisy_path, f"{file}.wav")
                clean_wav = os.path.join(self.clean_path, f"{file}.wav")
                noisy_audios.append(preprocess(noisy_wav))
                clean_audios.append(preprocess(clean_wav))
        else:
            raise f"index {self.idx} 超出界限"
        noisy_feat, _ = getStftSpec_tb(noisy_audios)
        clean_feat, _ = getStftSpec_tb(clean_audios)
        self.idx += 1
        return noisy_feat, clean_feat

    def __len__(self):
        return self.max_iter

    def test_run_out(self):
        if self.idx == self.max_iter:
            return True
        return False


def collate_fn(batch):
    x_batch = rearrange(pad_sequence([b[0].T for b in batch], batch_first=True), "B L (N C) -> B N C L", N=2)
    y_batch = rearrange(pad_sequence([b[1].T for b in batch], batch_first=True), "B L (N C) -> B N C L", N=2)
    return x_batch, y_batch


def getStftSpec(wav_path):
    """
    使用 librosa 处理数据（比较慢）
    Args:
        wav_path:

    Returns:

    """
    feat_wav, _ = sf.read(wav_path)
    c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
    feat_wav = feat_wav * c
    feat_x = librosa.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, win_length=win_size, window='hann').T
    feat_x, phase_x = np.abs(feat_x), np.angle(feat_x)
    feat_x = np.sqrt(feat_x)  # 压缩幅度
    return feat_x, phase_x


def getStftSpec_t(wav_path):
    """
    使用 torch 处理数据
    Args:
        wav_path:

    Returns:

    """
    feat_wav = preprocess(wav_path)

    feat_x = torch.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
                        window=torch.hann_window(win_size), return_complex=True)
    feat_x, phase_x = torch.sqrt(torch.abs(feat_x)), torch.angle(feat_x)

    # 先不堆叠，先连接起来，在 collate_fn 中 rearrange 即可

    feat_x = torch.cat([feat_x * torch.cos(phase_x), feat_x * torch.sin(phase_x)], dim=0)

    return feat_x, phase_x


def preprocess(wav_path):
    """
    预处理音频，约束波形幅度
    Args:
        wav_path:

    Returns:

    """
    wav = torchaudio.load(wav_path)[0].squeeze(0)
    c = torch.sqrt(wav.shape[0] / torch.sum((wav ** 2.0)))
    feat_wav = wav * c
    wav_len = len(feat_wav)

    frame_num = int(np.ceil((wav_len - 512 + 512) / 128 + 1))
    fake_wav_len = (frame_num - 1) * 128 + 512 - 512
    left_sample = fake_wav_len - wav_len
    feat_wav = torch.cat([feat_wav, torch.zeros([left_sample])], dim=0)
    return feat_wav


def getStftSpec_tb(batch: list):
    """
    使用 torch.stft 并行计算 stft
    Args:
        batch: 储存音频的列表

    Returns:

    """
    batch = pad_sequence(batch, batch_first=True)
    feat_x = torch.stft(batch, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
                        window=torch.hann_window(win_size), return_complex=True)
    # feat_x = feat_x.permute([0, 2, 1])
    feat_x, phase_x = torch.abs(feat_x), torch.angle(feat_x)
    feat_x = torch.sqrt(feat_x)  # 压缩幅度
    feat_x = torch.stack([feat_x * torch.cos(phase_x), feat_x * torch.sin(phase_x)], dim=1)

    return feat_x, phase_x


if __name__ == "__main__":
    base_path = r"D:\work\speechEnhancement\datasets\voicebank_demand"
    train_clean_path = os.path.join(base_path, "clean_trainset_28spk_wav")
    train_noisy_path = os.path.join(base_path, "noisy_trainset_28spk_wav")
    train_scp_path = os.path.join(base_path, "train.scp")
    # dataset = VoiceBankDemand(train_scp_path, train_noisy_path, train_clean_path)
    # train_loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)  # 使用torch 加速处理 53 s
    # for batch in tqdm(train_loader):
    #     pass
    # print(i)

    # 时间更短
    # train_loader = VoiceBankDemandBatch(train_scp_path, train_noisy_path, train_noisy_path, 16)  # 38s
    # for e in range(3):
    #     for i in tqdm(range(len(train_loader))):
    #         batch = train_loader.batch()
        # print(i)
