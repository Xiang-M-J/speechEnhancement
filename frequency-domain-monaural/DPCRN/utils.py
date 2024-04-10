import numpy as np
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
from einops import rearrange


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
        
        feat_noisy = getstftSpec_torch(noisy_wav)
        feat_clean = getstftSpec_torch(clean_wav)
        
        # return torch.tensor(feat_noisy, dtype=torch.float32), torch.tensor(feat_clean, dtype=torch.float32)
        return feat_noisy, feat_clean
    
    def __len__(self):
        return len(self.files)

class VoiceBankDemandBatch:
    def __init__(self, scp, noisy_path, clean_path, batch_size=16, files=None) -> None:
        if files == None:
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
            
            for file in self.files[self.idx * self.batch_size: (self.idx+1)*self.batch_size]:
                noisy_wav = os.path.join(self.noisy_path, f"{file}.wav")
                clean_wav = os.path.join(self.clean_path, f"{file}.wav")
                noisy_audios.append(preprocess(noisy_wav))
                clean_audios.append(preprocess(clean_wav))  
            noisy_feat, _ = getstftSpec_torch_batch(noisy_audios)
            clean_feat, _ = getstftSpec_torch_batch(clean_audios)
            self.idx += 1
            return noisy_feat, clean_feat
        
        elif self.idx == self.max_iter - 1:
            clean_audios = []
            noisy_audios = []
            for file in self.files[self.idx * self.batch_size: ]:
                noisy_wav = os.path.join(self.noisy_path, f"{file}.wav")
                clean_wav = os.path.join(self.clean_path, f"{file}.wav")
                noisy_audios.append(preprocess(noisy_wav))
                clean_audios.append(preprocess(clean_wav))

            self.idx = 0
            noisy_feat, _ = getstftSpec_torch_batch(noisy_audios)
            clean_feat, _ = getstftSpec_torch_batch(clean_audios)
            return noisy_feat, clean_feat
        else:
            raise f"{self.idx} is out of bound"
        
    def __len__(self):
        return  math.ceil(len(self.files) / self.batch_size)


def collate_fn(batch):
    x_batch = rearrange(pad_sequence([b[0] for b in batch], batch_first=True), "B L (N C) -> B N L C", N=2)
    y_batch = rearrange(pad_sequence([b[1] for b in batch], batch_first=True), "B L (N C) -> B N L C", N=2)
    return x_batch, y_batch


def getstftSpec(wav_path):
    feat_wav, _ = sf.read(wav_path)
    c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
    feat_wav = feat_wav * c
    feat_x = librosa.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, win_length=win_size, window='hann').T
    feat_x, phase_x = np.abs(feat_x), np.angle(feat_x)    
    feat_x = np.sqrt(feat_x)        # 压缩幅度

    return feat_x, phase_x

def getstftSpec_torch(wav_path):
    feat_wav, _ = torchaudio.load(wav_path)
    feat_wav = feat_wav.squeeze(0)
    c = torch.sqrt(feat_wav.shape[0] / torch.sum((feat_wav ** 2.0)))
    feat_wav = feat_wav * c
    feat_x = torch.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, win_length=win_size, window=torch.hann_window(win_size), return_complex=True).T
    feat_x, phase_x = torch.sqrt(torch.abs(feat_x)), torch.angle(feat_x)

    # 先不堆叠，先连接起来，在 collate_fn 中 rearange 即可
    # feat_x = torch.stack((feat_x * torch.cos(phase_x), feat_x * torch.sin(phase_x)), dim=0)
    feat_x = torch.cat([feat_x * torch.cos(phase_x), feat_x * torch.sin(phase_x)], dim=1)
    return feat_x

def wav2spec(feat_wav):
    c = torch.sqrt(feat_wav.shape[0] / torch.sum((feat_wav ** 2.0)))
    feat_wav = feat_wav * c

    feat_x = torch.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, win_length=win_size, window=torch.hann_window(win_size), return_complex=True).T
    feat_x, phase_x = torch.sqrt(torch.abs(feat_x)), torch.angle(feat_x)
    feat_x = torch.stack((feat_x * torch.cos(phase_x), feat_x * torch.sin(phase_x)), dim=0)

    return feat_x, c, len(feat_wav)

def spec2wav(feat, c, l):
    feat_mag = torch.norm(feat, dim=1)
    feat_phase = torch.atan2(feat[:, 1, :, :], feat[:, 0, :, :])
    feat_mag = torch.pow(feat_mag, 2)
    feat_mag = feat_mag.squeeze(dim=0)
    feat_phase = feat_phase.squeeze(dim=0)
    feat_de = torch.multiply(feat_mag, torch.exp(1j * feat_phase))
    est_audio = torch.istft((feat_de).T, n_fft=fft_num, hop_length=win_shift,
                                     win_length=win_size,  window=torch.hann_window(win_size), length=l)
    return est_audio / c


def preprocess(wav_path):
    wav = torchaudio.load(wav_path)[0].squeeze(0)
    c = torch.sqrt(wav.shape[0] / torch.sum((wav ** 2.0)))
    feat_wav = wav * c
    return feat_wav


def getstftSpec_torch_batch(batch):
    batch = pad_sequence(batch, batch_first=True)
    feat_x = torch.stft(batch, n_fft=fft_num, hop_length=win_shift, win_length=win_size, window=torch.hann_window(win_size), return_complex=True)
    feat_x = feat_x.permute([0, 2, 1])

    feat_x, phase_x = torch.sqrt(torch.abs(feat_x)), torch.angle(feat_x)

    # 先不堆叠，先连接起来，在 collate_fn 中 rearange 即可
    feat_x = torch.stack(((feat_x * torch.cos(phase_x)), (feat_x * torch.sin(phase_x))), dim=1)

    return feat_x, phase_x


if __name__ == "__main__":
    base_path = r"D:\work\speechEnhancement\datasets\voicebank_demand"
    train_clean_path = os.path.join(base_path, "clean_trainset_28spk_wav")
    train_noisy_path = os.path.join(base_path, "noisy_trainset_28spk_wav")
    train_scp_path = os.path.join(base_path, "train.scp")
    # dataset = VoiceBankDemand(train_scp_path, train_noisy_path, train_clean_path)
    # train_loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)  # 使用torch 加速处理 53 s
    # for batch in tqdm(train_loader):
    #     pass
        # print(i)
    
    # 时间更短
    # train_loader = VoiceBankDemandBatch(train_scp_path, train_noisy_path, train_noisy_path, 16)  # 38s
    # for e in range(3):
    #     for i in tqdm(range(len(train_loader))):
    #         batch = train_loader.batch()
    #         print(i)
        # print(i)