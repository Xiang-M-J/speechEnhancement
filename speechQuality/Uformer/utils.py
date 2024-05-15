from torch.utils.data import Dataset
import torchaudio
from config import *
import torch
from torch.nn.utils.rnn import pad_sequence


class DNSDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        with open(path, 'r') as f:
            self.files = f.read().splitlines()
        self.noise_path = []
        self.clean_path = []
        for file in self.files:
            noise, clean = file.split(",")
            self.noise_path.append(noise)
            self.clean_path.append(clean)

    def __getitem__(self, index):
        noisy_wav = self.noise_path[index]
        clean_wav = self.clean_path[index]

        pn_wav = preprocess(noisy_wav)
        pc_wav = preprocess(clean_wav)

        return pn_wav, pc_wav

    def __len__(self):
        return len(self.files)

def collate_fn(batch):
    x_batch = pad_sequence([b[0] for b in batch], batch_first=True)
    y_batch = pad_sequence([b[1] for b in batch], batch_first=True)
    return x_batch, y_batch


# def getStftSpec(wav_path):
#     """
#     使用 torch 处理数据
#     Args:
#         wav_path:

#     Returns:

#     """
#     feat_wav = preprocess(wav_path)
#     feat_x = torch.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
#                         window=torch.hann_window(win_size), return_complex=True).T
#     feat_x, phase_x = torch.abs(feat_x), torch.angle(feat_x)
#     feat_x = torch.sqrt(feat_x)  # 压缩幅度

#     return feat_x, phase_x


def preprocess(wav_path):
    """
    预处理音频，约束波形幅度
    Args:
        wav_path:

    Returns:

    """
    wav = torchaudio.load(wav_path)[0]
    wav = wav / torch.max(torch.abs(wav))
    return wav


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
