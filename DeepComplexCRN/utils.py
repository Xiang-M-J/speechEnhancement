import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import soundfile as sf
import librosa
import torchaudio
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

        feat_noisy = load_wav(noisy_wav)
        feat_clean = load_wav(clean_wav)

        return feat_noisy, feat_clean

    def __len__(self):
        return len(self.files)


def collate_fn(batch):
    x_batch = pad_sequence([b[0] for b in batch], batch_first=True)
    y_batch = pad_sequence([b[1] for b in batch], batch_first=True)
    return x_batch, y_batch


def load_wav(path, sr=16000):
    wav = torchaudio.load(path)[0].squeeze(0)
    return wav


if __name__ == "__main__":
    load_wav("D:\\work\\speechEnhancement\\datasets\\voicebank_demand\\clean_testset_wav\\p232_001.wav", frame_dur=37.5)
    print()