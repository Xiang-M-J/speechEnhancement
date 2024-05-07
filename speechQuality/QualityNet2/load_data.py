import os
import random
from glob import glob

import numpy as np
import soundfile
import multiprocessing
from utils import decode2
from CRN import crn_net
import torch

clean_wav_path = r"D:\work\speechEnhancement\datasets\TIMIT"
noise_wav_path = r"D:\work\speechEnhancement\datasets\Demand"

fs = 16
snr_list = [5 * i for i in range(-2, 6)]
snr_len = len(snr_list)
model = torch.load(r"D:\work\speechEnhancement\speechQuality\QualityNet2\speechEnhance.pt")
model = model.to("cpu")
model.eval()

class DemandNoise:
    def __init__(self, path, fs):
        self.path = path
        self.fs = fs
        wav_dirs = os.listdir(self.path)
        self.noise_types = []
        self.wav = {}
        for wav_dir in wav_dirs:
            if wav_dir.endswith(f"{fs}k"):
                self.noise_types.append(wav_dir)
                self.wav[wav_dir] = os.listdir(os.path.join(path, wav_dir, wav_dir[:-4]))

    def add_noise(self, clean_path, snr=10):
        noise_type = random.choice(self.noise_types)
        noise_path = random.choice(self.wav[noise_type])
        clean_data, sr = soundfile.read(clean_path)
        noise_data, sr = soundfile.read(os.path.join(self.path, noise_type, noise_type[:-4], noise_path))
        noise_data = noise_data / np.max(np.abs(noise_data))

        if len(noise_data) < len(clean_data):
            tmp = (len(clean_data) // len(noise_data)) + 1
            noise_data = np.tile(noise_data, tmp)
            noise_data = noise_data[:len(clean_data)]
        else:
            noise_data = noise_data[:len(clean_data)]
        clean_pwr = sum(abs(clean_data) ** 2)
        noise_pwr = sum(abs(noise_data) ** 2)  # 计算音频信号能量
        if noise_pwr == 0:
            noise_pwr = 1

        coefficient = np.sqrt((clean_pwr / (10 ** (snr * 0.1))) / noise_pwr)  # 噪声系数计算
        new_data = clean_data + coefficient * noise_data
        if max(abs(new_data)) > 32768:
            new_data = (new_data / max(abs(new_data))) * 32768
        return new_data


demandNoise = DemandNoise(noise_wav_path, fs)


def get_new_path(prefix: str, old_path: str, snr: int):
    new_path = "_".join(old_path.split("\\")[-3:])
    return os.path.join(prefix, new_path[:-8] + "_" + str(snr).replace("-", "m") + ".wav")


def start_fun(clean_path):
    for i in range(len(clean_path)):
        for k in range(snr_len):
            new_data = demandNoise.add_noise(clean_path[i], snr_list[k])
            new_path = get_new_path(r"D:\work\speechEnhancement\speechQuality\QualityNet2\noise_wavs", clean_path[i],
                                    snr_list[k])
            soundfile.write(new_path, new_data, samplerate=fs * 1000)


def start_fun2(clean_path):
    for i in range(len(clean_path)):
        for k in range(snr_len):
            new_data = demandNoise.add_noise(clean_path[i], snr_list[k])
            est_wav = decode2(new_data,model)
            new_path = get_new_path(r"D:\work\speechEnhancement\speechQuality\QualityNet2\enhance_wavs", clean_path[i],
                                    snr_list[k])
            soundfile.write(new_path, est_wav, samplerate=fs * 1000)


def get_wav_list():
    train_wav = glob("{}\\data\\TRAIN/*/*/*.wav".format(clean_wav_path), recursive=True)
    train_wav = np.array(train_wav[1::2])
    random_index = np.random.permutation(len(train_wav))
    clean_wav = train_wav[random_index[:250]]
    noise_wav = train_wav[random_index[250:500]]
    enhance_wav = train_wav[random_index[500:750]]

    with open("list/clean.list", "w") as f:
        for wav in clean_wav:
            f.write(wav + "\n")
    with open("list/noise.list", "w") as f:
        for wav in noise_wav:
            f.write(wav + "\n")
    with open("list/enhance.list", "w") as f:
        for wav in enhance_wav:
            f.write(wav + "\n")


def read_wav_list():
    with open("list/clean.list", "r") as f:
        clean_wav = f.read().splitlines()
    with open("list/noise.list", "r") as f:
        noise_wav = f.read().splitlines()
    with open("list/enhance.list", "r") as f:
        enhance_wav = f.read().splitlines()
    return clean_wav, noise_wav, enhance_wav


if __name__ == "__main__":
    # get_wav_list()
    clean_wav, noise_wav, enhance_wav = read_wav_list()

    cpu_num = multiprocessing.cpu_count()
    chunks = len(noise_wav) // cpu_num + 1
    # process = []
    # for i in range(cpu_num):
    #     temp = noise_wav[i * chunks:(i + 1) * chunks]
    #     p = multiprocessing.Process(target=start_fun, args=(temp,))
    #     p.start()
    #     process.append(p)
    # for p in process:
    #     p.join()

    start_fun2(clean_wav)
