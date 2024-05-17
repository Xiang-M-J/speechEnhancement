import os
import random
import time

import torch

from models import QualityNet, Cnn, TCN, QualityNetAttn, QualityNetClassifier, CnnClass, Cnn2d, CnnAttn
from lstm import lstm_net
from DPCRN import dpcrn
import numpy as np
import yaml
from matplotlib import pyplot as plt

from utils import ListRead, DNSPOLQADataset, DNSDataset

plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['font.size'] = "14.0"
dpi = 300
forget_gate_bias = -3


def load_dataset_qn(path, spilt_rate, fft_size=512, hop_size=256):
    wav_list = ListRead(path)
    random.shuffle(wav_list)

    train_length = int(len(wav_list) * spilt_rate[0])
    valid_length = int(len(wav_list) * spilt_rate[1])

    Train_list = wav_list[:train_length]

    Valid_list = wav_list[train_length:train_length + valid_length]

    Test_list = wav_list[train_length + valid_length:]

    train_dataset = DNSPOLQADataset(Train_list, fft_size=fft_size, hop_size=hop_size)
    valid_dataset = DNSPOLQADataset(Valid_list, fft_size=fft_size, hop_size=hop_size)
    test_dataset = DNSPOLQADataset(Test_list, fft_size=fft_size, hop_size=hop_size)
    return train_dataset, valid_dataset, test_dataset


def load_dataset_se(path, spilt_rate, fft_size=512, hop_size=256):
    wav_list = ListRead(path)
    random.shuffle(wav_list)

    train_length = int(len(wav_list) * spilt_rate[0])
    valid_length = int(len(wav_list) * spilt_rate[1])

    Train_list = wav_list[:train_length]

    Valid_list = wav_list[train_length:train_length + valid_length]

    Test_list = wav_list[train_length + valid_length:]

    train_dataset = DNSDataset(Train_list, fft_num=fft_size, win_shift=hop_size, win_size=fft_size)
    valid_dataset = DNSDataset(Valid_list, fft_num=fft_size, win_shift=hop_size, win_size=fft_size)
    test_dataset = DNSDataset(Test_list, fft_num=fft_size, win_shift=hop_size, win_size=fft_size)
    return train_dataset, valid_dataset, test_dataset


class Args:
    def __init__(self,
                 model_type,
                 model2_type=None,
                 model_name=None,
                 epochs=35,
                 lr=1e-3,
                 fft_size=512,
                 hop_size=256,
                 batch_size=64,
                 spilt_rate=None,
                 weight_decay=0,
                 patience=5,
                 delta_loss=1e-3,
                 optimizer_type=3,
                 shuffle=True,
                 beta1=0.99,
                 beta2=0.999,
                 random_seed=34,
                 save=True,
                 save_model_epoch=5,
                 scheduler_type=1,
                 gamma=0.3,
                 step_size=10,
                 dropout=0.3,
                 score_step=0.2,
                 load_weight=False,
                 enableFrame=True,
                 smooth=True,
                 cnn_filter=128,
                 cnn_feature=64,
                 focal_gamma=2,
                 ):
        """
        Args:
            epochs: number of epochs: 35
            lr: learning rate Default: 1e-3
            dropout: dropout rate Default: 0.3
            optimizer_type: 优化器类别(0: SGD, 1:Adam, 2:AdamW, 3: RMSp)
            scheduler_type: 计划器类别(0: None, 1: StepLR, 2: CosineAnnealingLR) Default: 1
            beta1: adam优化器参数
            beta2: adam优化器参数
            random_seed: 随机数种子
            save: 是否保存模型和结果
            scheduler_type: scheduler类型
            gamma: LR scheduler参数
            step_size: LR scheduler参数  Default: 5
            shuffle: 是否打乱数据 Default: True
            score_step: 分数分布步长
            enableFrame: 是否允许 Frame loss Default: True
            smooth: 是否平滑标签 Default: True
            focal_gamma: focal loss 中的gamma
            model2_type: 第二个模型的类型
        """

        # 基础参数
        if model_name is None:
            model_name = model_type
        self.model_name = model_name + time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.epochs = epochs
        self.dropout = dropout
        self.random_seed = random_seed
        self.model_type = model_type
        self.save = save
        self.save_model_epoch = save_model_epoch
        self.scheduler_type = scheduler_type
        self.load_weight = load_weight
        self.model2_type = model2_type

        # 损失函数相关
        self.enableFrame = enableFrame
        self.smooth = smooth
        self.score_step = score_step
        self.focal_gamma = focal_gamma

        # cnn 相关
        self.cnn_filter = cnn_filter
        self.cnn_feature = cnn_feature

        # 用于数据集
        if spilt_rate is None:
            spilt_rate = [0.8, 0.1, 0.1]
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.spilt_rate = spilt_rate

        # stft
        self.fft_size = fft_size
        self.hop_size = hop_size

        # 用于优化器
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.beta1 = beta1
        self.beta2 = beta2

        # 用于 step scheduler
        self.step_size = step_size
        self.gamma = gamma

        # 用于早停止
        self.patience = patience
        self.delta_loss = delta_loss

    def write(self, name="hyperparameter"):
        with open("./config/" + name + ".yml", 'w', encoding='utf-8') as f:
            yaml.dump(data=self.__dict__, stream=f, allow_unicode=True)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            hyperparameter = yaml.load(f.read(), Loader=yaml.FullLoader)
        for para in hyperparameter:
            self.__setattr__(para, hyperparameter[para])

    def items(self):
        return self.__dict__

    def __str__(self) -> str:
        info = "parameter setting:\t"
        for parameter in self.__dict__:
            info += f"{parameter}: {self.__dict__[parameter]}\t"
        info += '\n'
        return info


class Metric:
    """
    存储模型训练和测试时的指标
    """

    def __init__(self, mode="train", with_acc=False):
        if mode == "train":
            self.mode = "train"
            self.train_loss = []
            self.valid_loss = []
            self.best_valid_loss = 100.
            if with_acc:
                self.train_acc = []
                self.valid_acc = []
                self.best_valid_acc = 0.
        elif mode == "test":
            self.mode = "test"
            self.test_loss = 0
            self.mse = 0.
            self.lcc = None
            self.srcc = None
            self.pesq = None
            self.predictMos = None
            if with_acc:
                self.test_acc = 0
        else:
            print("wrong mode !!! use default mode train")
            self.mode = "train"
            self.train_loss = []
            self.valid_loss = []
            self.best_valid_loss = 0
            if with_acc:
                self.train_acc = []
                self.valid_acc = []
                self.best_valid_acc = 0.

    def items(self) -> dict:
        """
        返回各种指标的字典格式数据
        Returns: dict

        """
        # if self.mode == "train":
        #     data = {"train_loss": self.train_loss, "valid_loss": self.valid_loss,
        #             'best_valid_loss': self.best_valid_loss}
        # else:
        #     data = {"test_loss": self.test_loss, "mse": self.mse, "lcc": self.lcc, "srcc": self.srcc}
        data = self.__dict__.copy()
        data.pop("mode")
        key_list = list(data.keys())
        for key in key_list:
            if data[key] is None:
                data.pop(key)
        return data

    def __str__(self) -> str:
        info = ""
        items = self.items()
        for key in items.keys():
            info += f"{key}: {items[key]}\n"
        return info


class EarlyStopping:
    """Early stops the training if validation accuracy doesn't change after a given patience."""

    def __init__(self, patience=5, delta_loss=1e-3):
        """
        当损失变动小于 delta_loss 或者上升超过 delta_loss 超过 5 次，停止训练
        Args:
            patience (int): 可以容忍的次数
        """
        self.patience = patience
        self.patience2 = patience
        self.patience_ = patience
        self.delta_loss = delta_loss
        self.last_val_loss = 100.0

    def __call__(self, val_loss) -> bool:
        if abs(self.last_val_loss - val_loss) < self.delta_loss:
            self.patience -= 1
        else:
            self.patience = self.patience_
        if val_loss > self.last_val_loss:
            self.patience2 -= 1
        else:
            self.patience2 = self.patience_
        self.last_val_loss = val_loss
        if self.patience2 == 0:
            print(f"The validation loss continue increase in {self.patience_} iterations, stop train")
            print(f"The final validation loss is {val_loss}")
            return True
        if self.patience == 0:
            print(f"The validation loss has not changed in {self.patience_} iterations, stop train")
            print(f"The final validation loss is {val_loss}")
            return True
        return False


def plot_metric(metric: dict, title: str = '', xlabel: str = 'epoch', ylabel: str = 'loss', legend=None,
                filename: str = None, result_path: str = "results/"):
    if legend is None:
        legend = metric.keys()
    if filename is None:
        filename = title.replace(' ', '_')

    assert len(legend) == len(metric.keys())
    fig = plt.figure(dpi=dpi)
    for key in metric.keys():
        plt.plot(metric[key])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.savefig(os.path.join(result_path, f"{filename}.png"), dpi=dpi)
    return fig


def plot_spectrogram(wav, fs, fft_size, hop_size, title: str = "语谱图", filename=None, result_path: str = "results/"):
    if filename is None:
        filename = title.replace(' ', '_')
    plt.clf()
    fig = plt.figure(dpi=dpi, figsize=(20, 10))
    plt.specgram(wav, Fs=fs, window=np.hanning(fft_size), noverlap=hop_size, NFFT=fft_size)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(result_path, f"{filename}.png"), dpi=dpi)
    return fig


def load_qn_model(args: Args):
    if args.model_type == "lstm":
        model = QualityNet(args.dropout)
    elif args.model_type == "lstmA":
        model = QualityNetAttn(args.dropout)
    elif args.model_type == "cnn":
        model = Cnn(args.cnn_filter, args.cnn_feature, args.dropout)
    elif args.model_type == "cnn2d":
        model = Cnn2d()
    elif args.model_type == "cnnA":
        model = CnnAttn(args.cnn_filter, args.cnn_feature, args.dropout)
    elif args.model_type == "tcn":
        model = TCN()
    elif args.model_type == "lstmClass":
        model = QualityNetClassifier(args.dropout, args.score_step)
    elif args.model_type == "cnnClass":
        model = CnnClass(args.dropout, args.score_step)
    else:
        raise ValueError("Invalid model type")

    if "lstm" in args.model_type:
        W = dict(model.lstm.named_parameters())
        bias_init = np.concatenate((np.zeros([100]), forget_gate_bias * np.ones([100]), np.zeros([200])))

        for name, wight in model.lstm.named_parameters():
            if "bias" in name:
                W[name] = torch.tensor(bias_init, dtype=torch.float32)

        model.lstm.load_state_dict(W)
    return model


def load_se_model(args: Args):
    if args.model_type == "lstm_se":
        model = lstm_net(args.fft_size)
    elif args.model_type == "dpcrn_se":
        model = dpcrn(args.fft_size)
    else:
        raise ValueError("Error se_model_type")
    return model


if __name__ == '__main__':
    # m = {"train": [1, 2, 3, 4, 5, 6], "test": [2, 3, 4, 5, 6, 7]}
    # plot_metric(m, "train and test", xlabel="epoch", ylabel="loss", filename="train1")

    metric = Metric(mode="test")
    metric.lcc = [[1, 2], [3, 4]]
    metric.srcc = [[1, 2]]
    print(metric)
