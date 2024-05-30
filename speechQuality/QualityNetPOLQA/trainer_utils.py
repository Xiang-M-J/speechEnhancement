import os
import random
import time

import torch
from sklearn.metrics import classification_report, confusion_matrix
import torchinfo

from models import HASANetStack, Cnn, LstmClassifier, CnnClass, Cnn2d, \
    HASANet, CAN2dClass, LstmCANClass, CnnMAttn, HASAClassifier
from lstm import lstm_net
from DPCRN import dpcrn
from hubert import Hubert
import numpy as np
import yaml
from matplotlib import pyplot as plt

from utils import ListRead, DNSPOLQADataset, DNSDataset, floatTensorToClass, floatNumpyToClass

plt.rcParams['font.sans-serif'] = ['Simhei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['font.size'] = "14.0"
dpi = 300
forget_gate_bias = -3


def load_dataset_qn(path, spilt_rate, fft_size=512, hop_size=256, return_wav=False, input_type=1):
    wav_list = ListRead(path)
    random.shuffle(wav_list)

    train_length = int(len(wav_list) * spilt_rate[0])
    valid_length = int(len(wav_list) * spilt_rate[1])

    Train_list = wav_list[:train_length]

    Valid_list = wav_list[train_length:train_length + valid_length]

    Test_list = wav_list[train_length + valid_length:]

    train_dataset = DNSPOLQADataset(Train_list, fft_size, hop_size, return_wav=return_wav, input_type=input_type)
    valid_dataset = DNSPOLQADataset(Valid_list, fft_size, hop_size, return_wav=return_wav, input_type=input_type)
    test_dataset = DNSPOLQADataset(Test_list, fft_size, hop_size, return_wav=return_wav, input_type=input_type)
    return train_dataset, valid_dataset, test_dataset


def load_pretrained_model(path):
    model = torch.load(path)
    return model


def load_dataset_se(path, spilt_rate, fft_size=512, hop_size=256, input_type=2):
    wav_list = ListRead(path)
    random.shuffle(wav_list)

    train_length = int(len(wav_list) * spilt_rate[0])
    valid_length = int(len(wav_list) * spilt_rate[1])

    Train_list = wav_list[:train_length]

    Valid_list = wav_list[train_length:train_length + valid_length]

    Test_list = wav_list[train_length + valid_length:]

    train_dataset = DNSDataset(Train_list, fft_num=fft_size, win_shift=hop_size, win_size=fft_size,
                               input_type=input_type)
    valid_dataset = DNSDataset(Valid_list, fft_num=fft_size, win_shift=hop_size, win_size=fft_size,
                               input_type=input_type)
    test_dataset = DNSDataset(Test_list, fft_num=fft_size, win_shift=hop_size, win_size=fft_size, input_type=input_type)
    return train_dataset, valid_dataset, test_dataset


def qn_type(input_type: int):
    if input_type == 0:
        return "_or"
    elif input_type == 1:
        return "_cp"
    elif input_type == 2:
        return "_st"
    else:
        raise ValueError(f"Input type {input_type} is not supported")


class Args:
    def __init__(self,
                 model_type,
                 task_type="",
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
                 enable_frame=True,
                 smooth=True,
                 cnn_filter=128,
                 cnn_feature=64,
                 focal_gamma=2,
                 normalize_output=False,
                 se_input_type=2,
                 qn_input_type=1,
                 mask_target=None,
                 iteration=10000,
                 iter_step=100,
                 save_model_step=10,
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
            enable_frame: 是否允许 Frame loss Default: True
            smooth: 是否平滑标签 Default: True
            focal_gamma: focal loss 中的gamma
            model2_type: 第二个模型（qualityNet）的类型
            normalize_output: 归一化语音质量模型的输出
            se_input_type: 语音增强模型的输入类型（1：lstm，只输入压缩过的幅度谱，2：dpcrn）
            qn_input_type: 语音质量模型的输入类型 (0: 无压缩的幅度谱，1：有压缩的幅度谱，2：dpcrn的输入)
            task_type: 任务类型
            mask_target: 是否训练mask(IAM)
        """

        # 基础参数
        if task_type == "_se":
            self.model_type = model_type + task_type + ("" if mask_target is None else ("_" + mask_target))
        elif task_type == "_qn":
            self.model_type = model_type + qn_type(qn_input_type) + task_type
        else:
            self.model_type = model_type + task_type

        if model_name is None:
            self.now_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            model_name = self.model_type + ("" if model2_type is None else model2_type)
            self.model_name = model_name + self.now_time
        else:
            self.now_time = model_name[-15:]
            self.model_name = model_name
        self.epochs = epochs
        self.dropout = dropout
        self.random_seed = random_seed
        self.save = save
        self.save_model_epoch = save_model_epoch
        self.scheduler_type = scheduler_type
        self.load_weight = load_weight
        self.model2_type = model2_type

        now_time_f = time.mktime(time.strptime(self.now_time, "%Y%m%d_%H%M%S"))
        now_time_t = time.mktime(time.localtime())
        self.expire = now_time_t > now_time_f + 10  # 如果实际当前时间比当前认为的时间大10s，则证明传入了model_name

        # 语音质量模型
        self.normalize_output = normalize_output
        self.qn_input_type = qn_input_type

        # 语音质量模型与语音增强模型
        self.iteration = iteration
        self.iter_step = iter_step
        self.save_model_step = save_model_step

        # 语音增强模型相关
        self.se_input_type = se_input_type
        self.mask_target = mask_target

        # 损失函数相关
        self.enable_frame = enable_frame
        self.smooth = smooth
        self.score_step = score_step
        self.score_class_num = int(400) // int(score_step * 100)
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
            self.sig = None
            self.ovrl = None
            if with_acc:
                self.train_acc = []
                self.valid_acc = []
                self.best_valid_acc = 0.
        elif mode == "test":
            self.mode = "test"
            self.test_loss = 0
            self.mse = 0.
            self.cm = None
            self.lcc = None
            self.srcc = None
            self.pesq = None
            self.polqa = None
            self.stoi = None
            self.mos_48k_name = ["MOS_COL", "MOS_DISC", "MOS_LOUD", "MOS_NOISE", "MOS_REVERB", "MOS_SIG", "MOS_OVRL"]
            self.mos_48k = None
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
        if self.patience2 == 1:
            print(f"The validation loss continue increase in {self.patience_} iterations, stop train")
            print(f"The final validation loss is {val_loss}")
            return True
        if self.patience == 1:
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


def plot_quantity(quantity: list, title: str = '', xlabel=None, ylabel: str = '', filename: str = None,
                  result_path: str = "results/"):
    if filename is None:
        filename = title.replace(' ', '_')
    plt.clf()
    fig = plt.figure(dpi=dpi)
    mean_ = np.mean(quantity)
    plt.scatter(range(len(quantity)), quantity, s=3)
    x = [0, len(quantity)]
    y = [mean_, mean_]
    plt.plot(x, y, linewidth=2, color='r')
    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
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


class LoaderIterator:
    """
    将 loader 转为迭代器
    """

    def __init__(self, loader) -> None:
        self.idx = 0
        self.loader = loader
        self.max_idx = len(loader)
        self.iterator = iter(loader)

    def __call__(self):
        if self.idx == self.max_idx:
            self.iterator = iter(self.loader)
            self.idx = 0
        ret_value = next(self.iterator)
        self.idx += 1
        return ret_value


def get_model_type(full_model_type):
    if "_" in full_model_type:
        return full_model_type.split('_')[0]
    else:
        return full_model_type


def load_qn_model(args: Args):
    model_type = get_model_type(args.model_type)
    if model_type == "cnn":
        model = Cnn(args.cnn_filter, args.cnn_feature, args.dropout)
    elif model_type == "hasa":
        model = HASANet()
    elif model_type == "hasaStack":
        model = HASANetStack()
    elif model_type == "cnn2d":
        model = Cnn2d()
    elif model_type == "cnnA":
        model = CnnMAttn()
    elif model_type == "can2dClass":
        model = CAN2dClass(args.score_class_num)
    elif model_type == "lstmClass":
        model = LstmClassifier(args.dropout, args.score_class_num)
    elif model_type == "lstmcanClass":
        model = LstmCANClass(args.dropout, args.score_class_num)
    elif model_type == "hasaClass":
        model = HASAClassifier(args.score_class_num)
    elif model_type == "cnnClass":
        model = CnnClass(args.score_step)
    elif model_type == "hubert":
        model = Hubert()
    else:
        raise ValueError("Invalid model type")

    # if "lstm" in model_type:
    #     W = dict(model.lstm.named_parameters())
    #     bias_init = np.concatenate((np.zeros([100]), forget_gate_bias * np.ones([100]), np.zeros([200])))
    #
    #     for name, wight in model.lstm.named_parameters():
    #         if "bias" in name:
    #             W[name] = torch.tensor(bias_init, dtype=torch.float32)
    #
    #     model.lstm.load_state_dict(W)
    return model


def load_se_model(args: Args):
    if "lstm" in args.model_type:
        model = lstm_net(args.fft_size)
    elif "dpcrn" in args.model_type:
        model = dpcrn(args.fft_size)
    else:
        raise ValueError("Error se_model_type")
    return model


def report(y_true, y_pred):
    length = y_true.shape[1]
    r = classification_report(y_true, y_pred, target_names=np.arange(1, length + 1), output_dict=True)
    return r


def confuseMatrix(y_true, y_pred, step, num):
    pred = floatNumpyToClass(y_pred, step)
    labels = floatNumpyToClass(y_true, step)
    cm = confusion_matrix(labels, pred, labels=np.arange(1, num + 1))
    return cm


def plot_matrix(cm, labels_name, title='混淆矩阵', normalize=False, result_path: str = "results/"):
    """绘制混淆矩阵，保存并返回

    Args:
        cm: 计算出的混淆矩阵的值
        labels_name: 标签名
        title: 生成的混淆矩阵的标题
        normalize: True:显示百分比, False:显示个数
        result_path: 保存路径

    Returns: 图窗

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=plt.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例
    # 图像标题
    plt.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    # x,y轴长度一致(问题1解决办法）
    plt.axis("equal")
    # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    thresh = cm.max() / 2.
    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if normalize:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                         ha="center", va="center",
                         color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
            else:
                plt.text(j, i, str(int(cm[i][j])),
                         ha="center", va="center",
                         color="white" if cm[i][j] > thresh else "black")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.ylabel('真实类别')
    plt.xlabel('预测类别')
    img_path = os.path.join(result_path, f"{title}.png")
    plt.savefig(img_path, dpi=dpi)

    return fig


def log_model(model, result_path):
    summary = torchinfo.summary(model, col_names=("output_size", "num_params", "kernel_size"),
                                row_settings=("depth", "ascii_only"), input_size=(4, 512, 257))
    text = str(summary)
    with open(os.path.join(result_path, "model.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


# def plot_model(model, result_path):
#     summary = torchinfo.summary(model, col_names=("output_size", "num_params", "kernel_size"), row_settings=("depth","ascii_only"), input_size=(4, 512, 257))
#     text = str(summary)
#     width = summary.formatting.col_width * len(summary.formatting.col_names) * 16
#     height = len(summary.summary_list) * 24 + 100
#     im = Image.new("RGB", (width, height), (255, 255, 255))
#     dr = ImageDraw.Draw(im)
#     font = ImageFont.truetype(os.path.join("C:/Windows/fonts", "consola.ttf"), 16)

#     dr.text((10, 5), text, font=font,  fill="#000000")
#     fig = plt.figure(figsize=(width/200, height/200), dpi=300)
#     with open(os.path.join(result_path, "model.txt"), "w", encoding="utf-8") as f:
#         f.write(text)
#     plt.imshow(im)
#     return fig


if __name__ == '__main__':
    # m = {"train": [1, 2, 3, 4, 5, 6], "test": [2, 3, 4, 5, 6, 7]}
    # plot_metric(m, "train and test", xlabel="epoch", ylabel="loss", filename="train1")

    # metric = Metric(mode="test")
    # metric.lcc = [[1, 2], [3, 4]]
    # metric.srcc = [[1, 2]]
    # print(metric)
    predict = torch.randn(64, 20)
    true = torch.randn(64, 20)
    cm = confuseMatrix(true, predict)
    plot_matrix(cm, np.arange(1, 21), normalize=False)
