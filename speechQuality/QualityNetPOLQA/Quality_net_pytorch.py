import matplotlib
# Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot as plt
import soundfile

import torch
import torch.nn as nn
import scipy.io
import scipy.stats
import librosa
import os
import time
import numpy as np
import random

from torch.utils.data import DataLoader

from utils import DNSPOLQADataset
from tqdm import tqdm

random.seed(999)
matplotlib.use('Agg')

epoch = 15
batch_size = 16
forgetgate_bias = -3  # Please see tha paper for more details

NUM_EandN = 8000
NUM_Clean = 800

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FrameMse(nn.Module):
    def __init__(self) -> None:
        super(FrameMse, self).__init__()

    def forward(self, input, target):
        true_pesq = target[0, 0]
        return (10 ** (true_pesq - 4.5)) * torch.mean((input - target) ** 2)


def frame_mse(y_true,
              y_pred):  # Customized loss function  (frame-level loss, the second term of equation 1 in the paper)
    True_pesq = y_true[0, 0]
    # tf.reduce_mean --> torch.mean
    return (10 ** (True_pesq - 4.5)) * torch.mean((y_true - y_pred) ** 2)


def Global_average(x):
    return 4.5 * torch.mean(x, dim=-2)


def shuffle_list(x_old, index):
    x_new = [x_old[i] for i in index]
    return x_new


def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


def Sp_and_phase(path, Noisy=False):
    signal, rate = soundfile.read(path)
    signal = signal / np.max(abs(signal))

    F = librosa.stft(signal, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)

    Lp = np.abs(F)
    phase = np.angle(F)
    if Noisy:
        meanR = np.mean(Lp, axis=1).reshape((257, 1))
        stdR = np.std(Lp, axis=1).reshape((257, 1)) + 1e-12
        NLp = (Lp - meanR) / stdR
    else:
        NLp = Lp

    NLp = np.reshape(NLp.T, (1, NLp.shape[1], 257))
    return NLp, phase


def ListRead(path):
    with open(path, 'r') as f:
        file_list = f.read().splitlines()
    return file_list


def train_data_generator(file_list):
    index = 0
    while True:
        pesq_filepath = file_list[index].split(',')
        noisy_LP, _ = Sp_and_phase(pesq_filepath[0])
        pesq = np.asarray(float(pesq_filepath[1])).reshape([1])

        index += 1
        if index == len(file_list):
            index = 0

            random.shuffle(file_list)

        yield noisy_LP, [pesq, pesq[0] * np.ones([1, noisy_LP.shape[1], 1])]


def val_data_generator(file_list):
    index = 0
    while True:
        pesq_filepath = file_list[index].split(',')
        noisy_LP, _ = Sp_and_phase(pesq_filepath[0])
        pesq = np.asarray(float(pesq_filepath[1])).reshape([1])

        index += 1
        if index == len(file_list):
            index = 0

        yield noisy_LP, [pesq, pesq[0] * np.ones([1, noisy_LP.shape[1], 1])]


# Training data
wav_list = ListRead("wav_polqa.list")

Train_list = wav_list[:10000]
random.shuffle(Train_list)
Num_train = len(Train_list)

# Testing data
Test_list = wav_list[10000:12000]
Num_testdata = len(Test_list)

start_time = time.time()
print('model building...')


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class QualityNet(nn.Module):
    def __init__(self) -> None:
        super(QualityNet, self).__init__()
        self.lstm = nn.LSTM(257, 200, bidirectional=True, dropout=0.3, batch_first=True)
        self.linear1 = TimeDistributed(nn.Linear(400, 200))  # 2 * 100
        self.linear1_ = TimeDistributed(nn.Linear(200, 50))
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.3)

        self.linear2 = TimeDistributed(nn.Linear(50, 1))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        l1 = self.dropout(self.elu(self.linear1(lstm_out)))
        l1 = self.dropout(self.elu(self.linear1_(l1)))
        Frame_score = self.linear2(l1).squeeze(-1)
        Average_score = self.avgpool(Frame_score)
        return Frame_score, Average_score


model = QualityNet()

# Initialization of the forget gate bias (optional)
W = dict(model.lstm.named_parameters())
bias_init = np.concatenate((np.zeros([200]), forgetgate_bias * np.ones([200]), np.zeros([400])))

for name, wight in model.lstm.named_parameters():
    if "bias" in name:
        W[name] = torch.tensor(bias_init, dtype=torch.float32)

model.lstm.load_state_dict(W)

model.to(device=device)

loss1 = nn.MSELoss()
loss2 = FrameMse()
loss1.to(device=device)
loss2.to(device=device)
optim = torch.optim.RMSprop(model.parameters(), lr=1e-3)

print('training...')
g1 = train_data_generator(Train_list)
g2 = val_data_generator(Test_list)

train_dataset = DNSPOLQADataset(Train_list)
test_dataset = DNSPOLQADataset(Test_list)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

for e in tqdm(range(epoch)):
    train_loss = 0
    eval_loss = 0
    model.train()
    for step in range(NUM_Clean):
        x, y = g1.__next__()
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y1 = torch.tensor(y[0], dtype=torch.float32, device=device)
        y2 = torch.tensor(y[1], dtype=torch.float32, device=device)
        frameS, avgS = model(x)
        l1 = loss1(avgS, y1)
        l2 = loss2(frameS, y2)
        loss = l1 + l2
        train_loss += loss.cpu().detach().numpy()
        optim.zero_grad()
        loss.backward()
        optim.step()
    print("epoch: {}, train loss: {}".format(e, train_loss / Num_train))

    model.eval()
    with torch.no_grad():
        for step in range(Num_testdata):
            x, y = g2.__next__()
            x = torch.tensor(x, dtype=torch.float32, device=device)
            y1 = torch.tensor(y[0], dtype=torch.float32, device=device)
            y2 = torch.tensor(y[1], dtype=torch.float32, device=device)
            frameS, avgS = model(x)
            l1 = loss1(avgS, y1)
            l2 = loss2(frameS, y2)
            loss = l1 + l2
            eval_loss += loss.cpu().detach().numpy()
    print("epoch: {}, valid loss: {}".format(e, eval_loss / Num_testdata))
    torch.save(model, 'Quality-Net_(Non-intrusive).pt')

model = torch.load('Quality-Net_(Non-intrusive).pt')
model.eval()
print('testing...')
PESQ_Predict = np.zeros([len(Test_list), ])
PESQ_true = np.zeros([len(Test_list), ])
with torch.no_grad():
    for step in range(Num_testdata):
        x, y = g2.__next__()
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y1 = torch.tensor(y[0], dtype=torch.float32, device=device)
        frameS, avgS = model(x)
        PESQ_Predict[step] = avgS.cpu().detach().numpy()
        PESQ_true[step] = y1.cpu().detach().numpy()

MSE = np.mean((PESQ_true - PESQ_Predict) ** 2)
print('Test error= %f' % MSE)
LCC = np.corrcoef(PESQ_true, PESQ_Predict)
print('Linear correlation coefficient= %f' % LCC[0][1])
SRCC = scipy.stats.spearmanr(PESQ_true.T, PESQ_Predict.T)
print('Spearman rank correlation coefficient= %f' % SRCC[0])

# Plotting the scatter plot
M = np.max([np.max(PESQ_Predict), 4.55])
plt.figure(1)
plt.scatter(PESQ_true, PESQ_Predict, s=14)
plt.xlim([0, M])
plt.ylim([0, M])
plt.xlabel('True PESQ')
plt.ylabel('Predicted PESQ')
plt.title('LCC= %f, SRCC= %f, MSE= %f' % (LCC[0][1], SRCC[0], MSE))
plt.show()
plt.savefig('Scatter_plot_Quality-Net_(Non-intrusive)_pt.png', dpi=150)

# plotting the learning curve


end_time = time.time()
print('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))
