# Force matplotlib to not use any Xwindows backend.
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import POLQADataset, ListRead, FrameMse

epoch = 25
batch_size = 32
forget_gate_bias = -3  # Please see tha paper for more details

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training data
wav_list = ListRead("wav_polqa.list")
random.shuffle(wav_list)

Train_list = wav_list[:10000]
train_num = len(Train_list)

# Testing data
Test_list = wav_list[10000:12000]
test_num = len(Test_list)

start_time = time.time()
print('model building...')


class QualityNet(nn.Module):
    def __init__(self) -> None:
        super(QualityNet, self).__init__()
        self.lstm = nn.LSTM(257, 100, num_layers=2, bidirectional=True, dropout=0.3, batch_first=True)
        self.linear1 = nn.Linear(200, 50)  # 2 * 100
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.3)

        self.linear2 = nn.Linear(50, 1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        l1 = self.dropout(self.elu(self.linear1(lstm_out)))
        Frame_score = self.linear2(l1).squeeze(-1)
        Average_score = self.avgpool(Frame_score)
        return Frame_score, Average_score


model = QualityNet()

# Initialization of the forget gate bias (optional)
W = dict(model.lstm.named_parameters())
bias_init = np.concatenate((np.zeros([100]), forget_gate_bias * np.ones([100]), np.zeros([200])))

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

scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=15, gamma=0.2)

print('training...')

train_dataset = POLQADataset(Train_list)
test_dataset = POLQADataset(Test_list)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_steps = len(train_loader)
test_steps = len(test_loader)

train_losses = []
val_losses = []

for e in tqdm(range(epoch)):
    train_loss = 0
    eval_loss = 0
    model.train()
    for batch_idx, (x, y) in (enumerate(train_loader)):
        y1 = y[0]
        y2 = y[1]
        frameS, avgS = model(x)
        l1 = loss1(avgS.squeeze_(-1), y1)
        l2 = loss2(frameS, y2)
        loss = l1 + l2
        train_loss += loss.cpu().detach().numpy()
        optim.zero_grad()
        loss.backward()
        optim.step()
    print("epoch: {}, train loss: {}".format(e, train_loss / train_steps))
    train_losses.append(train_loss / train_steps)

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            y1 = y[0]
            y2 = y[1]
            frameS, avgS = model(x)
            l1 = loss1(avgS.squeeze_(-1), y1)
            l2 = loss2(frameS, y2)
            loss = l1 + l2
            eval_loss += loss.cpu().detach().numpy()
    print("epoch: {}, valid loss: {}".format(e, eval_loss / test_steps))
    val_losses.append(eval_loss / test_steps)
    scheduler.step()
    torch.save(model, 'Quality-Net1.pt')

np.save('data/train_loss.npy', train_losses)
np.save('data/val_loss.npy', val_losses)

plt.plot(range(epoch), train_losses, val_losses)
plt.savefig('train_loss.png')

model = torch.load('Quality-Net1.pt')
model.eval()
print('testing...')
POLQA_Predict = np.zeros([test_num, ])
POLQA_True = np.zeros([test_num, ])
idx = 0
with torch.no_grad():
    for batch_idx, (x, y) in enumerate(test_loader):
        y1 = y[0]
        frameS, avgS = model(x)
        batch = frameS.shape[0]
        est_polqa = avgS.cpu().detach().numpy()
        true_polqa = y1.cpu().detach().numpy()
        for i in range(batch):
            POLQA_Predict[idx + i] = est_polqa[i, 0]
            POLQA_True[idx + i] = true_polqa[i]
        idx += batch

MSE = np.mean((POLQA_True - POLQA_Predict) ** 2)
print('Test error= %f' % MSE)
LCC = np.corrcoef(POLQA_True, POLQA_Predict)
print('Linear correlation coefficient= %f' % float(LCC[0][1]))

SRCC = scipy.stats.spearmanr(POLQA_True.T, POLQA_Predict.T)
print('Spearman rank correlation coefficient= %f' % SRCC[0])

# Plotting the scatter plot
M = np.max([np.max(POLQA_Predict), 5])
plt.figure(1)
plt.scatter(POLQA_True, POLQA_Predict, s=14)
plt.xlim([0, M])
plt.ylim([0, M])
plt.xlabel('True PESQ')
plt.ylabel('Predicted PESQ')
plt.title('LCC= %f, SRCC= %f, MSE= %f' % (float(LCC[0][1]), SRCC[0], MSE))
plt.show()
plt.savefig('Scatter_plot_QualityNet1_pt.png', dpi=150)

end_time = time.time()
print('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))
