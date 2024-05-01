

import numpy as np
from dc_crn import DCCRN
from utils import VoiceBankDemand, collate_fn
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


epochs = 50
batch_size = 2
lr = 1e-3

base_path = r"D:\work\speechEnhancement\datasets\voicebank_demand"
train_clean_path = os.path.join(base_path, "clean_trainset_28spk_wav")
train_noisy_path = os.path.join(base_path, "noisy_trainset_28spk_wav")
train_scp_path = os.path.join(base_path, "train.scp")
test_clean_path = os.path.join(base_path, "clean_testset_wav")
test_noisy_path = os.path.join(base_path, "noisy_testset_wav")
test_scp_path = os.path.join(base_path, "test.scp")

train_dataset = VoiceBankDemand(train_scp_path, train_noisy_path, train_clean_path)
test_dataset = VoiceBankDemand(test_scp_path, test_noisy_path, test_clean_path)

train_dataset, valid_dataset = random_split(train_dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

writer = SummaryWriter()


net = DCCRN(rnn_units=256,masking_mode='E',use_clstm=True,kernel_num=[32, 64, 128, 256, 256,256])
net = net.to(device=device)

optim = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

model_path = "models"
loss_path = "loss"

train_step = len(train_loader)
valid_step = len(valid_loader)
last_valid_loss = 0
dec_counter = 0
train_losses = []
valid_losses = []


for epoch in tqdm(range(epochs)):
    train_loss = 0
    valid_loss = 0
    count = 0
    net.train()
    for i, batch in tqdm(enumerate(train_loader)):
        x = batch[0].to(device)
        y = batch[1].to(device)
        y_pred = net(x)[1]
        if y_pred.shape[1] < y.shape[1]:
            y = y[:, :y_pred.shape[1]]
        loss = net.loss(y_pred, y, loss_mode="SI-SNR")
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.data.item()
    print("epoch {}: train loss: {:.3f}".format(epoch, train_loss / train_step))
    train_losses.append(train_loss / train_step)

    net.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            x = batch[0].to(device)
            y = batch[1].to(device)
            y_pred = net(x)[1]
            if y_pred.shape[1] < y.shape[1]:
                y = y[:, :y_pred.shape[1]]
            loss = net.loss(y_pred, y, loss_mode="SI-SNR")
            valid_loss += loss.data.item()
    print("epoch {}: valid loss: {:.3f}".format(epoch, valid_loss / valid_step))
    valid_losses.append(valid_loss / valid_step)
    if valid_loss >= last_valid_loss:
        dec_counter += 1
    else:
        dec_counter = 0
    last_valid_loss = valid_loss

    if (epoch+1) % 5 == 0:
        torch.save(net, f"{model_path}/{epoch+1}.pt")

    writer.add_scalar("train loss", train_loss/train_step, epoch)
    writer.add_scalar("valid loss", valid_loss/train_step, epoch)
    np.save(f"{loss_path}/loss.npy", np.array({"train_loss": train_losses, "valid_loss": valid_losses}))

    if dec_counter == 3:    # 损失连续上升 3 次，降低学习率
        lr = lr / 2
        print(f"epoch {epoch}: lr half to {lr}")
        for param_groups in optim.param_groups:
            param_groups['lr'] = lr
    elif dec_counter == 5:                       # 损失连续上升 5 次，停止训练
        print(f"epoch {epoch}: early stop")
        torch.save(net, f"{model_path}/best.pt")
        break

    writer.add_scalar("lr", lr, epoch)

torch.save(net, f"{model_path}/final.pt")

# test phase

test_loss = 0
test_step = len(test_loader)
net.eval()

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        x = batch[0].to(device)
        y = batch[1].to(device)
        y_pred = net(x)[1]
        if y_pred.shape[1] < y.shape[1]:
            y = y[:, :y_pred.shape[1]]
        loss = net.loss(y_pred, y, loss_mode="SI-SNR")
        test_loss += loss.data.item()


np.save(f"{loss_path}/test_loss.npy", test_loss / len(test_loader))

writer.add_text("test loss", str(test_loss / test_step))
writer.close()
