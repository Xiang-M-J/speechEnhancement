import os

import numpy as np
import torch.utils
from tqdm import tqdm
from tensorboardX import SummaryWriter
from DCCRN_cprs import DCCRN
from config import batch_size, lr, epochs, loss_path, check_point_path
from utils import VoiceBankDemandIter

base_path = r"D:\work\speechEnhancement\datasets\voicebank_demand"
train_clean_path = os.path.join(base_path, "clean_trainset_28spk_wav")
train_noisy_path = os.path.join(base_path, "noisy_trainset_28spk_wav")
train_scp_path = os.path.join(base_path, "train.scp")
test_clean_path = os.path.join(base_path, "clean_testset_wav")
test_noisy_path = os.path.join(base_path, "noisy_testset_wav")
test_scp_path = os.path.join(base_path, "test.scp")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

temp_dataset = VoiceBankDemandIter(train_scp_path, train_noisy_path, train_clean_path, shuffle=True)
test_loader = VoiceBankDemandIter(test_scp_path, test_noisy_path, test_clean_path)

train_files, valid_files = temp_dataset.train_valid_spilt([0.8, 0.2])
train_loader = VoiceBankDemandIter(" ", train_noisy_path, train_clean_path,
                                   batch_size=batch_size, files=train_files, shuffle=True)
valid_loader = VoiceBankDemandIter(" ", train_noisy_path, train_clean_path,
                                   batch_size=batch_size, files=valid_files, shuffle=True)
writer = SummaryWriter()

model = DCCRN(rnn_units=256, masking_mode='E', use_clstm=True, kernel_num=[32, 64, 128, 256, 256, 256])
model = model.to(device=device)
loss_fn = torch.nn.MSELoss()
loss_fn = loss_fn.to(device=device)
optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

writer.add_text("epoch", str(epochs))
writer.add_text("betas", "0.9, 0.999")

train_step = len(train_loader)
valid_step = len(valid_loader)
last_valid_loss = 0
dec_counter = 0
train_losses = []
valid_losses = []

# train phase

for epoch in tqdm(range(epochs)):
    train_loss = 0
    valid_loss = 0
    count = 0
    t_files = []
    v_files = []
    model.train()
    if train_loader.test_run_out():
        t_files = train_loader.files
        v_files = valid_loader.files
        del train_loader, valid_loader
        train_loader = VoiceBankDemandIter(" ", train_noisy_path, train_clean_path,
                                           batch_size=batch_size, files=t_files, shuffle=True)
        valid_loader = VoiceBankDemandIter(" ", train_noisy_path, train_clean_path,
                                           batch_size=batch_size, files=v_files, shuffle=True)

    for i in range(train_step):
        batch = next(train_loader)
        x = batch[0].to(device)    # B 2 feat_dim seq_len
        y = batch[1].to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.data.item()
    print("epoch {}: train loss: {:.3f}".format(epoch, train_loss / train_step))
    train_losses.append(train_loss / train_step)
    model.eval()

    with torch.no_grad():
        for i in range(valid_step):
            batch = next(valid_loader)
            x = batch[0].to(device)
            y = batch[1].to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            valid_loss += loss.data.item()
    print("epoch {}: valid loss: {:.3f}".format(epoch, valid_loss / valid_step))
    valid_losses.append(valid_loss / valid_step)
    if valid_loss > last_valid_loss:
        dec_counter += 1
    else:
        dec_counter = 0
    last_valid_loss = valid_loss
    if (epoch + 1) % 5 == 0:
        torch.save(model, f"{check_point_path}/{epoch + 1}_bi.pt")

    writer.add_scalar("train loss", train_loss / train_step, epoch)
    writer.add_scalar("valid loss", valid_loss / train_step, epoch)
    np.save(f"{loss_path}/loss_bi.npy", np.array({"train_loss": train_losses, "valid_loss": valid_losses}))
    if dec_counter == 3:  # 损失连续上升 3 次，降低学习率
        lr = lr / 2
        print(f"epoch {epoch}: lr half to {lr}")
        for param_groups in optim.param_groups:
            param_groups['lr'] = lr
    elif dec_counter == 5:  # 损失连续上升 5 次，停止训练
        print(f"epoch {epoch}: early stop")
        torch.save(model, f"BEST_MODEL/{epoch}_bi.pt")
        break
    writer.add_scalar("lr", lr, epoch)

torch.save(model, f"{check_point_path}/final_bi.pt")

# test phase

test_loss = 0
test_step = len(test_loader)
model.eval()

with torch.no_grad():
    for i in range(test_step):
        batch = next(test_loader)
        x = batch[0].to(device)
        y = batch[1].to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        test_loss += loss.data.item()

np.save(f"{loss_path}/test_loss_bi.npy", test_loss / test_step)

writer.add_text("test loss", str(test_loss / test_step))
writer.close()
