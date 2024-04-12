import os

import numpy as np
import torch.utils
from tqdm import tqdm

from CRN import crn_net
from config import batch_size, lr, epochs
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

model = crn_net()
model = model.to(device=device)
loss_fn = torch.nn.MSELoss()
loss_fn = loss_fn.to(device=device)
optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

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
        x = batch[0].to(device)
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
        torch.save(model, f"CP_dir/{epoch + 1}_b.pt")
    np.save("loss_b.npy", np.array({"train_loss": train_losses, "valid_loss": valid_losses}))
    if dec_counter == 3:  # 损失连续上升 3 次，降低学习率
        lr = lr / 2
        print(f"epoch {epoch}: lr half to {lr}")
        for param_groups in optim.param_groups:
            param_groups['lr'] = lr
    elif dec_counter == 5:  # 损失连续上升 5 次，停止训练
        print(f"epoch {epoch}: early stop")
        torch.save(model, f"BEST_MODEL/{epoch}_b.pt")
        break
torch.save(model, "CP_dir/final_b.pt")

# test phase

test_loss = 0
model.eval()

with torch.no_grad():
    for i in range(len(test_loader)):
        batch = test_loader.batch()
        x = batch[0].to(device)
        y = batch[1].to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        test_loss += loss.data.item()

np.save("test_loss_b.npy", test_loss)
