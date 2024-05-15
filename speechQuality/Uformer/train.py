import torch
import os
from tensorboardX import SummaryWriter
import torch.utils
from utils import DNSDataset
from torch.utils.data import DataLoader, random_split
from config import batch_size, lr, epochs, loss_path, check_point_path, metric_path
from uformer import Uformer
from tqdm import tqdm
import numpy as np

path = r"D:\work\speechEnhancement\speechQuality\QualityNetPOLQA\wav_se.list"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

main_dataset = DNSDataset(path)

train_dataset, valid_dataset, test_dataset = random_split(main_dataset, [0.8, 0.1, 0.1])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

writer = SummaryWriter()

model = Uformer()
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
    model.train()
    for i, batch in tqdm(enumerate(train_loader)):
        x = batch[0].to(device)
        y = batch[1].to(device)
        y_pred, _, _, _ = model(x, x)
        loss = loss_fn(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss += loss.data.item()
    print("epoch {}: train loss: {:.3f}".format(epoch, train_loss / train_step))
    train_losses.append(train_loss / train_step)

    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            x = batch[0].to(device)
            y = batch[1].to(device)
            y_pred, _, _, _ = model(x, x)
            loss = loss_fn(y_pred, y)
            valid_loss += loss.data.item()
    print("epoch {}: valid loss: {:.3f}".format(epoch, valid_loss / valid_step))
    valid_losses.append(valid_loss / valid_step)
    if valid_loss >= last_valid_loss:
        dec_counter += 1
    else:
        dec_counter = 0
    last_valid_loss = valid_loss

    if (epoch+1) % 5 == 0:
        torch.save(model, f"CP_dir/{epoch+1}.pt")

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
        torch.save(model, f"BEST_MODEL/{epoch}.pt")
        break

    writer.add_scalar("lr", lr, epoch)

torch.save(model, "CP_dir/final.pt")

# test phase

test_loss = 0
test_step = len(test_loader)
model.eval()

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        x = batch[0].to(device)
        y = batch[1].to(device)
        y_pred, _, _, _ = model(x, x)
        loss = loss_fn(y_pred, y)
        test_loss += loss.data.item()


np.save("test_loss.npy", test_loss / len(test_loader))

writer.add_text("test loss", str(test_loss / test_step))
writer.close()