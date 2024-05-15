import os

# front-end parameter settings
win_size = 512
fft_num = 512
win_shift = 256

loss_dir = './LOSS/wsj0_si84_300h_dpcrn_noncprs_loss.mat'
batch_size = 4
epochs = 50
lr = 1e-3
model_best_path = './BEST_MODEL/wsj0_si84_300h_dpcrn_noncprs_model.pth'
check_point_path = './models'
loss_path = "./loss"
metric_path = "./metircs"

os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs(loss_path, exist_ok=True)
os.makedirs(check_point_path, exist_ok=True)
os.makedirs(metric_path, exist_ok=True)