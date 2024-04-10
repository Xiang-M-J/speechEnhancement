import os

# front-end parameter settings
win_size = 512
fft_num = 512
win_shift = 256
chunk_length = int(8.0*16000)
#WSJ
#json_dir = '/media/luoxiaoxue/datasets/wsj0_si84_300h/Json'
#file_path = '/media/luoxiaoxue/datasets/wsj0_si84_300h'
#VB
json_dir = '/media/luoxiaoxue/datasets/VB_DEMAND_48K/json'
file_path = '/media/luoxiaoxue/datasets/VB_DEMAND_48K'
loss_dir = './LOSS/vb_crn_noncprs_loss.mat'
batch_size = 2
epochs = 60
lr = 1e-3
model_best_path = './BEST_MODEL/vb_crn_noncprs_model.pth'
check_point_path = './CP_dir'

os.makedirs('./BEST_MODEL', exist_ok=True)
os.makedirs('./LOSS', exist_ok=True)
os.makedirs(check_point_path, exist_ok=True)