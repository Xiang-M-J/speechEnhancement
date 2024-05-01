

from wav_dataset import VoiceBankDemand, collate_fn
import wav_loader as loader
import net_config as net_config
import pickle
from torch.utils.data import DataLoader
import module as model_cov_bn
from si_snr import *
import train_utils
import os

base_path = r"D:\work\speechEnhancement\datasets\voicebank_demand"
train_clean_path = os.path.join(base_path, "clean_trainset_28spk_wav")
train_noisy_path = os.path.join(base_path, "noisy_trainset_28spk_wav")
train_scp_path = os.path.join(base_path, "train.scp")
test_clean_path = os.path.join(base_path, "clean_testset_wav")
test_noisy_path = os.path.join(base_path, "noisy_testset_wav")
test_scp_path = os.path.join(base_path, "test.scp")

########################################################################
# Change the path to the path on your computer

save_file = "./logs"  # model save
########################################################################

batch_size = 400  # calculate batch_size
load_batch = 100  # load batch_size(not calculate)
device = torch.device("cuda:0")  # device

lr = 0.001  # learning_rate

train_dataset = VoiceBankDemand(train_scp_path, train_noisy_path, train_clean_path)
test_dataset = VoiceBankDemand(test_scp_path, test_noisy_path, test_clean_path)

# dataloader
train_dataloader = DataLoader(train_dataset, batch_size=load_batch, collate_fn=collate_fn, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=load_batch, collate_fn=collate_fn, shuffle=True)

dccrn = model_cov_bn.DCCRN_(
    n_fft=512, hop_len=int(6.25 * 16000 / 1000), net_params=net_config.get_net_params(), batch_size=batch_size,
    device=device, win_length=int((25 * 16000 / 1000))).to(device)

optimizer = torch.optim.Adam(dccrn.parameters(), lr=lr)
criterion = SiSnr()
train_utils.train(model=dccrn, optimizer=optimizer, criterion=criterion, train_iter=train_dataloader,
                  test_iter=test_dataloader, max_epoch=500, device=device, batch_size=batch_size, log_path=save_file,
                  just_test=False)
