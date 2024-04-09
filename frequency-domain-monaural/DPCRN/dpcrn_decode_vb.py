import torch
import argparse
import librosa
import os
import numpy as np
import json
import scipy
from Backup import *
import pickle
from config import *
#from GCRN_noncprs import Net
from DPCRN import dpcrn
#from Step1_network_id_wi_ss import Step1_net
# from LSTM import lstm_net
import soundfile as sf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def enhance(args):
    model = dpcrn()
    model.load_state_dict(torch.load('./BEST_MODEL/vb_dpcrn_noncprs_model.pth'))
    # print(model)
    model.eval()
    model.cuda()

    with torch.no_grad():
        cnt = 0
        mix_file_path = args.mix_file_path
        esti_file_path = args.esti_clean_file_path
        if not os.path.exists(esti_file_path):
            os.makedirs(esti_file_path)

        file_list = os.listdir(mix_file_path)
        for file_id in file_list:
            feat_wav, _ = sf.read(os.path.join(mix_file_path, file_id))
            c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
            feat_wav = feat_wav * c
            feat_x = librosa.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, window='hanning').T

            #compress
#            feat_x, phase_x = np.abs(feat_x) ** 0.5, np.angle(feat_x)
            feat_x, phase_x = np.abs(feat_x) ** 1.0, np.angle(feat_x)

            feat_x = torch.FloatTensor(feat_x)
            phase_x = torch.FloatTensor(phase_x)
            feat_x = torch.stack((feat_x * torch.cos(phase_x), feat_x * torch.sin(phase_x)), dim=0)
            feat_x = feat_x.cuda()
            esti_x = model(feat_x.unsqueeze(dim=0))
            esti_x_mag = torch.norm(esti_x, dim=1)
            esti_x_phase = torch.atan2(esti_x[:, 1, :, :], esti_x[:, 0, :, :])

            #compressed
#            esti_x_mag = esti_x_mag ** 2.0
            esti_x_mag = esti_x_mag ** 1.0

            esti_x_mag = esti_x_mag.squeeze(dim=0).cpu().numpy()
            esti_x_phase = esti_x_phase.squeeze(dim=0).cpu().numpy()
            de_esti = np.multiply(esti_x_mag, np.exp(1j * esti_x_phase))
            esti_utt = librosa.istft((de_esti).T, hop_length=win_shift,
                                     win_length=win_size, window='hanning', length=len(feat_wav))
            esti_utt = esti_utt / c
            # if seen_flag == 1:
            #     os.makedirs(os.path.join(esti_file_path, noise_type, 'seen', str(snr)), exist_ok=True)
            #     librosa.output.write_wav(os.path.join(esti_file_path, noise_type, 'seen', str(snr), file_id), esti_utt,
            #                              args.fs)
            # else:
            #     os.makedirs(os.path.join(esti_file_path, noise_type, 'unseen', str(snr)), exist_ok=True)
            #     librosa.output.write_wav(os.path.join(esti_file_path, noise_type, 'unseen', str(snr), file_id), esti_utt,
            #                              args.fs)
            # os.makedirs(os.path.join(esti_file_path, str(snr)), exist_ok=True)
            sf.write(os.path.join(esti_file_path, file_id), esti_utt, args.fs)
            print(' The %d utterance has been decoded!' % (cnt + 1))
            cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str,
                        default='/media/luoxiaoxue/datasets/VB_DEMAND_48K/noisy_16k')
    parser.add_argument('--esti_clean_file_path', type=str,
                        default='./datasets/vb_dpcrn_noncprs_model_new')
    parser.add_argument('--fs', type=int, default=16000)

    args = parser.parse_args()
    print(args)
    enhance(args=args)

