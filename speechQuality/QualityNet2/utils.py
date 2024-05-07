import torch
import torchaudio

fft_num = 320
win_shift = 160
win_size = 320


def getStftSpec_t(wav_path):
    """
    使用 torch 处理数据
    Args:
        wav_path:

    Returns:

    """
    feat_wav = preprocess(wav_path)
    feat_x = torch.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
                        window=torch.hann_window(win_size), return_complex=True).T
    feat_x, phase_x = torch.abs(feat_x), torch.angle(feat_x)
    feat_x = torch.sqrt(feat_x)  # 压缩幅度

    return feat_x, phase_x


def preprocess(wav_path):
    """
    预处理音频，约束波形幅度
    Args:
        wav_path:

    Returns:

    """
    wav = torchaudio.load(wav_path)[0].squeeze(0)
    c = torch.sqrt(wav.shape[0] / torch.sum((wav ** 2.0)))
    feat_wav = wav * c
    return feat_wav


def wav2spec(feat_wav):
    c = torch.sqrt(feat_wav.shape[0] / torch.sum((feat_wav ** 2.0)))
    feat_wav = feat_wav * c

    feat_x = torch.stft(feat_wav, n_fft=fft_num, hop_length=win_shift, win_length=win_size,
                        window=torch.hann_window(win_size), return_complex=True).T
    feat_x, phase_x = torch.sqrt(torch.abs(feat_x)), torch.angle(feat_x)
    feat_x = torch.sqrt(feat_x)

    return feat_x, phase_x, c, len(feat_wav)


def spec2wav(feat, feat_phase, c, l):
    feat_mag = torch.pow(feat, 2)
    feat_mag = feat_mag.squeeze(dim=0)
    feat_phase = feat_phase.squeeze(dim=0)
    feat_de = torch.multiply(feat_mag, torch.exp(1j * feat_phase))
    est_audio = torch.istft(feat_de.T, n_fft=fft_num, hop_length=win_shift,
                            win_length=win_size, window=torch.hann_window(win_size), length=l)
    return est_audio / c


def decode(wav_path, model):
    wav, _ = torchaudio.load(wav_path)
    wav = wav.squeeze(0)
    feat, phase, c, l = wav2spec(wav)
    with torch.no_grad():
        est_feat = model(feat.unsqueeze(dim=0)).squeeze(0)
    est_wav = spec2wav(est_feat, phase, c, l)
    return est_wav


def decode2(wav, model):
    feat, phase, c, l = wav2spec(torch.tensor(wav, dtype=torch.float32))
    with torch.no_grad():
        est_feat = model(feat.unsqueeze(dim=0)).squeeze(0)
    est_wav = spec2wav(est_feat, phase, c, l)
    return est_wav.cpu().numpy()
