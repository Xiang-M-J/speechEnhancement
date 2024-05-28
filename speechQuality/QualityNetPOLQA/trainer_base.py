import abc
import os

import soundfile
import torch
from torch.utils.tensorboard import SummaryWriter

from trainer_utils import Args, plot_spectrogram
from utils import get_logging, preprocess, CalSigmos, getStftSpec, spec2wav

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TrainerBase(abc.ABC):
    """
    基类
    """

    def __init__(self, args: Args):
        self.args: Args = args
        if self.validateArg(self.args):
            raise ValueError("参数错误")
        self.optimizer_type = args.optimizer_type
        self.model_path = f"models/{args.model_name}/"
        self.best_model_path = self.model_path + "best.pt"  # 模型保存路径(max val acc)
        self.final_model_path = self.model_path + "final.pt"  # 模型保存路径(final)
        self.result_path = f"results/{args.model_name}/"  # 结果保存路径（分为数据和图片）
        self.image_path = self.result_path + "images/"
        self.data_path = self.result_path + "data/"
        self.inference_result_path = f"inference_results/{args.model_name}"
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.save_model_epoch = args.save_model_epoch
        self.lr = args.lr
        self.mask_target = args.mask_target
        self.se_input_type = args.se_input_type
        self.fft_size = args.fft_size
        self.hop_size = args.hop_size
        self.normalize_output = args.normalize_output
        self.score_step = args.score_step
        self.iteration = args.iteration
        self.iter_step = args.iter_step
        self.save_model_step = args.save_model_step
        self.qn_compress = args.qn_compress
        self.logging = get_logging("log.txt")
        if args.save:
            self.check_dir()
            self.writer = SummaryWriter("runs/" + self.args.model_name)

    @staticmethod
    def validateArg(arg: Args):
        if "lstm_se" in arg.model_name and arg.se_input_type != 1:
            return True
        if "dpcrn" in arg.model_name and arg.se_input_type != 2:
            return True
        if "dpcrn" in arg.model_name and arg.optimizer_type != 1:
            return True
        if "lstm_se" in arg.model_name and arg.optimizer_type != 3:
            return True
        print(f"optimizer type is {arg.optimizer_type}")
        return False

    def check_dir(self):
        """
        创建文件夹
        """

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    def get_optimizer(self, parameter, lr):
        if self.optimizer_type == 0:
            optimizer = torch.optim.SGD(params=parameter, lr=lr, weight_decay=self.args.weight_decay)
            # 对于SGD而言，L2正则化与weight_decay等价
        elif self.optimizer_type == 1:
            optimizer = torch.optim.Adam(params=parameter, lr=lr, betas=(self.args.beta1, self.args.beta2),
                                         weight_decay=self.args.weight_decay)
            # 对于Adam而言，L2正则化与weight_decay不等价
        elif self.optimizer_type == 2:
            optimizer = torch.optim.AdamW(params=parameter, lr=lr, betas=(self.args.beta1, self.args.beta2),
                                          weight_decay=self.args.weight_decay)
        elif self.optimizer_type == 3:
            optimizer = torch.optim.RMSprop(parameter, lr=lr, weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError
        return optimizer

    def get_scheduler(self, optimizer, arg: Args):
        if arg.scheduler_type == 0:
            return None
        elif arg.scheduler_type == 1:
            return torch.optim.lr_scheduler.StepLR(optimizer, arg.step_size, arg.gamma)
        elif arg.scheduler_type == 2:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=arg.epochs, eta_min=1e-6)
        else:
            raise NotImplementedError

    @abc.abstractmethod
    def get_loss_fn(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def train_epoch(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def test_step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def test(self, *args, **kwargs):
        pass

    def inference_step(self, model, noise_wav_path, target_wav_path):
        if not os.path.exists(self.inference_result_path):
            os.makedirs(self.inference_result_path)
        model.eval()
        wav_name = noise_wav_path.split('\\')[-1].split('.')[0]
        noise_wav, fs = preprocess(noise_wav_path)
        calSigmos = CalSigmos(fs=48000, batch=False)
        fig1 = plot_spectrogram(noise_wav, fs, self.args.fft_size, self.args.hop_size,
                                filename=wav_name + "_带噪语音语谱图",
                                result_path=self.inference_result_path)

        target_wav, fs = preprocess(target_wav_path)
        fig2 = plot_spectrogram(target_wav, fs, self.args.fft_size, self.args.hop_size,
                                filename=wav_name + "_干净语音语谱图",
                                result_path=self.inference_result_path)

        feat_x, phase_x = getStftSpec(noise_wav, self.args.fft_size, self.args.hop_size, self.args.fft_size,
                                      self.args.se_input_type)
        with torch.no_grad():
            est_x = model(feat_x.unsqueeze(0).to(device)).squeeze(0).cpu().detach()
        est_wav = spec2wav(est_x.unsqueeze(0), phase_x, self.args.fft_size, self.args.hop_size, self.args.fft_size,
                           self.args.se_input_type)
        est_wav = est_wav.squeeze(0).cpu()
        fig3 = plot_spectrogram(est_wav, fs, self.args.fft_size, self.args.hop_size,
                                filename=wav_name + "_增强语音语谱图",
                                result_path=self.inference_result_path)

        result = calSigmos(est_wav.cpu().detach().numpy())
        print(result)
        with open(os.path.join(self.inference_result_path, wav_name + "mos.txt"), 'w', encoding="utf-8") as f:
            f.write(noise_wav_path)
            for r in result:
                f.write(str(r) + "\t")
            f.write("\n")

        denoise_wav_path = wav_name + "_denoise.wav"
        soundfile.write(os.path.join(self.inference_result_path, denoise_wav_path), est_wav.numpy(), samplerate=48000)
        if self.args.save:
            self.writer.add_audio(wav_name, noise_wav, sample_rate=48000)
            if target_wav_path is not None:
                self.writer.add_audio(wav_name + "_无噪声", target_wav, sample_rate=48000)
            self.writer.add_audio(wav_name + "_增强后", est_wav, sample_rate=48000)
