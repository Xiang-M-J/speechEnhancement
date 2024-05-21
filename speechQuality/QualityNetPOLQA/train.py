import random

import numpy as np
import torch

from trainer import Trainer
from trainerClassifier import TrainerC
from trainerSe import TrainerSE
from trainer_utils import load_dataset_qn, Args, load_dataset_se, load_se_model, load_qn_model


def load_model(args: Args):
    if args.model_type.endswith('_se'):
        return load_se_model(args)
    else:
        return load_qn_model(args)


def load_trainer(args: Args):
    if args.model_type.endswith('_se'):
        return TrainerSE(args)
    elif args.model_type.endswith('Class'):
        return TrainerC(args)
    else:
        return Trainer(args)


def load_pretrained_model(path):
    model = torch.load(path)
    return model


def load_dataset(type, path, spilt_rate, fft_size=512, hop_size=256, input_type=2, return_wav=False):
    if type.endswith('_se'):
        if not path.endswith("_se.list"):
            raise ValueError("Path must end with _se")
        return load_dataset_se(path, spilt_rate, fft_size, hop_size, input_type)
    else:
        if path.endswith("_se.list"):
            raise ValueError("Path can not end with _se")
        return load_dataset_qn(path, spilt_rate, fft_size, hop_size, return_wav)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    # arg = Args("dpcrn_se", model_name="dpcrn_se20240518_224558")
    arg = Args("cn2n")
    arg.epochs = 30
    arg.batch_size = 128
    arg.save = True
    arg.lr = 5e-4
    arg.step_size = 10
    arg.delta_loss = 1e-3
    arg.se_input_type = 2

    # 用于 qualityNet
    arg.normalize_output = True

    # 训练Hubert
    # arg.optimizer_type = 1
    # arg.enable_frame = False

    # 训练 CNN / tcn
    arg.optimizer_type = 1
    arg.enableFrame = False

    # 训练分类模型
    # arg.score_step = 0.2
    # arg.focal_gamma = 2
    # arg.smooth = True

    if arg.save:
        arg.write(arg.model_name)
    print(arg)

    seed_everything(arg.random_seed)

    # 加载用于预测polqa分数的数据集 x: (B, L, C), y1: (B,), y2: (B, L)
    # 加载用于训练语音增强模型的数据集 x: (B, L, C)  y: (B L C)
    # train_dataset, valid_dataset, test_dataset = load_dataset(arg.model_type, "wav_train_se.list", arg.spilt_rate,
    #                                                           arg.fft_size, arg.hop_size, arg.se_input_type)
    train_dataset, valid_dataset, test_dataset = load_dataset(arg.model_type, "wav_train_qn.list", arg.spilt_rate,
                                                              arg.fft_size, arg.hop_size, arg.se_input_type, )

    model = load_model(arg)
    # model = load_pretrained_model(
    #     r"D:\work\speechEnhancement\speechQuality\QualityNetPOLQA\models\dpcrn_se20240518_224558\final.pt")

    # 当主模型名以_se结尾时，返回 TrainerSE，以Class结尾时，返回TrainerC，其余情况返回Trainer
    trainer = load_trainer(arg)
    model = trainer.train(model, train_dataset=train_dataset, valid_dataset=valid_dataset)
    trainer.test(test_dataset=test_dataset, model=model, q_len=200)

    # trainer.inference_step(model, r"D:\work\speechEnhancement\datasets\dns_to_liang\31435_nearend.wav",
    #                        r"D:\work\speechEnhancement\datasets\dns_to_liang\31435_target.wav")
    # trainer.test(test_dataset=test_dataset, model_path=r"D:\work\speechEnhancement\speechQuality\QualityNetPOLQA\models\QN20240508_174129\best.pt")
