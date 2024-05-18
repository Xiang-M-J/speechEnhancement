import random

import numpy as np
import torch

from trainerQSE import TrainerQSE
from trainer_utils import Args, load_dataset_se


def load_pretrained_model(path):
    model = torch.load(path)
    return model


def load_dataset(path, spilt_rate, fft_size=512, hop_size=256, input_type=2):
    return load_dataset_se(path, spilt_rate, fft_size, hop_size, input_type)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    path_se = r"D:\work\speechEnhancement\speechQuality\QualityNetPOLQA\models\dpcrn_se20240515_235913\final.pt"
    # path_qn = r"D:\work\speechEnhancement\speechQuality\QualityNetPOLQA\models\cnn20240515_100107\final.pt"
    path_qn = r"D:\work\speechEnhancement\speechQuality\HASANetPOLQA\models\hasa20240516_134107\final.pt"
    # arg = Args("dpcrn_qse", model_name="dpcrn_qse20240517_150155", model2_type="hasa")
    arg = Args("dpcrn_qse", model2_type="hasa")
    arg.epochs = 15
    arg.batch_size = 12
    arg.save = True
    arg.lr = 5e-4
    arg.step_size = 5
    arg.delta_loss = 2e-4

    if not arg.model_type.endswith("_qse"):
        raise ValueError("Model type must end with '_qse'")
    if arg.model2_type is None:
        raise ValueError("model qn type cannot be none")
    # 训练 CNN / tcn
    # arg.optimizer_type = 1
    # arg.enableFrame = False

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
    train_dataset, valid_dataset, test_dataset = load_dataset("wav_train_se.list", arg.spilt_rate,
                                                              arg.fft_size, arg.hop_size, arg.se_input_type)

    model_se = load_pretrained_model(path_se)
    model_qn = load_pretrained_model(path_qn)

    # 当主模型名以_se结尾时，返回 TrainerSE，以Class结尾时，返回TrainerC，其余情况返回Trainer
    trainer = TrainerQSE(arg)

    model = trainer.train(model_se, model_qn, train_dataset=train_dataset, valid_dataset=valid_dataset)
    trainer.test(test_dataset=test_dataset, model=model_se, model_qn=model_qn, q_len=500)
    # trainer.test(test_dataset=test_dataset, model_path=r"D:\work\speechEnhancement\speechQuality\QualityNetPOLQA\models\QN20240508_174129\best.pt")
