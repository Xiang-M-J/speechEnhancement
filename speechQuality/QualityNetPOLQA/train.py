import random

import numpy as np
import torch

from models import QualityNet, Cnn, TCN, QualityNetAttn, QualityNetClassifier, CnnClass, Cnn2d, CnnAttn
from trainer import Trainer
from trainerClassifier import TrainerC
from trainerSe import TrainerSE
from trainer_utils import load_dataset, Args


def load_model(args: Args):
    if args.model_type == "lstm":
        model = QualityNet(args.dropout)
    elif args.model_type == "lstmA":
        model = QualityNetAttn(args.dropout)
    elif args.model_type == "cnn":
        model = Cnn(args.cnn_filter, args.cnn_feature, args.dropout)
    elif args.model_type == "cnn2d":
        model = Cnn2d()
    elif args.model_type == "cnnA":
        model = CnnAttn(args.cnn_filter, args.cnn_feature, args.dropout)
    elif args.model_type == "tcn":
        model = TCN()
    elif args.model_type == "lstmClass":
        model = QualityNetClassifier(args.dropout, args.score_step)
    elif args.model_type == "cnnClass":
        model = CnnClass(args.dropout, args.score_step)
    else:
        raise ValueError("Invalid model type")

    if "lstm" in args.model_type:
        W = dict(model.lstm.named_parameters())
        bias_init = np.concatenate((np.zeros([100]), forget_gate_bias * np.ones([100]), np.zeros([200])))

        for name, wight in model.lstm.named_parameters():
            if "bias" in name:
                W[name] = torch.tensor(bias_init, dtype=torch.float32)

        model.lstm.load_state_dict(W)
    return model


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    arg = Args("lstmClass")
    arg.epochs = 35
    arg.batch_size = 32
    arg.save = False
    arg.lr = 1e-3
    # arg.step_size = 5
    arg.score_step = 0.2
    # 训练 CNN / tcn
    # arg.optimizer_type = 1
    # arg.enableFrame = False
    arg.focal_gamma = 2

    arg.smooth = True
    if arg.save:
        arg.write(arg.model_name)
    print(arg)

    forget_gate_bias = -3
    seed_everything(arg.random_seed)

    # x: (batch_size, seq_len, feature_dim), y1: (batch_size,), y2: (batch_size, seq_len)
    train_dataset, valid_dataset, test_dataset = load_dataset("wav_polqa_mini.list", arg.spilt_rate, arg.fft_size,
                                                              arg.hop_size)
    # trainer = Trainer(arg)
    trainer = TrainerC(arg)
    model = load_model(arg)
    model = trainer.train(model, train_dataset=train_dataset, valid_dataset=valid_dataset)
    trainer.test(test_dataset=test_dataset, model=model)
    # trainer.test(test_dataset=test_dataset, model_path=r"D:\work\speechEnhancement\speechQuality\QualityNetPOLQA\models\QN20240508_174129\best.pt")
