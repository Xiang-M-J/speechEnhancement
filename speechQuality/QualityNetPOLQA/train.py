import random

import numpy as np
import torch

from models import QualityNet, Cnn, TCN, QualityNetAttn, QualityNetClassifier, QualityNetClassifier2, CnnClass
from trainer import Trainer
from trainerClassifier import TrainerC
from trainer_utils import load_dataset, Args


def load_model(args: Args):
    if args.model_type == "lstm":
        model = QualityNet(args.dropout)
    elif args.model_type == "lstmA":
        model = QualityNetAttn(args.dropout)
    elif args.model_type == "cnn":
        model = Cnn(args.cnn_filter, args.cnn_feature, args.dropout)
    elif args.model_type == "tcn":
        model = TCN()
    elif args.model_type == "lstmClass":
        model = QualityNetClassifier(args.dropout, args.score_step)
    elif args.model_type == "lstmClass2":
        model = QualityNetClassifier2(args.dropout, args.score_step)
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


if __name__ == '__main__':
    arg = Args("lstmClass")
    arg.epochs = 35
    arg.batch_size = 64
    arg.save = False
    arg.lr = 1e-3
    # arg.step_size = 5
    # 训练 CNN / tcn
    arg.score_step = 0.4
    # arg.optimizer_type = 1
    # arg.enableFrame = False

    arg.smooth = False
    if arg.save:
        arg.write(arg.model_name)
    print(arg)
    torch.manual_seed(arg.random_seed)
    np.random.seed(arg.random_seed)
    random.seed(arg.random_seed)
    forget_gate_bias = -3

    # x: (batch_size, seq_len, feature_dim), y1: (batch_size,), y2: (batch_size, seq_len)
    train_dataset, valid_dataset, test_dataset = load_dataset("wav_polqa_mini.list", arg.spilt_rate, arg.fft_size,
                                                              arg.hop_size)
    # trainer = Trainer(arg)
    trainer = TrainerC(arg)
    model = load_model(arg)
    model = trainer.train(model, train_dataset=train_dataset, valid_dataset=valid_dataset)
    trainer.test(test_dataset=test_dataset, model=model)
    # trainer.test(test_dataset=test_dataset, model_path=r"D:\work\speechEnhancement\speechQuality\QualityNetPOLQA\models\QN20240508_174129\best.pt")
