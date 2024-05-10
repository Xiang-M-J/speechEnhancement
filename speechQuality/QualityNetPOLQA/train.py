import random

import numpy as np
import torch

from models import QualityNet, Cnn, TCN, QualityNetAttn
from trainer import Trainer
from trainer_utils import load_dataset, Args


def load_model(args: Args):
    if args.model_type == "lstm":
        model = QualityNet(args.dropout)
        W = dict(model.lstm.named_parameters())
        bias_init = np.concatenate((np.zeros([100]), forget_gate_bias * np.ones([100]), np.zeros([200])))

        for name, wight in model.lstm.named_parameters():
            if "bias" in name:
                W[name] = torch.tensor(bias_init, dtype=torch.float32)

        model.lstm.load_state_dict(W)
    elif args.model_type == "lstmA":
        model = QualityNetAttn(args.dropout)
        W = dict(model.lstm.named_parameters())
        bias_init = np.concatenate((np.zeros([100]), forget_gate_bias * np.ones([100]), np.zeros([200])))

        for name, wight in model.lstm.named_parameters():
            if "bias" in name:
                W[name] = torch.tensor(bias_init, dtype=torch.float32)
    elif args.model_type == "cnn":
        model = Cnn()
    elif args.model_type == "tcn":
        model = TCN()

    else:
        raise ValueError("Invalid model type")
    return model


if __name__ == '__main__':
    arg = Args("cnn")
    arg.epochs = 40
    arg.batch_size = 64
    arg.save = True
    arg.lr = 1e-3

    # 训练 CNN / tcn
    arg.optimizer_type=1
    # arg.weight_decay = 0.1
    arg.enableFrame = False
    if arg.save:
        arg.write(arg.model_name)
    torch.manual_seed(arg.random_seed)
    np.random.seed(arg.random_seed)
    random.seed(arg.random_seed)
    forget_gate_bias = -3

    train_dataset, valid_dataset, test_dataset = load_dataset("wav_polqa_mid.list", arg.spilt_rate, arg.fft_size,
                                                              arg.hop_size)
    trainer = Trainer(arg)
    model = load_model(arg)
    model = trainer.train(model, train_dataset=train_dataset, valid_dataset=valid_dataset)
    trainer.test(test_dataset=test_dataset, model=model)
    # trainer.test(test_dataset=test_dataset, model_path=r"D:\work\speechEnhancement\speechQuality\QualityNetPOLQA\models\QN20240508_174129\best.pt")
