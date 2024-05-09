import random

import numpy as np
import torch

from model import QualityNet, Cnn
from trainer import Trainer
from trainer_utils import load_dataset, Args


def load_model(model_type):
    if model_type == "lstm":
        m = QualityNet()
        W = dict(m.lstm.named_parameters())
        bias_init = np.concatenate((np.zeros([100]), forget_gate_bias * np.ones([100]), np.zeros([200])))

        for name, wight in m.lstm.named_parameters():
            if "bias" in name:
                W[name] = torch.tensor(bias_init, dtype=torch.float32)

        m.lstm.load_state_dict(W)
    elif model_type == "cnn":
        m = Cnn()
    return m


if __name__ == '__main__':
    arg = Args("cnn", model_name="cnn")
    arg.epochs = 20
    arg.batch_size = 64
    arg.save = False
    arg.write(arg.model_name)
    torch.manual_seed(arg.random_seed)
    np.random.seed(arg.random_seed)
    random.seed(arg.random_seed)
    forget_gate_bias = -3
    trainer = Trainer(arg)
    train_dataset, valid_dataset, test_dataset = load_dataset("wav_polqa_mini.list", arg.spilt_rate, arg.fft_size,
                                                              arg.hop_size)
    model = load_model(arg.model_type)

    model = trainer.train(model, train_dataset=train_dataset, valid_dataset=valid_dataset)

    trainer.test(test_dataset=test_dataset, model=model)
