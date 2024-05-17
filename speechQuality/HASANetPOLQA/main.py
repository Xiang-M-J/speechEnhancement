import argparse
import os
import random

import numpy as np
import torch
import yaml

from module import BLSTM_frame_sig_att
from trainer import Trainer
from trainer_utils import load_dataset, Args


def yaml_config_hook(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)
    if "defaults" in cfg.keys():
        del cfg["defaults"]
    return cfg


def load_model(config_path):

    parser = argparse.ArgumentParser(description="Combine_Net")
    config = yaml_config_hook(config_path)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    margs = parser.parse_args()
    model = BLSTM_frame_sig_att(margs.input_size, margs.hidden_size, margs.num_layers, margs.dropout, margs.linear_output, margs.act_fn)

    return model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    arg = Args("hasa")
    arg.epochs = 35
    arg.batch_size = 8
    arg.save = True
    arg.lr = 1e-3
    # arg.step_size = 5

    arg.smooth = True

    print(arg)
    setup_seed(arg.random_seed)
    forget_gate_bias = -3

    # x: (batch_size, seq_len, feature_dim), y1: (batch_size,), y2: (batch_size, seq_len)
    train_dataset, valid_dataset, test_dataset = load_dataset("wav_train_qn.list", arg.spilt_rate, arg.fft_size,
                                                              arg.hop_size)
    trainer = Trainer(arg)
    model = load_model("./hyper.yaml")
    model = trainer.train(model, train_dataset=train_dataset, valid_dataset=valid_dataset)
    trainer.test(test_dataset=test_dataset, model=model)
    # trainer.test(test_dataset=test_dataset,
    # model_path=r"D:\work\speechEnhancement\speechQuality\QualityNetPOLQA\models\QN20240508_174129\best.pt")
