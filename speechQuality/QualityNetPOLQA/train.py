from trainer import Trainer
from trainer_utils import load_dataset_qn, Args, load_qn_model
from utils import seed_everything

if __name__ == '__main__':

    # arg = Args("dpcrn_se", model_name="dpcrn_se20240518_224558")
    arg = Args("hasa")
    arg.epochs = 30
    arg.batch_size = 16
    arg.save = False
    arg.lr = 5e-4
    arg.step_size = 10
    arg.delta_loss = 1e-3

    # 用于 qualityNet
    arg.normalize_output = True

    # 训练Hubert
    # arg.optimizer_type = 1
    # arg.enable_frame = False

    # 训练 CNN / tcn
    arg.optimizer_type = 1
    arg.enableFrame = False

    print(arg)
    if arg.save:
        arg.write(arg.model_name)

    seed_everything(arg.random_seed)

    # 加载用于预测polqa分数的数据集 x: (B, L, C), y1: (B,), y2: (B, L)
    train_dataset, valid_dataset, test_dataset = load_dataset_qn("wav_train_qn.list", arg.spilt_rate,
                                                                 arg.fft_size, arg.hop_size, )

    model = load_qn_model(arg)

    trainer = Trainer(arg)
    model = trainer.train(model, train_dataset=train_dataset, valid_dataset=valid_dataset)
    trainer.test(test_dataset=test_dataset, model=model)
