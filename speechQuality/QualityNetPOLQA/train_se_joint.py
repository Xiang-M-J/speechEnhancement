import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from tqdm import tqdm

from losses import QNLoss
from trainer_base import TrainerBase
from trainer_utils import Args, EarlyStopping, Metric, plot_metric, plot_quantity, \
    load_pretrained_model, load_dataset_se, load_se_model
from utils import spec2wav, CalSigmos, seed_everything, cal_mask_target, apply_mask_target

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TrainerSEJ(TrainerBase):
    """
    训练语音增强模型
    """

    def __init__(self, args: Args):

        super().__init__(args)

    def get_loss_fn(self):
        loss_fn = QNLoss(self.normalize_output, ("Class" in self.args.model2_type), self.args.score_step).to(device)
        loss_fn2 = nn.MSELoss(reduction='mean').to(device)
        return loss_fn, loss_fn2

    def train_epoch(self, model, model_qn, x, y, loss_fn, loss_fn2, optimizer, batch_idx):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        mag_pred, mag_true = cal_mask_target(x, y, y_pred, self.mask_target, self.se_input_type)
        if (batch_idx + 1) % 20 == 0:
            l1 = loss_fn(model_qn, mag_pred)
            l2 = loss_fn2(mag_pred, mag_true)
            loss = l1 + l2
            loss.requires_grad_(True)
        else:
            loss = loss_fn2(mag_pred, mag_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(self, model, model_qn, x, y, loss_fn, loss_fn2):
        """
        Return loss, y_pred
        """
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        mag_pred, mag_true = cal_mask_target(x, y, y_pred, self.mask_target, self.se_input_type)
        l1 = loss_fn(model_qn, mag_pred)
        l2 = loss_fn2(mag_pred, mag_true)
        loss = l1 + l2
        return loss.item(), y_pred.cpu().detach()

    def freeze_parameters(self, model: nn.Module, names=None):
        if names is None:
            for name, parameter in model.named_parameters():
                parameter.requires_grad = False
        else:
            for name, parameter in model.named_parameters():
                if name.split(".")[0] in names:
                    parameter.requires_grad = False

    def train(self, model_se: nn.Module, model_qn: nn.Module, train_dataset, valid_dataset):
        print("begin train")
        # 设置一些参数
        best_valid_loss = 100.
        metric = Metric()

        # 加载数据集
        train_loader = dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=self.args.shuffle
        )
        valid_loader = dataloader.DataLoader(
            dataset=valid_dataset,
            batch_size=self.batch_size,
            shuffle=self.args.shuffle
        )

        train_step = len(train_loader)
        valid_step = len(valid_loader)

        # 设置优化器、早停止和scheduler

        early_stop = EarlyStopping(patience=self.args.patience, delta_loss=self.args.delta_loss)

        # 设置损失函数和模型
        # loss_fn = self.get_loss_fn()
        model_se = model_se.to(device)
        model_qn = model_qn.to(device)
        self.freeze_parameters(model=model_qn, names=None)

        loss_fn, loss_fn2 = self.get_loss_fn()

        optimizer = self.get_optimizer(
            [{"params": model_se.parameters(), "lr": self.lr}], lr=self.lr)
        scheduler = self.get_scheduler(optimizer, arg=self.args)

        # 保存一些信息
        if self.args.save:
            self.writer.add_text("模型名", self.args.model_name)
            self.writer.add_text('超参数', str(self.args))
            try:
                if "dpcrn" in self.args.model_type:
                    dummy_input = torch.rand(self.args.batch_size, 2, 128, self.args.fft_size // 2 + 1).to(device)
                else:
                    dummy_input = torch.rand(self.args.batch_size, 128, self.args.fft_size // 2 + 1).to(device)
                if self.args.model_type == "lstmA":
                    pass
                else:
                    self.writer.add_graph(model_se, dummy_input)
            except RuntimeError as e:
                print(e)

        plt.ion()
        start_time = time.time()
        calSigmos = CalSigmos()

        # 训练开始
        try:
            for epoch in tqdm(range(self.epochs), ncols=100):
                train_loss = 0
                valid_loss = 0
                model_se.train()
                model_qn.train()
                loop_train = tqdm(enumerate(train_loader), leave=False)
                for batch_idx, (x, _, y, _) in loop_train:
                    loss = self.train_epoch(model_se, model_qn, x, y, loss_fn, loss_fn2, optimizer, batch_idx)
                    train_loss += loss

                    loop_train.set_description_str(f'Training [{epoch + 1}/{self.epochs}]')
                    loop_train.set_postfix_str("step: {}/{} loss: {:.4f}".format(batch_idx, train_step, loss))

                model_se.eval()
                model_qn.eval()
                q_len = 50
                idx = 0
                mos_48k = {"col": [], "disc": [], "loud": [], "noise": [], "reverb": [], "sig": [], "ovrl": []}
                mos_48k_name = list(mos_48k.keys())

                with torch.no_grad():
                    loop_valid = tqdm(enumerate(valid_loader), leave=False)
                    for batch_idx, (x, xp, y, yp) in loop_valid:
                        loss, est_x = self.predict(model_se, model_qn, x, y, loss_fn, loss_fn2)
                        valid_loss += loss
                        batch = x.shape[0]
                        if idx < q_len:
                            est_wav = spec2wav(x, xp, est_x, self.fft_size, self.hop_size, self.fft_size,
                                               input_type=self.se_input_type, mask_target=self.mask_target)
                            results = calSigmos(est_wav.cpu().detach().numpy())
                            for i in range(7):
                                mos_48k[mos_48k_name[i]].extend(results[i])
                        idx += batch
                        loop_valid.set_description_str(f'Validating [{epoch + 1}/{self.epochs}]')
                        loop_valid.set_postfix_str("step: {}/{} loss: {:.4f}".format(batch_idx, valid_step, loss))

                print("")
                for k, v in mos_48k.items():
                    print(f"{k}: {np.mean(v)}", end="\t")
                print("")

                if self.args.scheduler_type != 0:
                    scheduler.step()

                # 保存每个epoch的训练信息
                metric.train_loss.append(train_loss / train_step)
                metric.valid_loss.append(valid_loss / valid_step)

                # 实时显示损失变化
                plt.clf()
                plt.plot(metric.train_loss)
                plt.plot(metric.valid_loss)
                plt.ylabel("loss")
                plt.xlabel("epoch")
                plt.legend(["train loss", "valid loss"])
                plt.title(f"{self.args.model_type} loss")
                plt.pause(0.02)
                plt.ioff()  # 关闭画图的窗口

                if self.args.save:
                    if (epoch + 1) % self.save_model_epoch == 0:
                        tqdm.write(f"save model to {self.model_path}" + f"{epoch}.pt")
                        torch.save(model_se, self.model_path + f"{epoch}.pt")
                    self.writer.add_scalar('train loss', metric.train_loss[-1], epoch + 1)
                    self.writer.add_scalar('valid loss', metric.valid_loss[-1], epoch + 1)
                    self.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
                    np.save(os.path.join(self.data_path, "train_metric.npy"), metric.items())
                tqdm.write(
                    'Epoch {}:  train Loss:{:.4f}\t val Loss:{:.4f}'.format(
                        epoch + 1, metric.train_loss[-1], metric.valid_loss[-1]))

                if metric.valid_loss[-1] < best_valid_loss:
                    tqdm.write(f"valid loss decrease from {best_valid_loss :.3f} to {metric.valid_loss[-1]:.3f}")
                    best_valid_loss = metric.valid_loss[-1]
                    metric.best_valid_loss = best_valid_loss
                    if self.args.save:
                        torch.save(model_se, self.best_model_path)
                        tqdm.write(f"saving model to {self.best_model_path}")
                else:
                    tqdm.write(f"validation loss did not decrease from {best_valid_loss}")

                if early_stop(metric.valid_loss[-1]):
                    if self.args.save:
                        torch.save(model_se, self.final_model_path)
                        tqdm.write(f"early stop..., saving model to {self.final_model_path}")
                    break
            # 训练结束时需要进行的工作
            end_time = time.time()
            tqdm.write('Train ran for %.2f minutes' % ((end_time - start_time) / 60.))

            self.logging.info(self.args.model_name + "\t{:.2f}".format((end_time - start_time) / 60.))
            self.logging.info("train loss: {:.4f}, valid loss: {:.4f}"
                              .format(metric.train_loss[-1], metric.valid_loss[-1]))

            if self.args.save:
                plt.clf()
                torch.save(model_se, self.final_model_path)
                tqdm.write(f"save model(final): {self.final_model_path}")
                np.save(os.path.join(self.data_path, "train_metric.npy"), metric.items())
                fig = plot_metric({"train_loss": metric.train_loss, "valid_loss": metric.valid_loss},
                                  title="train and valid loss", result_path=self.image_path)
                self.writer.add_figure("learn loss", fig)
                self.writer.add_text("beat valid loss", f"{metric.best_valid_loss}")
                self.writer.add_text("duration", "{:2f}".format((end_time - start_time) / 60.))
            return model_se
        except KeyboardInterrupt as e:
            tqdm.write("正在退出")
            # 训练结束时需要进行的工作
            end_time = time.time()
            tqdm.write('Train ran for %.2f minutes' % ((end_time - start_time) / 60.))
            self.logging.info(self.args.model_name + "\t{:.2f}".format((end_time - start_time) / 60.))
            self.logging.info("train loss: {:.4f}, valid loss: {:.4f}"
                              .format(metric.train_loss[-1], metric.valid_loss[-1]))
            if self.args.save:
                plt.clf()
                torch.save(model_se, self.final_model_path)
                tqdm.write(f"save model(final): {self.final_model_path}")
                np.save(os.path.join(self.data_path, "train_metric.npy"), metric.items())
                fig = plot_metric({"train_loss": metric.train_loss, "valid_loss": metric.valid_loss},
                                  title="train and valid loss", result_path=self.image_path)
                self.writer.add_figure("learn loss", fig)
                self.writer.add_text("beat valid loss", f"{metric.best_valid_loss}")
                self.writer.add_text("duration", "{:2f}".format((end_time - start_time) / 60.))
            return model_se

    def test_step(self, model: nn.Module, model_qn: nn.Module, test_loader, test_num, q_len):
        """
        Args:
            model: 模型
            test_loader: 测试集 loader
            test_num: 测试集样本数
        """

        model = model.to(device=device)
        model_qn = model_qn.to(device=device)
        model_qn.eval()
        model.eval()
        self.freeze_parameters(model_qn)
        metric = Metric(mode="test")
        test_step = len(test_loader)
        calSigmos = CalSigmos(fs=48000, batch=True)

        loss_fn, loss_fn2 = self.get_loss_fn()

        test_loss = 0
        metric.mos_48k = {}
        for i in range(7):
            metric.mos_48k[metric.mos_48k_name[i]] = []

        idx = 0
        with torch.no_grad():
            for batch_idx, (x, xp, y, yp) in tqdm(enumerate(test_loader)):
                loss, est_x = self.predict(model, model_qn, x, y, loss_fn, loss_fn2)
                test_loss += loss
                batch = x.shape[0]
                if idx < q_len:
                    est_wav = spec2wav(x, xp, est_x, self.fft_size, self.hop_size, self.fft_size,
                                       input_type=self.se_input_type, mask_target=self.mask_target)

                    results = calSigmos(est_wav.cpu().detach().numpy())
                    for i in range(7):
                        metric.mos_48k[metric.mos_48k_name[i]].extend(results[i])
                idx += batch

        metric.test_loss = test_loss / test_step
        print("Test loss: {:.4f}".format(metric.test_loss))

        mean_mos_str = ""
        for i in range(7):
            name = metric.mos_48k_name[i]
            mean_mos_str += name
            mean_mos_str += ":{:.4f}\t".format(np.mean(metric.mos_48k[name]))
        print(mean_mos_str)

        self.logging.info("test loss: {:.4f}".format(metric.test_loss))

        if self.args.save:
            self.writer.add_text("test metric", str(metric))

            for i in range(7):
                name = metric.mos_48k_name[i]
                fig = plot_quantity(metric.mos_48k[name], f"测试语音的{name}", ylabel=name, result_path=self.image_path)
                self.writer.add_figure(name, fig)
            with open(self.image_path + "/mos.txt", "w", encoding="utf-8") as f:
                f.write(mean_mos_str)
            np.save(self.data_path + "test_metric.npy", metric.items())

    def test(self, test_dataset, model: nn.Module = None, model_qn: nn.Module = None, model_path: str = None,
             q_len=500):

        test_loader = dataloader.DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
        )

        test_num = len(test_dataset)
        start_time = time.time()
        if model is not None:
            self.test_step(model, model_qn, test_loader, test_num, q_len)
        elif model_path is None:
            model_path = self.final_model_path
            assert os.path.exists(model_path)
            print(f"load model: {model_path}")
            model = torch.load(model_path)
            self.test_step(model, model_qn, test_loader, test_num, q_len)
        elif model_path is not None:
            assert os.path.exists(model_path)
            print(f"load model: {model_path}")
            model = torch.load(model_path)
            self.test_step(model, model_qn, test_loader, test_num, q_len)
        else:
            raise "model_path and model can not be none simultaneously"

        end_time = time.time()
        print('Test ran for %.2f minutes' % ((end_time - start_time) / 60.))


if __name__ == "__main__":
    path_se = r"models\dpcrn_se20240518_224558\final.pt"
    path_qn = r"models\hasa20240522_223914\final.pt"
    # arg = Args("dpcrn_qse", model_name="dpcrn_se20240518_224558", model2_type="cnn")
    arg = Args("lstm", task_type="_joint", model2_type="hasa")
    arg.epochs = 25
    arg.batch_size = 8
    arg.save = True
    arg.lr = 2e-4
    arg.step_size = 5
    arg.delta_loss = 2e-4

    if arg.model2_type is None:
        raise ValueError("model qn type cannot be none")
    
    arg.se_input_type = 1

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
    train_dataset, valid_dataset, test_dataset = load_dataset_se("wav_train_se.list", arg.spilt_rate,
                                                                 arg.fft_size, arg.hop_size, arg.se_input_type)

    # model_se = load_pretrained_model(path_se)
    model_se = load_se_model(arg)
    model_qn = load_pretrained_model(path_qn)

    trainer = TrainerSEJ(arg)

    model_se = trainer.train(model_se, model_qn, train_dataset=train_dataset, valid_dataset=valid_dataset)
    trainer.test(test_dataset=test_dataset, model=model_se, model_qn=model_qn, q_len=200)
