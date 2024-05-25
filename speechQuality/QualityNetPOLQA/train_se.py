import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from tqdm import tqdm
from trainer_base import TrainerBase
from trainer_utils import Args, EarlyStopping, Metric, load_pretrained_model, plot_metric, plot_quantity, load_dataset_se, \
    load_se_model
from utils import spec2wav, CalSigmos, seed_everything, apply_mask_target

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TrainerSE(TrainerBase):
    """
    训练语音增强模型
    """

    def __init__(self, args: Args):
        if "_se" not in args.model_type:
            raise ValueError("Model type must end with '_se'")
        super().__init__(args)

    @staticmethod
    def get_loss_fn():
        loss_fn = nn.MSELoss()
        loss_fn.to(device=device)
        return loss_fn

    def train_epoch(self, model, x, y, loss_fn, optimizer):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = apply_mask_target(x, y, y_pred, loss_fn, self.mask_target, self.se_input_type)
        # loss = loss_fn(y_pred * torch.pow(x, 2), torch.pow(y, 2))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(self, model, x, y, loss_fn):
        """
        Return loss, y_pred
        """
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = apply_mask_target(x, y, y_pred, loss_fn, self.mask_target, self.se_input_type)
        return loss.item(), y_pred.cpu().detach()

    def train(self, model: nn.Module, train_dataset, valid_dataset):
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
        if self.args.load_weight:
            # 修改
            optimizer = self.get_optimizer(model.parameters(), self.lr)
        else:
            optimizer = self.get_optimizer(model.parameters(), self.lr)

        early_stop = EarlyStopping(patience=self.args.patience, delta_loss=self.args.delta_loss)

        scheduler = self.get_scheduler(optimizer, arg=self.args)

        # 设置损失函数和模型
        loss_fn = self.get_loss_fn()
        model = model.to(device)

        # 保存一些信息
        if self.args.save:
            self.writer.add_text("模型名", self.args.model_name)
            self.writer.add_text('超参数', str(self.args))
            try:
                if "dpcrn" in self.args.model_type:
                    dummy_input = torch.rand(4, 2, 128, self.args.fft_size // 2 + 1).to(device)
                else:
                    dummy_input = torch.rand(4, 128, self.args.fft_size // 2 + 1).to(device)
                if self.args.model_type == "lstmA":
                    pass
                else:
                    self.writer.add_graph(model, dummy_input)
            except RuntimeError as e:
                print(e)

        plt.ion()
        start_time = time.time()

        # 训练开始
        try:
            for epoch in tqdm(range(self.epochs), ncols=100):
                train_loss = 0
                valid_loss = 0
                model.train()
                loop_train = tqdm(enumerate(train_loader), leave=False)
                for batch_idx, (x, _, y, _) in loop_train:
                    loss = self.train_epoch(model, x, y, loss_fn, optimizer)
                    train_loss += loss
                    loop_train.set_description_str(f'Training [{epoch + 1}/{self.epochs}]')
                    loop_train.set_postfix_str("step: {}/{} loss: {:.4f}".format(batch_idx, train_step, loss))

                model.eval()
                with torch.no_grad():
                    loop_valid = tqdm(enumerate(valid_loader), leave=False)
                    for batch_idx, (x, _, y, _) in loop_valid:
                        loss, _ = self.predict(model, x, y, loss_fn)
                        valid_loss += loss
                        loop_valid.set_description_str(f'Validating [{epoch + 1}/{self.epochs}]')
                        loop_valid.set_postfix_str("step: {}/{} loss: {:.4f}".format(batch_idx, valid_step, loss))

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
                        torch.save(model, self.model_path + f"{epoch}.pt")
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
                        torch.save(model, self.best_model_path)
                        tqdm.write(f"saving model to {self.best_model_path}")
                else:
                    tqdm.write(f"validation loss did not decrease from {best_valid_loss}")

                if early_stop(metric.valid_loss[-1]):
                    if self.args.save:
                        torch.save(model, self.final_model_path)
                        tqdm.write(f"early stop..., saving model to {self.final_model_path}")
                    break
            # 训练结束时需要进行的工作
            end_time = time.time()
            tqdm.write('Train ran for %.2f minutes' % ((end_time - start_time) / 60.))
            self.logging.info(self.args.model_name + "\t{:.2f}".format((end_time - start_time) / 60.))
            self.logging.info(
                "train loss: {:.4f}, valid loss: {:.4f}".format(metric.train_loss[-1], metric.valid_loss[-1]))
            if self.args.save:
                plt.clf()
                torch.save(model, self.final_model_path)
                tqdm.write(f"save model(final): {self.final_model_path}")
                np.save(os.path.join(self.data_path, "train_metric.npy"), metric.items())
                fig = plot_metric({"train_loss": metric.train_loss, "valid_loss": metric.valid_loss},
                                  title="train and valid loss", result_path=self.image_path)
                self.writer.add_figure("learn loss", fig)
                self.writer.add_text("beat valid loss", f"{metric.best_valid_loss}")
                self.writer.add_text("duration", "{:2f}".format((end_time - start_time) / 60.))
            return model
        except KeyboardInterrupt as e:
            tqdm.write("正在退出")
            # 训练结束时需要进行的工作
            end_time = time.time()
            tqdm.write('Train ran for %.2f minutes' % ((end_time - start_time) / 60.))
            self.logging.info(self.args.model_name + "\t{:.2f}".format((end_time - start_time) / 60.))
            self.logging.info(
                "train loss: {:.4f}, valid loss: {:.4f}".format(metric.train_loss[-1], metric.valid_loss[-1]))
            if self.args.save:
                plt.clf()
                torch.save(model, self.final_model_path)
                tqdm.write(f"save model(final): {self.final_model_path}")
                np.save(os.path.join(self.data_path, "train_metric.npy"), metric.items())
                fig = plot_metric({"train_loss": metric.train_loss, "valid_loss": metric.valid_loss},
                                  title="train and valid loss", result_path=self.image_path)
                self.writer.add_figure("learn loss", fig)
                self.writer.add_text("beat valid loss", f"{metric.best_valid_loss}")
                self.writer.add_text("duration", "{:2f}".format((end_time - start_time) / 60.))
            return model

    def test_step(self, model: nn.Module, test_loader, test_num, q_len):
        """
        Args:
            model: 模型
            test_loader: 测试集 loader
            test_num: 测试集样本数
        """

        model = model.to(device=device)
        model.eval()
        metric = Metric(mode="test")
        test_step = len(test_loader)

        calSigmos = CalSigmos(fs=48000, batch=True)

        loss_fn = self.get_loss_fn()
        test_loss = 0
        metric.mos_48k = {}
        for i in range(7):
            metric.mos_48k[metric.mos_48k_name[i]] = []
        idx = 0
        with torch.no_grad():
            for batch_idx, (x, xp, y, yp) in tqdm(enumerate(test_loader)):
                batch = x.shape[0]
                loss, est_x = self.predict(model, x, y, loss_fn)
                test_loss += loss
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
        else:
            pass

    def test(self, test_dataset, model: nn.Module = None, model_path: str = None, q_len=500):

        test_loader = dataloader.DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
        )

        test_num = len(test_dataset)
        start_time = time.time()
        if model is not None:
            self.test_step(model, test_loader, test_num, q_len)
        elif model_path is None:
            model_path = self.final_model_path
            assert os.path.exists(model_path)
            print(f"load model: {model_path}")
            model = torch.load(model_path)
            self.test_step(model, test_loader, test_num, q_len)
        elif model_path is not None:
            assert os.path.exists(model_path)
            print(f"load model: {model_path}")
            model = torch.load(model_path)
            self.test_step(model, test_loader, test_num, q_len)
        else:
            raise "model_path and model can not be none simultaneously"

        end_time = time.time()
        print('Test ran for %.2f minutes' % ((end_time - start_time) / 60.))


if __name__ == "__main__":
    # arg = Args("lstm", task_type="_se", model_name="lstm_se_IAM20240522_174211", mask_target="IAM")
    arg = Args("lstm", task_type="_se", mask_target="IAM")
    arg.epochs = 35
    arg.batch_size = 32
    arg.save = True
    arg.lr = 4e-4
    arg.step_size = 5
    if arg.mask_target is None:
        arg.delta_loss = 2e-4

    arg.se_input_type = 1

    # arg.optimizer_type = 1

    if arg.save and not  arg.expire:
        arg.write(arg.model_name)
    print(arg)

    seed_everything(arg.random_seed)

    # 加载用于训练语音增强模型的数据集 x: (B, L, C)  y: (B L C)
    train_dataset, valid_dataset, test_dataset = load_dataset_se("wav_train_se.list", arg.spilt_rate,
                                                                 arg.fft_size, arg.hop_size, arg.se_input_type)

    model = load_se_model(arg)
    # model = load_pretrained_model(r"models\lstm_se_IAM20240522_174211\final.pt")

    trainer = TrainerSE(arg)
    model = trainer.train(model, train_dataset=train_dataset, valid_dataset=valid_dataset)
    trainer.test(test_dataset=test_dataset, model=model, q_len=200)

    # trainer.inference_step(model, r"D:\work\speechEnhancement\datasets\dns_to_liang\31435_nearend.wav",
    #                        r"D:\work\speechEnhancement\datasets\dns_to_liang\31435_target.wav")
