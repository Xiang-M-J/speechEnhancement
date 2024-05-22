import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from trainer_utils import Args, EarlyStopping, Metric, plot_metric, load_qn_model, load_dataset_qn
from losses import FrameMse, FrameMse2, FrameMseNo
from utils import norm_label, seed_everything, get_logging
from trainer_base import TrainerBase
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Trainer(TrainerBase):
    """
    训练
    """

    def __init__(self, args: Args):
        super().__init__(args)

    def get_loss_fn(self):
        loss1 = nn.MSELoss()
        if "hasa" in self.args.model_type:
            loss2 = FrameMse2(self.args.enable_frame)
        else:
            if self.args.normalize_output:
                loss2 = FrameMseNo(self.args.enable_frame)
            else:
                loss2 = FrameMse(self.args.enable_frame)
        loss1.to(device=device)
        loss2.to(device=device)
        return loss1, loss2

    def train_epoch(self, model, norm, x, y, loss1, loss2, optimizer):
        y1 = y[0]
        y2 = y[1]
        if self.args.normalize_output:
            y1 = norm_label(y1)
            y2 = norm_label(y2)
        if "cnn" in self.args.model_type or "hubert" in self.args.model_type:
            avgS = model(x)
            if self.args.normalize_output:
                avgS = norm(avgS)
            loss = loss1(avgS.squeeze(-1), y1)
        else:
            frameS, avgS = model(x)
            if self.args.normalize_output:
                avgS = norm(avgS)
                frameS = norm(frameS)
            l1 = loss1(avgS.squeeze(-1), y1)
            l2 = loss2(frameS, y2)
            loss = l1 + l2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(self, model, norm, x, y, loss1, loss2):
        """
        Return loss, predict score, true score
        """
        y1 = y[0]
        y2 = y[1]
        if self.args.normalize_output:
            y1 = norm_label(y1)
            y2 = norm_label(y2)
        if "cnn" in self.args.model_type or "hubert" in self.args.model_type:
            avgS = model(x)
            if self.args.normalize_output:
                avgS = norm(avgS)
            loss = loss1(avgS.squeeze(-1), y1)
        else:
            frameS, avgS = model(x)
            if self.args.normalize_output:
                avgS = norm(avgS)
                frameS = norm(frameS)
            l1 = loss1(avgS.squeeze(-1), y1)
            l2 = loss2(frameS, y2)
            loss = l1 + l2
        return loss.item(), avgS.squeeze(-1).cpu().detach().numpy(), y1.cpu().detach().numpy()

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

        # 保存一些信息
        if self.args.save:
            self.writer.add_text("模型名", self.args.model_name)
            self.writer.add_text('超参数', str(self.args))
            try:
                if "hubert" in self.args.model_type:
                    dummy_input = torch.rand(4, 1, 48000)
                else:
                    dummy_input = torch.rand(4, 128, self.args.fft_size // 2 + 1)
                if self.args.model_type == "lstmA" or "hubert" in self.args.model_type:
                    # mask = torch.randn([512, 512]).to(device)
                    # self.writer.add_graph(model, dummy_input)
                    pass
                else:
                    self.writer.add_graph(model, dummy_input)
            except RuntimeError as e:
                print(e)

        # 设置损失函数和模型
        loss1, loss2 = self.get_loss_fn()
        model = model.to(device)

        norm = nn.Sigmoid().to(device)

        plt.ion()
        start_time = time.time()

        # 训练开始
        try:
            for epoch in tqdm(range(self.epochs), ncols=100):
                train_loss = 0
                valid_loss = 0
                model.train()
                loop_train = tqdm(enumerate(train_loader), leave=False)
                for batch_idx, (x, y) in loop_train:
                    loss = self.train_epoch(model, norm, x, y, loss1, loss2, optimizer)
                    train_loss += loss

                    loop_train.set_description_str(f'Training [{epoch + 1}/{self.epochs}]')
                    loop_train.set_postfix_str("step: {}/{} loss: {:.4f}".format(batch_idx, train_step, loss))

                model.eval()
                with torch.no_grad():
                    loop_valid = tqdm(enumerate(valid_loader), leave=False)
                    for batch_idx, (x, y) in loop_valid:
                        loss, _, _ = self.predict(model, norm, x, y, loss1, loss2)
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
                        if "hubert" in self.args.model_type:
                            torch.save(model.state_dict(), self.model_path + f"{epoch}.pt")
                        else:
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
                        if "hubert" in self.args.model_type:
                            torch.save(model.state_dict(), self.best_model_path)
                        else:
                            torch.save(model, self.best_model_path)
                        tqdm.write(f"saving model to {self.best_model_path}")
                else:
                    tqdm.write(f"validation loss did not decrease from {best_valid_loss}")

                if early_stop(metric.valid_loss[-1]):
                    if self.args.save:
                        if "hubert" in self.args.model_type:
                            torch.save(model.state_dict(), self.final_model_path)
                        else:
                            torch.save(model, self.final_model_path)
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
                if "hubert" in self.args.model_type:
                    torch.save(model.state_dict(), self.final_model_path)
                else:
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
            self.logging.info("train loss: {:.4f}, valid loss: {:.4f}"
                              .format(metric.train_loss[-1], metric.valid_loss[-1]))
            if self.args.save:
                plt.clf()
                if "hubert" in self.args.model_type:
                    torch.save(model.state_dict(), self.final_model_path)
                else:
                    torch.save(model, self.final_model_path)
                tqdm.write(f"save model(final): {self.final_model_path}")
                np.save(os.path.join(self.data_path, "train_metric.npy"), metric.items())
                fig = plot_metric({"train_loss": metric.train_loss, "valid_loss": metric.valid_loss},
                                  title="train and valid loss", result_path=self.image_path)
                self.writer.add_figure("learn loss", fig)
                self.writer.add_text("beat valid loss", f"{metric.best_valid_loss}")
                self.writer.add_text("duration", "{:2f}".format((end_time - start_time) / 60.))
            return model

    def test_step(self, model: nn.Module, test_loader, test_num, q_len=200):
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

        POLQA_Predict = np.zeros([test_num, ])
        POLQA_True = np.zeros([test_num, ])
        idx = 0
        loss1, loss2 = self.get_loss_fn()
        test_loss = 0
        norm = nn.Sigmoid().to(device)
        with torch.no_grad():
            for batch_idx, (x, y) in tqdm(enumerate(test_loader)):
                batch = x.shape[0]
                loss, est_polqa, true_polqa = self.predict(model, norm, x, y, loss1, loss2)
                test_loss += loss
                POLQA_Predict[idx: idx + batch] = est_polqa
                POLQA_True[idx: idx + batch] = true_polqa
                idx += batch

        metric.test_loss = test_loss / test_step
        print("Test loss: {:.4f}".format(metric.test_loss))
        metric.mse = np.mean((POLQA_True - POLQA_Predict) ** 2)
        print('Test error= %f' % metric.mse)
        metric.lcc = np.corrcoef(POLQA_True, POLQA_Predict)
        print('Linear correlation coefficient= %f' % float(metric.lcc[0][1]))

        metric.srcc = scipy.stats.spearmanr(POLQA_True.T, POLQA_Predict.T)
        print('Spearman rank correlation coefficient= %f' % metric.srcc[0])
        self.logging.info("test loss: {:.4f}, mse: {:.4f},  lcc: {:.4f}, srcc: {:.4f}"
                          .format(metric.test_loss, metric.mse, float(metric.lcc[0][1]), metric.srcc[0]))

        if self.args.save:
            M = np.max([np.max(POLQA_Predict), 5])
            plt.clf()
            fig = plt.figure(1)
            plt.scatter(POLQA_True, POLQA_Predict, s=3)
            plt.xlim([0, M])
            plt.ylim([0, M])
            plt.xlabel('True PESQ')
            plt.ylabel('Predicted PESQ')
            plt.title('LCC= %f, SRCC= %f, MSE= %f' % (float(metric.lcc[0][1]), metric.srcc[0], metric.mse))
            plt.savefig(self.image_path + 'Scatter_plot.png', dpi=300)
            self.writer.add_text("test metric", str(metric))
            self.writer.add_figure("predict score", fig)
            np.save(self.data_path + "test_metric.npy", metric.items())
        else:
            plt.scatter(POLQA_True, POLQA_Predict, s=6)
            plt.show()
            plt.pause(2)
            plt.ioff()

    def test(self, test_dataset, model: nn.Module = None, model_path: str = None):

        test_loader = dataloader.DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
        )

        test_num = len(test_dataset)
        start_time = time.time()
        if model is not None:
            self.test_step(model, test_loader, test_num)
        elif model_path is None:
            model_path = self.final_model_path
            assert os.path.exists(model_path)
            print(f"load model: {model_path}")
            model = torch.load(model_path)
            self.test_step(model, test_loader, test_num)
        elif model_path is not None:
            assert os.path.exists(model_path)
            print(f"load model: {model_path}")
            model = torch.load(model_path)
            self.test_step(model, test_loader, test_num)
        else:
            raise "model_path and model can not be none simultaneously"

        end_time = time.time()
        print('Test ran for %.2f minutes' % ((end_time - start_time) / 60.))


if __name__ == "__main__":
    arg = Args("hasa")
    arg.epochs = 35
    arg.batch_size = 12
    arg.save = True
    arg.lr = 5e-4
    arg.step_size = 10
    arg.delta_loss = 1e-3

    # 用于 qualityNet
    arg.normalize_output = True

    # 训练Hubert
    # arg.optimizer_type = 1
    # arg.enable_frame = False

    # 训练 CNN / tcn
    # arg.optimizer_type = 1
    # arg.enableFrame = False

    print(arg)
    if arg.save:
        arg.write(arg.model_name)

    seed_everything(arg.random_seed)

    # 加载用于预测polqa分数的数据集 x: (B, L, C), y1: (B,), y2: (B, L)
    train_dataset, valid_dataset, test_dataset = load_dataset_qn("wav_train_qn.list", arg.spilt_rate,
                                                                 arg.fft_size, arg.hop_size, return_wav=False)

    model = load_qn_model(arg)

    trainer = Trainer(arg)
    model = trainer.train(model, train_dataset=train_dataset, valid_dataset=valid_dataset)
    trainer.test(test_dataset=test_dataset, model=model)
