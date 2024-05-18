import os
import time
import abc

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from trainer_utils import Args, EarlyStopping, Metric, plot_metric
from utils import FrameMse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TrainerBase(abc.ABC):
    """
    训练
    """

    def __init__(self, args: Args):
        self.args: Args = args
        self.optimizer_type = args.optimizer_type
        self.model_path = f"models/{args.model_name}/"
        self.best_model_path = self.model_path + "best.pt"  # 模型保存路径(max val acc)
        self.final_model_path = self.model_path + "final.pt"  # 模型保存路径(final)
        self.result_path = f"results/{args.model_name}/"  # 结果保存路径（分为数据和图片）
        self.image_path = self.result_path + "images/"
        self.data_path = self.result_path + "data/"
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.save_model_epoch = args.save_model_epoch
        self.lr = args.lr
        self.test_acc = []
        if args.save:
            self.check_dir()
            self.writer = SummaryWriter("runs/"+self.args.model_name)

    def check_dir(self):
        """
        创建文件夹
        """

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    def get_optimizer(self, parameter, lr):
        if self.optimizer_type == 0:
            optimizer = torch.optim.SGD(params=parameter, lr=lr, weight_decay=self.args.weight_decay)
            # 对于SGD而言，L2正则化与weight_decay等价
        elif self.optimizer_type == 1:
            optimizer = torch.optim.Adam(params=parameter, lr=lr, betas=(self.args.beta1, self.args.beta2),
                                         weight_decay=self.args.weight_decay)
            # 对于Adam而言，L2正则化与weight_decay不等价
        elif self.optimizer_type == 2:
            optimizer = torch.optim.AdamW(params=parameter, lr=lr, betas=(self.args.beta1, self.args.beta2),
                                          weight_decay=self.args.weight_decay)
        elif self.optimizer_type == 3:
            optimizer = torch.optim.RMSprop(parameter, lr=lr, weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError
        return optimizer

    def get_scheduler(self, optimizer, arg: Args):
        if arg.scheduler_type == 0:
            return None
        elif arg.scheduler_type == 1:
            return torch.optim.lr_scheduler.StepLR(optimizer, arg.step_size, arg.gamma)
        elif arg.scheduler_type == 2:
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=arg.epochs, eta_min=1e-6)
        else:
            raise NotImplementedError

    @abc.abstractclassmethod
    def get_loss_fn(self):
        pass
    
    @abc.abstractclassmethod
    def train_step(self, model, x, y, loss, optimizer):
        pass


    @abc.abstractclassmethod
    def predict(self, model, x, y, loss1, loss2):
        """
        Return loss, predict score, true score
        """
        pass

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
        loss1, loss2 = self.get_loss_fn()
        model = model.to(device)

        # 保存一些信息
        if self.args.save:
            self.writer.add_text("模型名", self.args.model_name)
            self.writer.add_text('超参数', str(self.args))
            try:
                dummy_input = torch.rand(self.args.batch_size, 128, self.args.fft_size // 2 + 1).to(device)
                if self.args.model_type == "lstmA":
                    # mask = torch.randn([512, 512]).to(device)
                    # self.writer.add_graph(model, dummy_input)
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
                for batch_idx, (x, y) in loop_train:
                    loss = self.train_step(model, x, y, loss1, loss2, optimizer)
                    train_loss += loss

                    loop_train.set_description_str(f'Training [{epoch + 1}/{self.epochs}]')
                    loop_train.set_postfix_str("step: {}/{} loss: {:.4f}".format(batch_idx, train_step, loss))

                model.eval()
                with torch.no_grad():
                    loop_valid = tqdm(enumerate(valid_loader), leave=False)
                    for batch_idx, (x, y) in loop_valid:
                        loss, _, _ = self.predict(model, x, y, loss1, loss2)
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
            with open("log.txt", mode='a', encoding="utf-8") as f:
                f.write(
                    self.args.model_name + f"\t{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\t" + "{:.2f}".format(
                        (end_time - start_time) / 60.) + "\n")
                f.write(
                    "train loss: {:.4f}, valid loss: {:.4f} \n".format(metric.train_loss[-1], metric.valid_loss[-1]))
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
            with open("log.txt", mode='a', encoding="utf-8") as f:
                f.write(
                    self.args.model_name + f"\t{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\t" + "{:.2f}".format(
                        (end_time - start_time) / 60.) + "\n")
                f.write("train loss: {:.4f}, valid loss: {:.4f}\n".format(metric.train_loss[-1], metric.valid_loss[-1]))
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

    def test_step(self, model: nn.Module, test_loader, test_num):
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
        with torch.no_grad():
            for batch_idx, (x, y) in tqdm(enumerate(test_loader)):
                batch = x.shape[0]
                loss, est_polqa, true_polqa = self.predict(model, x, y, loss1, loss2)
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
        with open("log.txt", mode='a', encoding="utf-8") as f:
            f.write("test loss: {:.4f}, mse: {:.4f},  lcc: {:.4f}, srcc: {:.4f} \n"
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
