import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tensorboardX import SummaryWriter
from torch.utils.data import dataloader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

import yaml


class Args:
    def __init__(self,
                 epochs=10,
                 lr=1e-3,
                 model_name = "",
                 batch_size=64,
                 spilt_rate=None,
                 weight_decay=0.2,
                 patience=10,
                 delta_loss = 1e-4,
                 optimizer_type=2,
                 beta1=0.99,
                 beta2=0.999,
                 random_seed=34,
                 model_type="MTCN",
                 save=True,
                 scheduler_type=0,
                 gamma=0.3,
                 step_size=10,
                 dropout=0.1,
                 load_weight=False,
                 ):
        """
        Args:
            optimizer_type: 优化器种类(0: SGD, 1:Adam, 2:AdamW)
            beta1: adam优化器参数
            beta2: adam优化器参数
            random_seed: 随机数种子
            data_type: 数据类型
            save: 是否保存模型和结果
            scheduler_type: scheduler类型
            gamma: LR scheduler参数
            step_size: LR scheduler参数
            warmup: Warm scheduler参数
        """
        if spilt_rate is None:
            spilt_rate = [0.8, 0.1, 0.1]
        self.model_name = model_name + time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.spilt_rate = spilt_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        
        self.dropout = dropout
        self.random_seed = random_seed
        self.model_type = model_type
        self.save = save
        self.scheduler_type = scheduler_type
        self.load_weight = load_weight

        # 用于 Adam 优化器
        self.beta1 = beta1
        self.beta2 = beta2

        # 用于 step scheduler
        self.step_size = step_size
        self.gamma = gamma

        # 用于早停止
        self.patience = patience
        self.delta_loss = delta_loss
        

    def write(self, name="hyperparameter"):
        with open("./config/" + name + ".yml", 'w', encoding='utf-8') as f:
            yaml.dump(data=self.__dict__, stream=f, allow_unicode=True)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            hyperparameter = yaml.load(f.read(), Loader=yaml.FullLoader)
        for para in hyperparameter:
            self.__setattr__(para, hyperparameter[para])

    def items(self):
        return self.__dict__

    def string(self) -> str:
        info = "parameter setting:\t"
        for parameter in self.__dict__:
            info += f"{parameter}: {self.__dict__[parameter]}\t"
        info += '\n'
        return info

class Metric:
    """
    存储模型训练和测试时的指标
    """

    def __init__(self, mode="train"):
        if mode == "train":
            self.mode = "train"
            self.train_loss = []
            self.valid_loss = []
            self.best_valid_loss = 0
        elif mode == "test":
            self.mode = "test"
            self.test_loss = 0
        else:
            print("wrong mode !!! use default mode train")
            self.mode = "train"
            self.train_loss = []
            self.valid_loss = []
            self.best_valid_loss = 0

    def items(self) -> dict:
        """
        返回各种指标的字典格式数据
        Returns: dict

        """
        if self.mode == "train":
            data = {"train_loss": self.train_loss, "valid_loss": self.valid_loss, 'best_valid_loss': self.best_valid_loss}
        else:
            data = {"test_loss": self.test_loss}
        return data

def check_dir():
    """
    创建models, results/images/, results/data 文件夹
    """
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/images"):
        os.makedirs("results/images")
    if not os.path.exists("results/data"):
        os.makedirs("results/data")


class EarlyStopping:
    """Early stops the training if validation accuracy doesn't change after a given patience."""

    def __init__(self, patience=5, delta_loss=1e-4):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 5
        """
        self.patience = patience
        self.patience_ = patience
        self.delta_loss = delta_loss
        self.last_val_loss = 0

    def __call__(self, val_loss) -> bool:
        if abs(self.last_val_loss - val_loss) < self.delta_loss:
            self.patience -= 1
        else:
            self.patience = self.patience_
        self.last_val_loss = val_loss
        if self.patience == 1:
            print(f"The validation loss has not changed in {self.patience_} iterations, stop train")
            print(f"The final validation loss is {val_loss}")
            return True
        else:
            return False


class Trainer:
    """
    训练
    """
    def __init__(self, args: Args):
        self.args: Args = args
        self.optimizer_type = args.optimizer_type
        self.model_type = args.model_type
        self.best_path = f"models/" + args.model_name + "_best" + ".pt"  # 模型保存路径(max val acc)
        self.final_path = f"models/" + args.model_name + ".pt"  # 模型保存路径(final)
        self.result_path = f"results/"  # 结果保存路径（分为数据和图片）
        self.save_path = f"models/" + args.model_name
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.test_acc = []
        check_dir()
        if args.save:
            self.writer = SummaryWriter("runs/" + time.strftime('%Y%m%d_%H%M%S', time.localtime()))

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

    def train(self, model: nn.Module, loss_fn: nn.Module, train_dataset, valid_dataset, test_dataset):
        
        if self.args.save:
            self.writer.add_text("模型名", self.args.model_name)
            self.writer.add_text('超参数', self.args.string())
        metric = Metric()
        train_loader = dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
        )
        valid_loader = dataloader.DataLoader(
            dataset=valid_dataset,
            batch_size=self.batch_size,
        )
        train_num = len(train_dataset)
        valid_num = len(valid_dataset)

        if self.args.load_weight:
            # 修改
            optimizer = self.get_optimizer(model.parameters(), self.lr)
            
        else:
            optimizer = self.get_optimizer(model.parameters(), self.lr)

        early_stop = EarlyStopping(patience=5, delta_loss=2e-4)

        scheduler = self.get_scheduler(optimizer, arg=self.args)

        best_val_accuracy = 0
        model = model.to(device)
        steps = 0  # 用于warmup
        plt.ion()
        for epoch in range(self.epochs):
            model.train()
            train_loss = 0
            val_loss = 0
            for step, (bx, by) in enumerate(train_loader):
                bx, by = bx.to(device), by.to(device)
                output = model(bx)
                loss = loss_fn(output, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.data.item()
                steps += 1

            model.eval()
            with torch.no_grad():
                for step, (vx, vy) in enumerate(valid_loader):
                    vx, vy = vx.to(device), vy.to(device)
                    output = model(vx)
                    loss = loss_fn(output, vy)
                    val_loss += loss.data.item()

            if self.args.scheduler_type == 0:
                pass
            elif self.args.scheduler_type == 2:
                scheduler.step(steps)
            else:
                scheduler.step()

            metric.train_loss.append(train_loss / math.ceil((train_num / self.batch_size)))
            metric.valid_loss.append(val_loss / math.ceil(valid_num / self.batch_size))

            plt.clf()
            plt.plot(metric.train_acc)
            plt.plot(metric.val_acc)
            plt.ylabel("accuracy(%)")
            plt.xlabel("epoch")
            plt.legend(["train acc", "val acc"])
            plt.title("train accuracy and validation accuracy")
            plt.pause(0.02)
            plt.ioff()  # 关闭画图的窗口

            if self.args.save:
                self.writer.add_scalar('train loss', metric.train_loss[-1], epoch + 1)
                self.writer.add_scalar('validation loss', metric.valid_loss[-1], epoch + 1)

            print(
                'Epoch :{}\t train Loss:{:.4f}\t train Accuracy:{:.3f}\t val Loss:{:.4f} \t val Accuracy:{:.3f}'.format(
                    epoch + 1, metric.train_loss[-1], metric.train_acc[-1], metric.valid_loss[-1],
                    metric.val_acc[-1]))
            print(optimizer.param_groups[0]['lr'])
            if metric.val_acc[-1] > best_val_accuracy:
                print(f"val_accuracy improved from {best_val_accuracy :.3f} to {metric.val_acc[-1]:.3f}")
                best_val_accuracy = metric.val_acc[-1]
                metric.best_val_acc[0] = best_val_accuracy
                metric.best_val_acc[1] = metric.train_acc[-1]
                if self.args.save:
                    torch.save(model, self.best_path)
                    print(f"saving model to {self.best_path}")
            elif metric.val_acc[-1] == best_val_accuracy:
                if metric.train_acc[-1] > metric.best_val_acc[1]:
                    metric.best_val_acc[1] = metric.train_acc[-1]
                    if self.args.save:
                        torch.save(model, self.best_path)
            else:
                print(f"val_accuracy did not improve from {best_val_accuracy}")
            
            if early_stop(metric.val_loss[-1]):
                break
            if metric.val_acc[-1] > 99.4:
                self.test_acc.append(self.multi_test_step(model, test_dataset=test_dataset))

            # model_save(model, epoch)

        if self.args.save:
            torch.save(model, self.final_path)
            print(f"save model(last): {self.final_path}")
            plot(metric.item(), self.args.model_name, self.result_path)
            np.save(self.result_path + "data/" + self.args.model_name + "_train_metric", metric.item())
            self.writer.add_text("beat validation accuracy", f"{metric.best_val_acc}")
            self.writer.add_text("parameter setting", self.args.addition())
            self.writer.add_text("model name", model.name)

            dummy_input = torch.rand(self.args.batch_size, self.args.feature_dim, self.args.seq_len).to(device)
            mask = torch.rand(self.args.seq_len, self.args.seq_len).to(device)
            if self.model_type in ["TIM", "LSTM", "TCN"]:
                self.writer.add_graph(model, dummy_input)
            else:
                self.writer.add_graph(model, [dummy_input, mask])
            self.logger.train(train_metric=metric)

    def test_step(self, model_path, test_loader, loss_fn, test_num, metric, best=False):
        """
        Args:
            model_path: 模型路径

        Returns:
            metric fig
        """
        if not os.path.exists(model_path):
            print(f"error! cannot find the model in {model_path}")
            return
        print(f"load model: {model_path}")
        model = torch.load(model_path)
        model.eval()
        test_correct = 0
        test_loss = 0
        y_pred = torch.zeros(test_num)
        y_true = torch.zeros(test_num)
        for step, (vx, vy) in enumerate(test_loader):
            vx, vy = vx.to(device), vy.to(device)
            with torch.no_grad():
                output = model(vx)
                y_pred[step * self.batch_size: step * self.batch_size + vy.shape[0]] = torch.max(output.data, 1)[1]
                y_true[step * self.batch_size: step * self.batch_size + vy.shape[0]] = torch.max(vy.data, 1)[1]
                loss = loss_fn(output, vy)
                test_loss += loss.data.item()
        conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(self.args.num_class))

        
        test_acc = float((test_correct * 100) / test_num)
        test_loss = test_loss / math.ceil(test_num / self.batch_size)
        metric.confusion_matrix.append(conf_matrix)
        metric.test_loss.append(test_loss)
        
        return metric

    def test(self, test_dataset, model_path: str = None):
        metric = Metric(mode="test")
        metric.test_loss = 0
        test_loader = dataloader.DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
        )
        
        test_num = len(test_dataset)
        if model_path is None:
            model_path = self.final_path
            
            self.test_step(model_path, test_loader, loss_fn, test_num, metric)
            
            print("{} final test Loss:{:.4f} test Accuracy:{:.3f}".format(
                self.args.model_name, metric.test_loss[0], metric.test_acc[0]))

            if self.args.save:
                self.writer.add_text("test loss(final)", str(metric.test_loss[0]))
                self.writer.add_text("classification report(final)", metric.report[0])
                self.writer.add_text("test loss(best)", str(metric.test_loss[1]))
                self.writer.add_text("classification report(best)", metric.report[1])
                self.logger.test(test_metric=metric)
                np.save(self.result_path + "data/" + self.args.model_name + "_test_metric.npy", metric.item())
        else:
            metric = self.test_step(model_path, test_loader, loss_fn, test_num, metric)
            print("{} test Loss:{:.4f} test Accuracy:{:.3f}".format(
                self.args.model_name, metric.test_loss[0], metric.test_acc[0]))

            if self.args.save:
                self.writer.add_text("test loss(final)", str(metric.test_loss[0]))
                np.save(self.result_path + "data/" + self.args.model_name + "_test_metric.npy", metric.item())
