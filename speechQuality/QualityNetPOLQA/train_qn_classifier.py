import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from tqdm import tqdm
from trainer_base import TrainerBase

from losses import EDMLoss, AvgCrossEntropyLoss, NormMseLoss, FrameEDMLoss
from trainer_utils import Args, EarlyStopping, Metric, plot_metric, load_qn_model, load_dataset_qn, confuseMatrix, \
    plot_matrix, load_pretrained_model, log_model
from utils import accurate_num_cal, oneHotToFloat, seed_everything

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TrainerC(TrainerBase):
    """
    训练分类器
    """

    def __init__(self, args: Args):
        if "Class" not in args.model_type:
            raise TypeError("Error model type")
        super().__init__(args)

    def get_loss_fn(self):
        if self.args.normalize_output:
            loss1 = NormMseLoss()
        else:
            loss1 = nn.MSELoss()
        # loss1 = FrameEDMLoss(self.args.smooth, self.args.enable_frame, self.args.score_step)
        # loss2 = EDMLoss(self.args.score_step, self.args.smooth)
        loss2 = AvgCrossEntropyLoss(step=self.args.score_step)

        loss1.to(device=device)
        loss2.to(device=device)
        return [loss1, loss2]

    def train_epoch(self, model, x, y, loss_fns, optimizer):
        loss1 = loss_fns[0]
        loss2 = loss_fns[1]

        y1 = y[0]
        y2 = y[1]
        avg, c = model(x)
        l1 = loss1(avg.squeeze(-1), y1)
        l2 = loss2(c, y1)
        loss = l1 + l2
        # loss = l2
        loss.requires_grad_(True)
        accurate_num = accurate_num_cal(c, y1, self.args.score_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), accurate_num

    def predict(self, model, x, y, loss_fns):
        """
        Return loss, predict score, true score, accuracy num
        """
        loss1 = loss_fns[0]
        loss2 = loss_fns[1]
        y1 = y[0]
        y2 = y[1]
        avg, c = model(x)
        accurate_num = accurate_num_cal(c, y1, self.args.score_step)
        l1 = loss1(avg.squeeze(-1), y1)
        l2 = loss2(c, y1)
        loss = l1 + l2
        # loss = l2
        predict_cls = oneHotToFloat(c.cpu().detach().numpy(), self.args.score_step)
        predict_avg = avg.squeeze(-1).sigmoid().cpu().detach().numpy()
        if self.args.normalize_output:
            predict_avg = predict_avg * 4.0 + 1.0
        # return loss.item(), (predict_avg + predict_cls)/2, y1.cpu().detach().numpy(), accurate_num
        return loss.item(), predict_cls, y1.cpu().detach().numpy(), accurate_num

    def train(self, model: nn.Module, train_dataset, valid_dataset):
        print("begin train")
        # 设置一些参数
        best_valid_loss = 100.
        metric = Metric(with_acc=True)

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
        train_num = len(train_dataset)
        valid_num = len(valid_dataset)

        train_step = len(train_loader)
        valid_step = len(valid_loader)

        # 设置优化器、早停止和scheduler

        optimizer = self.get_optimizer(model.parameters(), self.lr)

        early_stop = EarlyStopping(patience=self.args.patience, delta_loss=self.args.delta_loss)

        scheduler = self.get_scheduler(optimizer, arg=self.args)

        # 设置损失函数和模型
        loss_fns = self.get_loss_fn()
        model = model.to(device)

        # 保存一些信息
        if self.args.save:
            self.writer.add_text("模型名", self.args.model_name)
            self.writer.add_text('超参数', str(self.args))
            info = log_model(model, self.image_path)
            try:
                dummy_input = torch.rand(self.args.batch_size, 256, self.args.fft_size // 2 + 1).to(device)
                self.writer.add_graph(model, dummy_input)
            except Exception as e:
                print("can not save graph")
                self.writer.add_text("model info", info)

        plt.ion()
        start_time = time.time()

        # 训练开始
        try:
            for epoch in tqdm(range(self.epochs), ncols=100):
                train_loss = 0
                valid_loss = 0
                train_acc_num = 0
                valid_acc_num = 0
                model.train()
                loop_train = tqdm(enumerate(train_loader), leave=False)
                for batch_idx, (x, y) in loop_train:
                    loss, num = self.train_epoch(model, x, y, loss_fns, optimizer)
                    train_loss += loss
                    train_acc_num += num
                    loop_train.set_description_str(f'Training [{epoch + 1}/{self.epochs}]')
                    loop_train.set_postfix_str("step: {}/{} loss: {:.4f}".format(batch_idx, train_step, loss))
                    # break

                model.eval()
                with torch.no_grad():
                    loop_valid = tqdm(enumerate(valid_loader), leave=False)
                    for batch_idx, (x, y) in loop_valid:
                        loss, _, _, num = self.predict(model, x, y, loss_fns)
                        valid_loss += loss
                        valid_acc_num += num
                        loop_valid.set_description_str(f'Validating [{epoch + 1}/{self.epochs}]')
                        loop_valid.set_postfix_str("step: {}/{} loss: {:.4f}".format(batch_idx, valid_step, loss))

                if self.args.scheduler_type != 0:
                    scheduler.step()

                # 保存每个epoch的训练信息
                metric.train_loss.append(train_loss / train_step)
                metric.valid_loss.append(valid_loss / valid_step)
                metric.train_acc.append(train_acc_num / train_num * 100)
                metric.valid_acc.append(valid_acc_num / valid_num * 100)

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
                    self.writer.add_scalar("train acc", metric.train_acc[-1], epoch + 1)
                    self.writer.add_scalar("valid acc", metric.valid_acc[-1], epoch + 1)
                    self.writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
                    np.save(os.path.join(self.data_path, "train_metric.npy"), metric.items())
                tqdm.write(
                    'Epoch {}:  train Loss:{:.4f}\t val Loss:{:.4f}\t train Acc:{:.4f}\t valid Acc:{:.4f}'.format(
                        epoch + 1, metric.train_loss[-1], metric.valid_loss[-1], metric.train_acc[-1],
                        metric.valid_acc[-1]))

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
            self.logging.info("train loss: {:.4f}, valid loss: {:.4f} train acc: {:.4f}, valid acc: {:.4f}"
                              .format(metric.train_loss[-1], metric.valid_loss[-1],
                                      metric.train_acc[-1], metric.valid_acc[-1]))
            if self.args.save:
                plt.clf()
                torch.save(model, self.final_model_path)
                tqdm.write(f"save model(final): {self.final_model_path}")
                np.save(os.path.join(self.data_path, "train_metric.npy"), metric.items())
                fig1 = plot_metric({"train_loss": metric.train_loss, "valid_loss": metric.valid_loss},
                                   title="train and valid loss", result_path=self.image_path)
                fig2 = plot_metric({"train acc": metric.train_acc, "valid acc": metric.valid_acc},
                                   title="train and valid acc", ylabel="acc", result_path=self.image_path)
                self.writer.add_figure("learn loss", fig1)
                self.writer.add_figure("learn acc", fig2)
                self.writer.add_text("beat valid loss", f"{metric.best_valid_loss}")
                self.writer.add_text("duration", "{:2f}".format((end_time - start_time) / 60.))
            return model
        except KeyboardInterrupt as e:
            tqdm.write("正在退出")
            # 训练结束时需要进行的工作
            end_time = time.time()
            tqdm.write('Train ran for %.2f minutes' % ((end_time - start_time) / 60.))
            self.logging.info(self.args.model_name + "\t{:.2f}".format((end_time - start_time) / 60.))
            self.logging.info("train loss: {:.4f}, valid loss: {:.4f} train acc: {:.4f}, valid acc: {:.4f}"
                              .format(metric.train_loss[-1], metric.valid_loss[-1],
                                      metric.train_acc[-1], metric.valid_acc[-1]))

            if self.args.save:
                plt.clf()
                torch.save(model, self.final_model_path)
                tqdm.write(f"save model(final): {self.final_model_path}")
                np.save(os.path.join(self.data_path, "train_metric.npy"), metric.items())
                fig1 = plot_metric({"train_loss": metric.train_loss, "valid_loss": metric.valid_loss},
                                   title="train and valid loss", result_path=self.image_path)
                fig2 = plot_metric({"train acc": metric.train_acc, "valid acc": metric.valid_acc},
                                   title="train and valid acc", ylabel="acc", result_path=self.image_path)
                self.writer.add_figure("learn loss", fig1)
                self.writer.add_figure("learn acc", fig2)
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
        metric = Metric(mode="test", with_acc=True)
        test_step = len(test_loader)

        POLQA_Predict = np.zeros([test_num, ])
        POLQA_True = np.zeros([test_num, ])
        idx = 0
        loss_fns = self.get_loss_fn()
        test_loss = 0
        test_acc_num = 0
        cm = np.zeros([self.args.score_class_num, self.args.score_class_num])
        with torch.no_grad():
            for batch_idx, (x, y) in tqdm(enumerate(test_loader)):
                batch = x.shape[0]
                loss, est_polqa, true_polqa, num = self.predict(model, x, y, loss_fns)
                test_loss += loss
                test_acc_num += num
                cm += confuseMatrix(true_polqa, est_polqa, self.args.score_step, self.args.score_class_num)
                POLQA_Predict[idx: idx + batch] = est_polqa
                POLQA_True[idx: idx + batch] = true_polqa
                idx += batch
        metric.cm = cm
        print(metric.cm)
        metric.test_loss = test_loss / test_step
        print("Test loss: {:.4f}".format(metric.test_loss))
        metric.test_acc = test_acc_num / test_num
        print("Test acc: {:.4f}".format(metric.test_acc))
        metric.mse = np.mean((POLQA_True - POLQA_Predict) ** 2)
        print('Test error= %f' % metric.mse)
        metric.lcc = np.corrcoef(POLQA_True, POLQA_Predict)
        print('Linear correlation coefficient= %f' % float(metric.lcc[0][1]))

        metric.srcc = scipy.stats.spearmanr(POLQA_True.T, POLQA_Predict.T)
        print('Spearman rank correlation coefficient= %f' % metric.srcc[0])

        self.logging.info("test loss: {:.4f}, mse: {:.4f},  lcc: {:.4f}, srcc: {:.4f}"
                          .format(metric.test_loss, metric.mse, float(metric.lcc[0][1]), metric.srcc[0]))

        if self.args.save:
            fig = plot_matrix(cm, labels_name=np.arange(1, self.args.score_class_num + 1), result_path=self.image_path)
            self.writer.add_figure("confusion_matrix", fig)
            plt.clf()
            M = np.max([np.max(POLQA_Predict), 5])
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
    # arg = Args("can2dClass", task_type="_qn", qn_input_type=1)
    arg = Args("hasaClass", task_type="_qn", qn_input_type=1, score_step=0.5)
    # arg = Args("hasaClass", task_type="_qn", qn_input_type=1, model_name="hasaClass_cp_qn20240530_001033")
    # arg = Args("can2dClass", model_name="can2dClass20240524_203611")
    # arg = Args("lstmcanClass", model_name="lstmcanClass20240524_185704")

    arg.epochs = 35
    arg.batch_size = 32
    arg.save = True
    arg.lr = 5e-4
    arg.step_size = 5
    arg.delta_loss = 2e-4

    # 用于 qualityNet
    # arg.normalize_output = True

    # 训练Hubert
    # arg.optimizer_type = 1
    # arg.enable_frame = False

    # 训练 CNN / tcn
    # arg.optimizer_type = 1
    # arg.enableFrame = False

    # 训练分类模型
    # arg.focal_gamma = 2
    # arg.smooth = True

    print(arg)

    seed_everything(arg.random_seed)

    # 以Class结尾时，返回TrainerC
    trainer = TrainerC(arg)

    if arg.save and not arg.expire:
        arg.write(arg.model_name)

    # 加载用于预测polqa分数的数据集 x: (B, L, C), y1: (B,), y2: (B, L)
    train_dataset, valid_dataset, test_dataset = load_dataset_qn("wav_train_qn_rs2.list", arg.spilt_rate,
                                                                 arg.fft_size, arg.hop_size,
                                                                 input_type=arg.qn_input_type)

    model = load_qn_model(arg)
    # model = load_pretrained_model(r"models\hasaClass_cp_qn20240530_001033\final.pt")

    model = trainer.train(model, train_dataset=train_dataset, valid_dataset=valid_dataset)
    trainer.test(test_dataset=test_dataset, model=model)
