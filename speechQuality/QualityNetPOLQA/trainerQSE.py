import os
import time

import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
import torch.nn as nn
from torch.utils.data import dataloader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from losses import QNLoss
from trainer_utils import Args, EarlyStopping, Metric, plot_metric, plot_spectrogram, plot_quantity, load_dataset_se, \
    load_pretrained_model
from utils import getStftSpec, spec2wav, preprocess, CalSigmos, seed_everything, get_logging

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TrainerQSE:
    """
    训练语音增强模型
    """

    def __init__(self, args: Args):

        if not args.model_type.endswith("_qse"):
            raise ValueError("Model type must end with '_qse'")
        self.args: Args = args
        self.optimizer_type = args.optimizer_type
        self.model_path = f"models/{args.model_name}/"
        self.best_model_path = self.model_path + "best.pt"  # 模型保存路径(max val acc)
        self.final_model_path = self.model_path + "final.pt"  # 模型保存路径(final)
        self.result_path = f"results/{args.model_name}/"  # 结果保存路径（分为数据和图片）
        self.image_path = self.result_path + "images/"
        self.data_path = self.result_path + "data/"
        self.inference_result_path = f"inference_results/{args.model_name}"
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.save_model_epoch = args.save_model_epoch
        self.lr = args.lr
        self.test_acc = []
        self.logging = get_logging("log_se.txt")
        if args.save:
            self.check_dir()
            self.writer = SummaryWriter("runs/" + self.args.model_name)

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
        # if not os.path.exists(self.inference_result_path):
        #     os.makedirs(self.inference_result_path)

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

    @staticmethod
    def get_loss_fn():
        loss_fn = nn.MSELoss()
        loss_fn.to(device=device)
        return loss_fn

    def train_epoch(self, model, model_qn, x, y, loss_fn, optimizer):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        if self.args.se_input_type == 2:
            mag_pred = torch.pow(y_pred[:, 0, :, :], 2) + torch.pow(y_pred[:, 1, :, :], 2)
            mag_true = torch.pow(y[:, 0, :, :], 2) + torch.pow(y[:, 1, :, :], 2)
        else:
            mag_pred = torch.pow(y_pred, 2)
            mag_true = torch.pow(y, 2)

        mag = torch.concat([mag_pred, mag_true], 0)
        # GRL.apply(mag, 1)
        loss = loss_fn(model_qn, mag)  # qualityNet让loss尽可能小，即更接近0，对应的语音增强模型则让loss尽可能小
        loss.requires_grad_(True)
        # loss_qn = -loss_fn(mag_pred)
        # loss_qn.requires_grad_(True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


    def predict(self, model, model_qn, x, y, loss_fn):
        """
        Return loss, y_pred
        """
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        if self.args.se_input_type == 2:
            mag_pred = torch.pow(y_pred[:, 0, :, :], 2) + torch.pow(y_pred[:, 1, :, :], 2)
            mag_true = torch.pow(y[:, 0, :, :], 2) + torch.pow(y[:, 1, :, :], 2)
        else:
            mag_pred = torch.pow(y_pred, 2)
            mag_true = torch.pow(y, 2)
        loss = loss_fn(model_qn, mag_pred)
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
        self.freeze_parameters(model=model_qn)
        loss_fn = QNLoss(isClass=("Class" in self.args.model2_type), step=self.args.score_step).to(device)

        optimizer = self.get_optimizer([{"params": model_se.parameters(), "lr": self.lr}], lr=self.lr)
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
                    loss = self.train_epoch(model_se, model_qn, x, y, loss_fn, optimizer)
                    train_loss += loss

                    loop_train.set_description_str(f'Training [{epoch + 1}/{self.epochs}]')
                    loop_train.set_postfix_str("step: {}/{} loss: {:.4f}".format(batch_idx, train_step, loss))

                model_se.eval()
                model_qn.eval()
                q_len = 100
                idx = 0
                mos_48k = {"col": [], "disc": [], "loud": [], "noise": [], "reverb": [], "sig": [], "ovrl": []}
                mos_48k_name = list(mos_48k.keys())

                with torch.no_grad():
                    loop_valid = tqdm(enumerate(valid_loader), leave=False)
                    for batch_idx, (x, xp, y, yp) in loop_valid:
                        loss, est_x = self.predict(model_se, model_qn, x, y, loss_fn)
                        valid_loss += loss
                        batch = x.shape[0]
                        if idx < q_len:
                            est_wav = spec2wav(est_x, xp, fft_size=self.args.fft_size, hop_size=self.args.hop_size,
                                               win_size=self.args.fft_size, input_type=self.args.se_input_type)
                            results = calSigmos(est_wav.cpu().detach().numpy())
                            for i in range(7):
                                mos_48k[mos_48k_name[i]].extend(results[i])
                        idx += batch
                        loop_valid.set_description_str(f'Validating [{epoch + 1}/{self.epochs}]')
                        loop_valid.set_postfix_str("step: {}/{} loss: {:.4f}".format(batch_idx, valid_step, loss))

                print("")
                msg = ""
                for k, v in mos_48k.items():
                    msg += f"{k}: {np.mean(v)}\t"
                    # print(f"{k}: {np.mean(v)}", end="\t")
                print(msg)
                self.logging.info(msg)

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
            with open("log.txt", mode='a', encoding="utf-8") as f:
                f.write(
                    self.args.model_name + f"\t{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\t" + "{:.2f}".format(
                        (end_time - start_time) / 60.) + "\n")
                f.write(
                    "train loss: {:.4f}, valid loss: {:.4f} \n".format(metric.train_loss[-1], metric.valid_loss[-1]))
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
            with open("log.txt", mode='a', encoding="utf-8") as f:
                f.write(
                    self.args.model_name + f"\t{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\t" + "{:.2f}".format(
                        (end_time - start_time) / 60.) + "\n")
                f.write("train loss: {:.4f}, valid loss: {:.4f}\n".format(metric.train_loss[-1], metric.valid_loss[-1]))
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
        # calQuantity = CalQuality(fs=48000, batch=True)
        calSigmos = CalSigmos(fs=48000, batch=True)

        loss_fn = QNLoss(isClass=("Class" in self.args.model2_type), step=self.args.score_step)
        test_loss = 0
        metric.mos_48k = {}
        for i in range(7):
            metric.mos_48k[metric.mos_48k_name[i]] = []
        # predict_pesq = []
        # predict_stoi = []
        idx = 0
        with torch.no_grad():
            for batch_idx, (x, xp, y, yp) in tqdm(enumerate(test_loader)):
                loss, est_x = self.predict(model, model_qn, x, y, loss_fn)
                test_loss += loss
                batch = x.shape[0]
                if idx < q_len:
                    est_wav = spec2wav(est_x, xp, fft_size=self.args.fft_size, hop_size=self.args.hop_size,
                                       win_size=self.args.fft_size, input_type=self.args.se_input_type)
                    # p, s = calQuantity(est_wav.cpu().detach(), yp.cpu().detach())
                    # predict_pesq[idx: idx + batch] = p
                    # predict_stoi[idx: idx + batch] = s
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

        # metric.pesq = predict_pesq
        # metric.stoi = predict_stoi

        with open("log.txt", mode='a', encoding="utf-8") as f:
            f.write("test loss: {:.4f} \n".format(metric.test_loss))

        # fig1 = plot_quantity(predict_pesq, "测试语音的pesq", ylabel="pesq", result_path=self.image_path)
        # fig2 = plot_quantity(predict_stoi, "测试语音的stoi", ylabel="stoi", result_path=self.image_path)
        if self.args.save:
            self.writer.add_text("test metric", str(metric))
            # self.writer.add_figure("pesq", fig1)
            # self.writer.add_figure("stoi", fig2)
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

    def inference_step(self, model, noise_wav_path, target_wav_path):
        if not os.path.exists(self.inference_result_path):
            os.makedirs(self.inference_result_path)
        model.eval()
        wav_name = noise_wav_path.split('\\')[-1].split('.')[0]
        noise_wav, fs = preprocess(noise_wav_path)
        # calQuantity = CalQuality(fs=fs, batch=False)
        calSigmos = CalSigmos(fs=48000, batch=False)
        fig1 = plot_spectrogram(noise_wav, fs, self.args.fft_size, self.args.hop_size,
                                filename=wav_name + "_带噪语音语谱图",
                                result_path=self.inference_result_path)

        target_wav, fs = preprocess(target_wav_path)
        fig2 = plot_spectrogram(target_wav, fs, self.args.fft_size, self.args.hop_size,
                                filename=wav_name + "_干净语音语谱图",
                                result_path=self.inference_result_path)

        feat_x, phase_x = getStftSpec(noise_wav, self.args.fft_size, self.args.hop_size, self.args.fft_size, self.args.se_input_type)
        with torch.no_grad():
            est_x = model(feat_x.unsqueeze(0).to(device)).squeeze(0).cpu().detach()
        est_wav = spec2wav(est_x.unsqueeze(0), phase_x, self.args.fft_size, self.args.hop_size, self.args.fft_size,
                           self.args.se_input_type)
        est_wav = est_wav.squeeze(0).cpu()
        fig3 = plot_spectrogram(est_wav, fs, self.args.fft_size, self.args.hop_size,
                                filename=wav_name + "_增强语音语谱图",
                                result_path=self.inference_result_path)

        # p, s = calQuantity(est_wav.cpu().detach().squeeze(0), target_wav)
        # print(p, s)
        # with open(os.path.join(self.inference_result_path, wav_name + "ps.txt"), 'w', encoding="utf-8") as f:
        #     f.write(str(p) + "\t" + str(s) + "\n")
        result = calSigmos(est_wav.cpu().detach().numpy())
        print(result)
        with open(os.path.join(self.inference_result_path, wav_name + "mos.txt"), 'w', encoding="utf-8") as f:
            f.write(noise_wav_path)
            for r in result:
                f.write(str(r) + "\t")
            f.write("\n")

        denoise_wav_path = wav_name + "_denoise.wav"
        soundfile.write(os.path.join(self.inference_result_path, denoise_wav_path), est_wav.numpy(), samplerate=48000)
        if self.args.save:
            self.writer.add_audio(wav_name, noise_wav, sample_rate=48000)
            if target_wav_path is not None:
                self.writer.add_audio(wav_name + "_无噪声", target_wav, sample_rate=48000)
            self.writer.add_audio(wav_name + "_增强后", est_wav, sample_rate=48000)


if __name__ == "__main__":
    # path_se = r"D:\work\speechEnhancement\speechQuality\QualityNetPOLQA\models\dpcrn_se20240518_224558\final.pt"
    path_se = r"D:\work\speechEnhancement\speechQuality\QualityNetPOLQA\models\lstm_se20240521_173158\final.pt"
    # path_qn = r"D:\work\speechEnhancement\speechQuality\QualityNetPOLQA\models\lstmClass20240515_200350\final.pt"
    # path_qn = r"D:\work\speechEnhancement\speechQuality\QualityNetPOLQA\models\cnn20240515_100107\final.pt"
    path_qn = r"D:\work\speechEnhancement\speechQuality\HASANetPOLQA\models\hasa20240516_134107\final.pt"
    # arg = Args("dpcrn", task_type="_qse", model_name="dpcrn_se20240518_224558", model2_type="cnn")
    arg = Args("lstm", task_type="_qse", model2_type="hasa")
    arg.epochs = 15
    arg.batch_size = 4
    arg.save = False
    arg.lr = 1e-4
    arg.step_size = 5
    arg.delta_loss = 2e-4

    if not arg.model_type.endswith("_qse"):
        raise ValueError("Model type must end with '_qse'")
    if arg.model2_type is None:
        raise ValueError("model qn type cannot be none")

    # 训练 CNN / tcn
    arg.optimizer_type = 3
    # arg.enableFrame = False
    arg.se_input_type = 1

    # 训练分类模型
    # arg.score_step = 0.2
    # arg.focal_gamma = 2
    # arg.smooth = True

    print(arg)
    if arg.save:
        arg.write(arg.model_name)

    seed_everything(arg.random_seed)

    # 加载用于训练语音增强模型的数据集 x: (B, L, C)  y: (B L C) 或者 x: (B, 2, L, C)  y: (B, 2, L, C)
    train_dataset, valid_dataset, test_dataset = load_dataset_se("wav_train_se.list", arg.spilt_rate,
                                                                 arg.fft_size, arg.hop_size, arg.se_input_type)

    model_se = load_pretrained_model(path_se)
    model_qn = load_pretrained_model(path_qn)

    trainer = TrainerQSE(arg)

    model_se = trainer.train(model_se, model_qn, train_dataset=train_dataset, valid_dataset=valid_dataset)
    trainer.test(test_dataset=test_dataset, model=model_se, model_qn=model_qn, q_len=200)
    # trainer.inference_step(model_se, r"D:\work\speechEnhancement\datasets\dns_to_liang\31435_nearend.wav",
    #                        r"D:\work\speechEnhancement\datasets\dns_to_liang\31435_target.wav")
    # trainer.test(test_dataset=test_dataset, model_path=r"D:\work\speechEnhancement\speechQuality\QualityNetPOLQA\models\QN20240508_174129\best.pt")
