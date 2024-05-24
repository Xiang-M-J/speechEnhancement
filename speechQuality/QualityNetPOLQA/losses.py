import torch.nn as nn
import torch

from utils import oneHotToFloatTorch, floatTensorToOnehot, floatTensorToClass


class FrameMse(nn.Module):
    def __init__(self, enable=True) -> None:
        super(FrameMse, self).__init__()
        self.enable = enable
        if not enable:
            print("warning! FrameMse 损失被设置为永远返回0")

    def forward(self, inp, tgt):
        true_pesq = tgt[:, 0]

        if self.enable:
            return torch.mean((10 ** (true_pesq - 5.0)) * torch.mean((inp - true_pesq.unsqueeze(1)) ** 2, dim=1))
        else:
            return 0


class FrameMseNorm(nn.Module):
    def __init__(self, enable=True) -> None:
        super(FrameMseNorm, self).__init__()
        self.enable = enable
        if not enable:
            print("warning! FrameMseNorm 损失被设置为永远返回0")

    def forward(self, inp, tgt):
        true_pesq = tgt[:, 0]

        if self.enable:
            return torch.mean((10 ** (true_pesq - 1.0)) * torch.mean((inp - true_pesq.unsqueeze(1)) ** 2, dim=1))
        else:
            return 0


class FrameMse2(nn.Module):
    def __init__(self, enable=True) -> None:
        super(FrameMse2, self).__init__()
        self.enable = enable
        if not enable:
            print("warning! FrameMse2 损失被设置为永远返回0")

    def forward(self, inp, tgt):
        if self.enable:
            y_pred = inp.squeeze(1)  # (B,T)
            loss = torch.mean((tgt - y_pred.detach()) ** 2)
            return loss
        else:
            return 0


class EDMLoss(nn.Module):
    """
    inp: N
    forward(estimate, tgt, step)
    estimate: N C, C=4//step
    tgt: N 1
    """

    def __init__(self, step, smooth):
        super(EDMLoss, self).__init__()
        self.step = step
        self.smooth = smooth

    def forward(self, p_estimate: torch.Tensor, p_target: torch.Tensor):
        """
        p_target: [B, N]
        p_estimate: [B, N]
        B 为批次大小，N 为类别数
        """

        p_estimate_ = torch.softmax(p_estimate, dim=-1)
        p_target_ = floatTensorToOnehot(p_target, self.step, self.smooth)
        assert p_target_.shape == p_estimate_.shape
        cdf_target = torch.cumsum(p_target_, dim=1)
        cdf_estimate = torch.cumsum(p_estimate_, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        emd = torch.mean(torch.pow(torch.abs(cdf_diff), 2), dim=-1)
        return emd.mean()


class FocalEDMLoss(nn.Module):
    def __init__(self, step, smooth, gamma=2):
        super(FocalEDMLoss, self).__init__()
        self.step = step
        self.smooth = smooth
        self.gamma = gamma

    def forward(self, p_estimate: torch.Tensor, p_target: torch.Tensor):
        p_estimate_ = torch.softmax(p_estimate, dim=-1)
        p_target_ = floatTensorToOnehot(p_target, self.step, self.smooth)
        assert p_target_.shape == p_estimate_.shape
        cdf_target = torch.cumsum(p_target_, dim=1)
        cdf_estimate = torch.cumsum(p_estimate_, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        emd = torch.sqrt(torch.sum(torch.pow(torch.abs(cdf_diff), 2), dim=-1) + 1e-5)
        focal_emd = torch.pow(emd, self.gamma) * torch.log(1 + emd)
        return self.step * focal_emd.mean()


class FrameEDMLoss(nn.Module):
    """
    inp: N L C, C = 4 // step
    tgt: N L 1
    """

    def __init__(self, smooth, enable, step=0.2) -> None:
        super(FrameEDMLoss, self).__init__()
        self.enable = enable
        self.step = step
        self.smooth = smooth
        if not enable:
            print("warning! FrameEDMLoss 损失被设置为永远返回0")

    def forward(self, inp, tgt):
        if self.enable:
            input_ = torch.softmax(inp, dim=-1)
            target_ = floatTensorToOnehot(tgt, self.step, self.smooth)
            assert input_.shape[0] == target_.shape[0]
            cdf_input = torch.cumsum(input_, dim=-1)
            cdf_target = torch.cumsum(target_, dim=-1)
            cdf_diff = cdf_input - cdf_target
            samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_diff, 2), dim=-1) + 1e-5)
            return samplewise_emd.mean()
        else:
            return 0


class FocalFrameEDMLoss(nn.Module):
    """
    inp: N L C, C = 4 // step
    tgt: N L 1
    """

    def __init__(self, step, smooth, enable, gamma=2) -> None:
        super(FocalFrameEDMLoss, self).__init__()
        self.enable = enable
        self.step = step
        self.smooth = smooth
        self.gamma = gamma
        if not enable:
            print("warning! FrameEDMLoss 损失被设置为永远返回0")

    def forward(self, inp, tgt):
        if self.enable:
            input_ = torch.softmax(inp, dim=-1)
            target_ = floatTensorToOnehot(tgt, self.step, self.smooth)
            assert input_.shape[0] == target_.shape[0]
            cdf_input = torch.cumsum(input_, dim=-1)
            cdf_target = torch.cumsum(target_, dim=-1)
            cdf_diff = cdf_input - cdf_target
            emd = torch.sqrt(torch.sum(torch.pow(cdf_diff, 2), dim=-1) + 1e-5)
            focal_emd = torch.pow(emd, self.gamma) * torch.log(1 + emd)
            return self.step * focal_emd.mean()
        else:
            return 0


class AvgCrossEntropyLoss(nn.Module):
    """
    inp: N C, C = 4 // step
    tgt: N
    """

    def __init__(self, enable=True, step=0.2) -> None:
        super(AvgCrossEntropyLoss, self).__init__()
        self.enable = enable
        self.loss = nn.CrossEntropyLoss()
        self.step = step
        if not enable:
            print("warning! FrameCrossEntropyLoss 损失被设置为永远返回0")

    def forward(self, input, target):
        if self.enable:
            # input_class = torch.argmax(inp, dim=-1).float()
            target_class = floatTensorToClass(target, self.step).squeeze(-1)
            return self.loss(input, target_class)
        else:
            return 0


class FrameCrossEntropyLoss(nn.Module):
    def __init__(self, enable=True, step=0.2) -> None:
        super(FrameCrossEntropyLoss, self).__init__()
        self.enable = enable
        self.loss = nn.CrossEntropyLoss()
        self.step = step
        if not enable:
            print("warning! FrameCrossEntropyLoss 损失被设置为永远返回0")

    def forward(self, input, target):
        if self.enable:
            target_class = floatTensorToClass(target, self.step)
            return self.loss(torch.permute(input, [0, 2, 1]), target_class)
        else:
            return 0


class QNLoss(nn.Module):
    def __init__(self, norm, isClass, step):
        super(QNLoss, self).__init__()
        self.norm = norm
        self.isClass = isClass
        self.step = step

    def forward(self, model, inp):
        score = model(inp)
        if type(score) is tuple:
            score = score[1].squeeze(0)
        if self.isClass:
            score = oneHotToFloatTorch(score, self.step)
            score = (score - 1.0) / 4.0
        else:
            if not self.norm:
                score = (score - 1.0) / 4.0
        return torch.mean(torch.pow(1 - score, 2))


class CriticLoss(nn.Module):
    def __init__(self, norm, isClass, step):
        super(CriticLoss, self).__init__()
        self.norm = norm
        self.isClass = isClass
        self.step = step

    def forward(self, model, inp):
        score = model(inp)
        if type(score) is tuple:
            score = score[1].squeeze(0)
        if self.isClass:
            score = oneHotToFloatTorch(score, self.step)
            score = (score - 1.0) / 4.0
        else:
            if not self.norm:
                score = (score - 1.0) / 4.0

        score = (score - 1.0) / 4.0  # 放缩在 0-1之间
        score[score > 1.0] = 1.
        score[score < 0.0] = 0.
        return torch.mean(score)


class NormMseLoss(nn.Module):
    """
    带归一化的 MSE 损失
    """
    def __init__(self, ):
        super(NormMseLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, inp, tgt):
        inp_ = inp.sigmoid()
        tgt_ = (tgt - 1.0) / 4.0
        loss = self.mse(inp_, tgt_)
        return loss


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(FocalCrossEntropyLoss, self).__init__()
        self.base_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, inp, target):
        """
        inp: N C
        tgt: N / N C
        """
        if len(target.shape) == 2:
            target_class = torch.max(target, -1)[1]
        else:
            target_class = target.long()
        inp_ = torch.softmax(inp, dim=-1)
        pt = inp_[torch.arange(len(target_class)), target_class]
        base_loss = self.base_loss(inp_, target_class)
        return torch.mean(torch.pow(1 - pt, 2) * base_loss)


if __name__ == "__main__":
    step = 0.5
    p_est = torch.randn([4, 8])
    p_est = torch.softmax(p_est, dim=-1)
    p_tgt = torch.abs(torch.randn([4, 8])) * 2 + 1.0
    p_tgt[p_tgt > 5.0] = 4.99
    # loss = FrameEDMLoss()
    # loss = topKError(topK=5, step=0.2)
    # loss = FocalFrameEDMLoss(0.5, True, enable=True, gamma=2)
    loss = FocalCrossEntropyLoss()
    y = (loss(p_est, p_tgt))
    print(y)
