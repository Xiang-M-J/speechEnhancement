import torch.nn as nn
import torch

from utils import oneHotToFloatTorch, floatTensorToOnehot, floatTensorToClass


class FrameMse(nn.Module):
    def __init__(self, enable=True) -> None:
        super(FrameMse, self).__init__()
        self.enable = enable
        if not enable:
            print("warning! FrameMse 损失被设置为永远返回0")

    def forward(self, input, target):
        true_pesq = target[:, 0]

        if self.enable:
            return torch.mean((10 ** (true_pesq - 5.0)) * torch.mean((input - true_pesq.unsqueeze(1)) ** 2, dim=1))
        else:
            return 0


class FrameMseNo(nn.Module):
    def __init__(self, enable=True) -> None:
        super(FrameMseNo, self).__init__()
        self.enable = enable
        if not enable:
            print("warning! FrameMseNo 损失被设置为永远返回0")

    def forward(self, input, target):
        true_pesq = target[:, 0]

        if self.enable:
            return torch.mean((10 ** (true_pesq - 1.0)) * torch.mean((input - true_pesq.unsqueeze(1)) ** 2, dim=1))
        else:
            return 0


class FrameMse2(nn.Module):
    def __init__(self, enable=True) -> None:
        super(FrameMse2, self).__init__()
        self.enable = enable
        if not enable:
            print("warning! FrameMse2 损失被设置为永远返回0")

    def forward(self, input, target):
        if self.enable:
            y_pred = input.squeeze(1)  # (B,T)
            loss = torch.mean((target - y_pred.detach()) ** 2)
            return loss
        else:
            return 0


class EDMLoss(nn.Module):
    """
    input: N
    forward(estimate, target, step)
    estimate: N C, C=4//step
    target: N 1
    """

    def __init__(self, step, smooth):
        super(EDMLoss, self).__init__()
        self.step = step
        self.smooth = smooth

    def forward(self, p_estimate: torch.Tensor, p_target: torch.Tensor):
        """
        p_target: [B, N]
        p_estimate: [B,]
        B 为批次大小，N 为类别数
        """
        if len(p_estimate.shape) == 1:
            p_estimate_ = floatTensorToOnehot(p_estimate, self.step, self.smooth)
        else:
            p_estimate_ = p_estimate
        p_target_ = floatTensorToOnehot(p_target, self.step, self.smooth)
        assert p_target_.shape == p_estimate_.shape
        cdf_target = torch.cumsum(p_target_, dim=1)
        cdf_estimate = torch.cumsum(p_estimate_, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.mean(torch.pow(torch.abs(cdf_diff), 2), dim=-1)
        return samplewise_emd.mean()


class FocalEDMLoss(nn.Module):
    def __init__(self, step, smooth, gamma=2):
        super(FocalEDMLoss, self).__init__()
        self.step = step
        self.smooth = smooth
        self.gamma = gamma

    def forward(self, p_estimate: torch.Tensor, p_target: torch.Tensor):
        if len(p_estimate.shape) == 1:
            p_estimate_ = floatTensorToOnehot(p_estimate, self.step, self.smooth)
        else:
            p_estimate_ = p_estimate
        p_target_ = floatTensorToOnehot(p_target, self.step, self.smooth)
        assert p_target_.shape == p_estimate_.shape
        cdf_target = torch.cumsum(p_target_, dim=1)
        cdf_estimate = torch.cumsum(p_estimate_, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.sum(torch.pow(torch.abs(cdf_diff), 2), dim=-1) + 1e-5)
        focal_samplewise_emd = torch.pow(samplewise_emd, self.gamma) * torch.log(1 + samplewise_emd)
        return self.step * focal_samplewise_emd.mean()


class FrameEDMLoss(nn.Module):
    """
    input: N L C, C = 4 // step
    target: N L 1
    """

    def __init__(self, smooth, enable, step=0.2) -> None:
        super(FrameEDMLoss, self).__init__()
        self.enable = enable
        self.step = step
        self.smooth = smooth
        if not enable:
            print("warning! FrameEDMLoss 损失被设置为永远返回0")

    def forward(self, input, target):
        if self.enable:
            if len(input.shape) == 2:
                input_ = floatTensorToOnehot(input, self.step, self.smooth)
            else:
                input_ = input
            target_ = floatTensorToOnehot(target, self.step, self.smooth)
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
    input: N L C, C = 4 // step
    target: N L 1
    """

    def __init__(self, step, smooth, enable, gamma=2) -> None:
        super(FocalFrameEDMLoss, self).__init__()
        self.enable = enable
        self.step = step
        self.smooth = smooth
        self.gamma = gamma
        if not enable:
            print("warning! FrameEDMLoss 损失被设置为永远返回0")

    def forward(self, input, target):
        if self.enable:
            if len(input.shape) == 2:
                input_ = floatTensorToOnehot(input, self.step, self.smooth)
            else:
                input_ = input
            target_ = floatTensorToOnehot(target, self.step, self.smooth)
            assert input_.shape[0] == target_.shape[0]
            cdf_input = torch.cumsum(input_, dim=-1)
            cdf_target = torch.cumsum(target_, dim=-1)
            cdf_diff = cdf_input - cdf_target
            samplewise_emd = torch.sqrt(torch.sum(torch.pow(cdf_diff, 2), dim=-1) + 1e-5)
            focal_samplewise_emd = torch.pow(samplewise_emd, self.gamma) * torch.log(1 + samplewise_emd)
            return self.step * focal_samplewise_emd.mean()
        else:
            return 0


class AvgCrossEntropyLoss(nn.Module):
    """
    input: N C, C = 4 // step
    target: N
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
            # input_class = torch.argmax(input, dim=-1).float()
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


class topKLoss(nn.Module):
    def __init__(self, topK, step):
        super(topKLoss, self).__init__()
        self.topK = topK
        self.step = step

    def forward(self, input, target):
        value, index = torch.topk(input, self.topK, dim=-1, out=None)
        pred = value @ (index * self.step + 1.0).t()
        pred = torch.diag(pred)
        return torch.mean(torch.pow(target - pred, 2))


class disLoss(nn.Module):
    def __init__(self, step):
        super(disLoss, self).__init__()
        self.step = step

    def forward(self, input, target):
        pred = oneHotToFloatTorch(input, self.step).squeeze(1)
        return torch.mean(torch.pow(target - pred, 2))


class shiftLoss(nn.Module):
    def __init__(self, step, topk):
        super(shiftLoss, self).__init__()
        self.step = step
        self.topk = topk

    def forward(self, input):
        value, index = torch.topk(input, self.topk, dim=-1, out=None)
        non_pLoss = torch.mean(torch.pow(1 - torch.sum(value, dim=-1), 2))
        # non_cIndex = index - index[:, 0].unsqueeze(1).repeat(1, self.topk)
        # non_cIndex = non_cIndex[:, 1:]
        # non_cLoss = torch.mean(torch.sum(torch.abs(non_cIndex), dim=-1))
        return non_pLoss


class shiftLossWithTarget(nn.Module):
    def __init__(self, step, topk):
        super(shiftLossWithTarget, self).__init__()
        self.step = step
        self.topk = topk
        self.left = (topk - 1) // 2

    def forward(self, input, target):
        # true_index = torch.argmax(target, dim=-1)
        true_index = (((target - 1.0) * 100).int() // int(self.step * 100)).long().unsqueeze(1)
        topk_p = torch.zeros([input.shape[0], 1]).to(input.device)
        input_extend = torch.cat([torch.zeros([input.shape[0], self.left]).to(input.device), input,
                                  torch.zeros([input.shape[0], self.left]).to(input.device)], dim=-1)

        for i in range(0, self.topk):
            topk_p = topk_p + torch.gather(input_extend, dim=-1, index=true_index + i)

        non_pLoss = torch.mean(torch.abs(1 - topk_p))

        # pred_index = torch.argmax(input, dim=-1).unsqueeze(-1)
        # non_cLoss = torch.mean(torch.sum(torch.abs(pred_index - true_index), dim=-1))
        return non_pLoss


class QNLoss(nn.Module):
    def __init__(self, isClass, step):
        super(QNLoss, self).__init__()
        self.isClass = isClass
        self.step = step

    def forward(self, model, input):
        # with torch.no_grad():
        score = model(input)
        if type(score) is tuple:
            score = score[1].squeeze(0)
        if self.isClass:
            score = oneHotToFloatTorch(score, self.step)
        score = (score - 1.0) / 4.0  # 放缩在 0-1之间
        score[score > 1.0] = 1.
        score[score < 0.0] = 0.
        return torch.mean(torch.pow(1 - score, 2))


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, enable=True, step=0.2):
        super(FocalCrossEntropyLoss, self).__init__()
        self.base_loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        """
        input: N C
        target: N / N C
        """
        if len(target.shape) == 2:
            target_class = torch.max(target, -1)[1]
        else:
            target_class = target.long()
        pt = torch.exp(-self.base_loss(input, target))


class CriticLoss(nn.Module):
    def __init__(self, isClass, step):
        super(CriticLoss, self).__init__()
        self.isClass = isClass
        self.step = step

    def forward(self, model, input):
        score = model(input)
        if type(score) is tuple:
            score = score[1].squeeze(0)
        if self.isClass:
            score = oneHotToFloatTorch(score, self.step)
        score = (score - 1.0) / 4.0  # 放缩在 0-1之间
        score[score > 1.0] = 1.
        score[score < 0.0] = 0.
        return torch.mean(score)


if __name__ == "__main__":
    step = 0.5
    p_est = torch.randn([4, 128, 8])
    p_est = torch.softmax(p_est, dim=-1)
    p_tgt = torch.abs(torch.randn([4, 128])) * 2 + 1.0
    p_tgt[p_tgt > 5.0] = 4.99
    # loss = FrameEDMLoss()
    # loss = topKError(topK=5, step=0.2)
    loss = FocalFrameEDMLoss(0.5, True, enable=True, gamma=2)
    y = (loss(p_est, p_tgt))
    print(y)
