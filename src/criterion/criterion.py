import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import one_hot


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = -1, gamma: float = 2, reduction: str = 'mean', star: bool = False):
        """
        初始化 Focal Loss 类。
        :param alpha: 用于平衡正负样本的权重因子，默认为 -1（不加权）。
        :param gamma: 用于调节难易样本的权重，默认为 2。
        :param reduction: 'none' | 'mean' | 'sum'，指定如何归约损失。默认为 'mean'。
        :param star:
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.star = star

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失。
        :param inputs: 模型的输出（logits）。
        :param targets: 真实标签，0 或 1。
        :return: 计算的损失值。
        """
        inputs = inputs.float()
        if targets.shape != inputs.shape:
            targets = one_hot(targets, 2)
        targets = targets.float()

        if self.star:
            shifted_inputs = self.gamma * (inputs * (2 * targets - 1))
            loss = -(F.logsigmoid(shifted_inputs)) / self.gamma
        else:
            p = torch.sigmoid(inputs)
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            p_t = p * targets + (1 - p) * (1 - targets)
            loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class CLIPLoss(torch.nn.Module):
    def __init__(self):
        """
        CLIP 的 InfoNCE 损失函数
        :param temperature: 温度参数 τ，控制分布平滑度
        """
        super().__init__()
        self.img_loss = nn.CrossEntropyLoss()
        self.text_loss = nn.CrossEntropyLoss()

    def forward(self, input):
        """
        计算 CLIP 的 InfoNCE 损失
        :return: 对比学习损失值
        """
        logits_per_image, logits_per_text = input
        labels = torch.arange(len(logits_per_image), dtype=torch.long, device=logits_per_image.device)
        # 计算交叉熵损失 (图像->文本 & 文本->图像)
        loss_img = self.img_loss(logits_per_image, labels)
        loss_txt = self.text_loss(logits_per_text, labels)
        # 返回平均损失
        return (loss_img + loss_txt) / 2
