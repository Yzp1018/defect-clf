"""
损失函数模块 - 包含 Focal Loss 等处理类别不平衡的损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡
    
    Args:
        alpha: 类别权重，可以是标量或列表 [C]
        gamma: 聚焦参数，越大越关注难分类样本
        reduction: 'mean' | 'sum' | 'none'
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha  # [C] or scalar
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] 预测 logits
            targets: [B] 真实标签
        Returns:
            loss: 标量或 [B]
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [B]
        
        # 计算概率
        probs = F.softmax(logits, dim=-1)  # [B, C]
        pt = probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)  # [B]
        
        # Focal Loss 权重
        focal_weight = (1 - pt) ** self.gamma
        
        # 类别权重
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # alpha: [C] -> [B]
                alpha_t = self.alpha.gather(0, targets)  # [B]
            focal_weight = focal_weight * alpha_t
        
        loss = focal_weight * ce_loss  # [B]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss  # [B]


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy"""
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C]
            targets: [B]
        """
        B, C = logits.shape
        
        # One-hot 编码
        one_hot = torch.zeros_like(logits).scatter(1, targets.unsqueeze(1), 1)
        
        # Smooth
        smooth_labels = one_hot * (1 - self.smoothing) + self.smoothing / C
        
        # Log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Loss
        loss = -(smooth_labels * log_probs).sum(dim=1)  # [B]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """组合损失：Focal Loss + Label Smoothing"""
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        focal_weight: float = 0.8,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        self.smooth_loss = LabelSmoothingCrossEntropy(smoothing=smoothing, reduction=reduction)
        self.focal_weight = focal_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal = self.focal_loss(logits, targets)
        smooth = self.smooth_loss(logits, targets)
        return self.focal_weight * focal + (1 - self.focal_weight) * smooth


def build_loss_fn(
    name: str = 'focal',
    class_weights: Optional[list] = None,
    gamma: float = 2.0,
    smoothing: float = 0.1,
    device: str = 'cpu',
    **kwargs
) -> nn.Module:
    """构建损失函数工厂
    
    Args:
        name: 'focal' | 'smooth_ce' | 'combined' | 'ce'
        class_weights: 类别权重列表
        gamma: Focal Loss gamma
        smoothing: Label Smoothing 系数
        device: 设备
    """
    if class_weights is not None:
        alpha = torch.tensor(class_weights, dtype=torch.float32).to(device)
    else:
        alpha = None
    
    if name == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma, **kwargs)
    elif name == 'smooth_ce':
        return LabelSmoothingCrossEntropy(smoothing=smoothing, **kwargs)
    elif name == 'combined':
        return CombinedLoss(alpha=alpha, gamma=gamma, smoothing=smoothing, **kwargs)
    elif name == 'ce':
        return nn.CrossEntropyLoss(weight=alpha, **kwargs)
    else:
        raise ValueError(f"Unknown loss: {name}")
