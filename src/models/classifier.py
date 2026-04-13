"""
缺陷分类模型
"""
import torch
import torch.nn as nn
import timm
from typing import Dict, List, Optional


class DefectClassifier(nn.Module):
    """缺陷分类器 - 图像 + 位置特征融合"""
    
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        img_size: int = 256,
        pos_feat_dim: int = 11,
        fusion_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # 图像分支 (CNN)
        self.cnn = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头
            global_pool='avg',
        )
        img_feat_dim = self.cnn.num_features
        
        # 位置特征分支 (MLP)
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(img_feat_dim + 64, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )
        
        # 分类头
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, img: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: [B, 3, H, W] 图像
            pos: [B, pos_feat_dim] 位置特征
        Returns:
            [B, num_classes] 分类logits
        """
        # 图像特征
        img_feat = self.cnn(img)  # [B, img_feat_dim]
        
        # 位置特征
        pos_feat = self.pos_mlp(pos)  # [B, 64]
        
        # 融合
        fused = torch.cat([img_feat, pos_feat], dim=-1)  # [B, img_feat_dim+64]
        fused = self.fusion(fused)
        
        # 分类
        logits = self.classifier(fused)
        return logits
    
    def get_feature(self, img: torch.Tensor, pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        """获取特征用于可视化/分析"""
        img_feat = self.cnn(img)
        pos_feat = self.pos_mlp(pos)
        fused = torch.cat([img_feat, pos_feat], dim=-1)
        fused = self.fusion(fused)
        
        return {
            'img_feat': img_feat,
            'pos_feat': pos_feat,
            'fused': fused,
        }


class MultiScaleClassifier(nn.Module):
    """多尺度分类器 - 处理不同大小缺陷"""
    
    def __init__(
        self,
        num_classes: int = 5,
        backbone: str = "efficientnet_b0",
        scales: List[float] = [1.0, 1.5, 2.0],
    ):
        super().__init__()
        
        self.scales = scales
        
        # 共享CNN
        self.cnn = timm.create_model(backbone, pretrained=True, num_classes=0)
        img_feat_dim = self.cnn.num_features
        
        # 位置编码
        self.pos_encoder = nn.Linear(11, 64)
        
        # 尺度注意力
        self.scale_attention = nn.Sequential(
            nn.Linear(img_feat_dim * len(scales), len(scales)),
            nn.Softmax(dim=-1),
        )
        
        # 融合分类
        self.classifier = nn.Sequential(
            nn.Linear(img_feat_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, multi_scale_imgs: List[torch.Tensor], pos: torch.Tensor) -> torch.Tensor:
        """多尺度输入
        
        Args:
            multi_scale_imgs: [(B,3,H,W), ...] 不同尺度的图像
        """
        # 提取各尺度特征
        scale_feats = []
        for img in multi_scale_imgs:
            feat = self.cnn(img)
            scale_feats.append(feat)
        
        # 注意力加权
        concat_feats = torch.cat(scale_feats, dim=-1)
        attention = self.scale_attention(concat_feats)
        weighted_feat = sum(f * a.view(-1, 1) for f, a in zip(scale_feats, attention))
        
        # 位置特征
        pos_enc = torch.relu(self.pos_encoder(pos))
        
        # 融合
        fused = torch.cat([weighted_feat, pos_enc], dim=-1)
        return self.classifier(fused)


class LightweightClassifier(nn.Module):
    """轻量级分类器 (边缘设备用)"""
    
    def __init__(self, num_classes: int = 5):
        super().__init__()
        
        # 轻量CNN (MobileNetV3 风格)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
        )
        
        # 位置 + 图像融合
        self.classifier = nn.Sequential(
            nn.Linear(64 + 11, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, img: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        feat = self.features(img).flatten(1)
        fused = torch.cat([feat, pos], dim=-1)
        return self.classifier(fused)


def build_model(name: str = "defect_classifier", **kwargs) -> nn.Module:
    """模型工厂"""
    models = {
        'defect_classifier': DefectClassifier,
        'multiscale': MultiScaleClassifier,
        'lightweight': LightweightClassifier,
    }
    
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    
    return models[name](**kwargs)
