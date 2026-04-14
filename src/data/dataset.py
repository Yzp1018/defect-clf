"""
缺陷数据集 - 支持 train/val/test 三个数据集
"""
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import pandas as pd

from .augment import DataAugmentor


class DefectDataset(Dataset):
    """缺陷数据集
    
    bbox格式: [cx, cy, w, h] - 中心点 + 宽高
    
    目录结构:
    data/
        train/
            annotations.json
            images/
        val/
            annotations.json
            images/
        test/
            annotations.json
            images/
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        crop_size: int = 256,
        use_context: float = 0.2,
        enable_augment: bool = True,
        augment_config: Optional[Dict] = None,
    ):
        """
        Args:
            data_dir: 数据根目录 (如 'data')
            split: 'train' | 'val' | 'test'
            transform: 数据增强 (已废弃，使用 enable_augment)
            crop_size: 裁剪尺寸
            use_context: 上下文比例
            enable_augment: 是否启用数据增强
            augment_config: 数据增强配置
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        self.transform = transform
        self.crop_size = crop_size
        self.use_context = use_context
        
        # 数据增强器
        self.enable_augment = enable_augment and (split == 'train')
        if self.enable_augment:
            aug_cfg = augment_config or {}
            self.augmentor = DataAugmentor(img_size=crop_size, **aug_cfg)
        else:
            self.augmentor = None
        
        # 加载标注 (新路径: data/{split}/annotations.json)
        self.annotations = self._load_annotations()
        
        # 类别映射
        self.classes = sorted(set(a['class'] for a in self.annotations))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        
    def _load_annotations(self) -> List[Dict]:
        """加载标注"""
        # 新路径: data/{train,val,test}/annotations.json
        ann_file = self.split_dir / "annotations.json"
        
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                data = json.load(f)
            return data.get('annotations', [])
        
        # 旧路径兼容: data/annotations/{split}.json
        ann_file = self.data_dir / "annotations" / f"{self.split}.json"
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                data = json.load(f)
            return data.get('annotations', [])
        
        # CSV格式
        ann_file = self.split_dir / "annotations.csv"
        if ann_file.exists():
            df = pd.read_csv(ann_file)
            return df.to_dict('records')
        
        raise FileNotFoundError(f"No annotations found: {self.split_dir}/")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        ann = self.annotations[idx]
        
        # 加载图像 (优先从 split 目录查找)
        img_path = self.split_dir / "images" / ann['image_name']
        if not img_path.exists():
            # 回退到 data/images/
            img_path = self.data_dir / "images" / ann['image_name']
        
        image = cv2.imread(str(img_path))
        if image is None:
            # 如果图像不存在，创建空白图像用于测试
            image = np.zeros((4096, 8192, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # bbox: [cx, cy, w, h] 中心点 + 宽高
        cx, cy, w, h = ann['bbox']
        
        # 转换为左上角坐标 (x, y, w, h) 用于裁剪
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        bbox_xywh = [x, y, int(w), int(h)]
        
        # 裁剪缺陷ROI
        crop = self._crop_defect(image, bbox_xywh)
        crop = cv2.resize(crop, (self.crop_size, self.crop_size))
        
        # 数据增强 (训练时)
        if self.augmentor is not None:
            crop = self.augmentor(crop)
        elif self.transform and self.split == 'train':
            crop = self.transform(crop)
        
        # 转换为tensor
        crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        
        # 位置特征 (使用 cx, cy, w, h 格式)
        h_img, w_img = image.shape[:2]
        pos_feat = self._extract_position_features(ann['bbox'], (h_img, w_img))
        pos_tensor = torch.tensor(pos_feat, dtype=torch.float32)
        
        # 类别
        label = self.class_to_idx[ann['class']]
        
        return {
            'image': crop_tensor,
            'position': pos_tensor,
            'label': label,
            'image_id': ann.get('id', idx),
            'bbox': ann['bbox'],  # [cx, cy, w, h]
            'class_name': ann['class'],
        }
    
    def _crop_defect(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """裁剪缺陷区域"""
        x, y, w, h = bbox
        
        # 添加上下文
        cx, cy = x + w/2, y + h/2
        pw, ph = w * (1 + self.use_context), h * (1 + self.use_context)
        
        # 边界处理
        h_img, w_img = image.shape[:2]
        x1 = max(0, int(cx - pw/2))
        y1 = max(0, int(cy - ph/2))
        x2 = min(w_img, int(cx + pw/2))
        y2 = min(h_img, int(cy + ph/2))
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros((int(ph), int(pw), 3), dtype=np.uint8)
        
        return image[y1:y2, x1:x2]
    
    def _extract_position_features(self, bbox: List[float], img_shape: Tuple) -> List[float]:
        """提取位置特征"""
        from ..features.position import PositionFeatureExtractor
        extractor = PositionFeatureExtractor()
        
        cx, cy, w, h = bbox
        x = cx - w / 2
        y = cy - h / 2
        bbox_xywh = [x, y, w, h]
        
        feat = extractor.extract(bbox_xywh, img_shape)
        
        features = [
            feat['cx_norm'], feat['cy_norm'],
            feat['rel_width'], feat['rel_height'], feat['rel_area'],
            feat['min_edge_dist'], feat['min_corner_dist'],
            feat['is_edge'], feat['is_corner'],
            feat['orientation'], feat['density'],
        ]
        return features
    
    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        dist = {}
        for ann in self.annotations:
            cls = ann['class']
            dist[cls] = dist.get(cls, 0) + 1
        return dist


def create_dataloaders(
    data_dir: str, 
    batch_size: int = 32, 
    num_workers: int = 4,
    crop_size: int = 256,
    use_context: float = 0.2,
):
    """创建 train/val/test 三个数据加载器"""
    from torch.utils.data import DataLoader
    
    train_ds = DefectDataset(data_dir, split="train", crop_size=crop_size, use_context=use_context)
    val_ds = DefectDataset(data_dir, split="val", crop_size=crop_size, use_context=use_context)
    test_ds = DefectDataset(data_dir, split="test", crop_size=crop_size, use_context=use_context)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
