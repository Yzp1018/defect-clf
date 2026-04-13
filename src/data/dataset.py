"""
缺陷数据集 - 支持 cx, cy, w, h 格式
"""
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd


class DefectDataset(Dataset):
    """缺陷数据集
    
    bbox格式: [cx, cy, w, h] - 中心点 + 宽高
    
    目录结构:
    data/
        images/          # 高分辨率原图
        annotations/     # 标注文件
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        crop_size: int = 256,
        use_context: float = 0.2,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.crop_size = crop_size
        self.use_context = use_context
        
        # 加载标注
        self.annotations = self._load_annotations()
        
        # 类别映射
        self.classes = sorted(set(a['class'] for a in self.annotations))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
    def _load_annotations(self) -> List[Dict]:
        """加载标注"""
        ann_file = self.data_dir / "annotations" / f"{self.split}.json"
        
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                data = json.load(f)
            return data.get('annotations', [])
        
        # CSV格式
        ann_file = self.data_dir / "annotations" / f"{self.split}.csv"
        if ann_file.exists():
            df = pd.read_csv(ann_file)
            return df.to_dict('records')
        
        raise FileNotFoundError(f"No annotations found: {ann_file}")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        ann = self.annotations[idx]
        
        # 加载图像
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
        }
    
    def _crop_defect(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """裁剪缺陷区域 (输入为 x, y, w, h 格式)"""
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
        
        # 确保有效裁剪
        if x2 <= x1 or y2 <= y1:
            return np.zeros((int(ph), int(pw), 3), dtype=np.uint8)
        
        return image[y1:y2, x1:x2]
    
    def _extract_position_features(self, bbox: List[float], img_shape: Tuple) -> List[float]:
        """提取位置特征 (输入为 cx, cy, w, h 格式)"""
        from ..features.position import PositionFeatureExtractor
        extractor = PositionFeatureExtractor()
        
        # 将 cx, cy, w, h 转换为 x, y, w, h
        cx, cy, w, h = bbox
        x = cx - w / 2
        y = cy - h / 2
        bbox_xywh = [x, y, w, h]
        
        feat = extractor.extract(bbox_xywh, img_shape)
        
        # 展平 (与之前相同)
        features = [
            feat['cx_norm'], feat['cy_norm'],
            feat['rel_width'], feat['rel_height'], feat['rel_area'],
            feat['min_edge_dist'], feat['min_corner_dist'],
            feat['is_edge'], feat['is_corner'],
            feat['orientation'], feat['density'],
        ]
        return features


def create_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 4):
    """创建数据加载器"""
    from torch.utils.data import DataLoader
    
    train_ds = DefectDataset(data_dir, split="train")
    val_ds = DefectDataset(data_dir, split="val")
    test_ds = DefectDataset(data_dir, split="test")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
