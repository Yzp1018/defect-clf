"""
推理脚本
"""
import torch
import cv2
import numpy as np
from pathlib import Path

from features.shape import ShapeFeatureExtractor
from features.position import PositionFeatureExtractor
from models.classifier import build_model


class DefectClassifierInference:
    """缺陷分类推理"""
    
    def __init__(self, model_path: str, config: dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # 模型
        self.model = build_model(
            config.get('model_name', 'defect_classifier'),
            num_classes=config.get('num_classes', 5),
            backbone=config.get('backbone', 'efficientnet_b0'),
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        
        # 特征提取器
        self.shape_extractor = ShapeFeatureExtractor()
        self.pos_extractor = PositionFeatureExtractor()
        
        # 类别
        self.classes = config.get('classes', ['scratch', 'dent', 'spot', 'crack', 'bubble'])
    
    @torch.no_grad()
    def predict(self, image_path: str, bbox: list) -> dict:
        """预测单个缺陷
        
        Args:
            image_path: 图像路径
            bbox: [cx, cy, w, h] 中心点 + 宽高
        """
        # 加载图像
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # bbox: [cx, cy, w, h] → 转换为 [x, y, w, h] 用于裁剪
        cx, cy, w, h = bbox
        bbox_xywh = [cx - w/2, cy - h/2, w, h]
        
        # 裁剪ROI
        crop = self._crop_roi(img, bbox_xywh)
        crop = cv2.resize(crop, (self.config.get('crop_size', 256), self.config.get('crop_size', 256)))
        
        # 预处理
        img_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # 位置特征 (使用 cx, cy, w, h 格式)
        img_h, img_w = img.shape[:2]
        pos_feat = self.pos_extractor.extract(bbox_xywh, (img_h, img_w))
        pos_values = [pos_feat['cx_norm'], pos_feat['cy_norm'], pos_feat['rel_width'],
                      pos_feat['rel_height'], pos_feat['rel_area'], pos_feat['min_edge_dist'],
                      pos_feat['min_corner_dist'], pos_feat['is_edge'], pos_feat['is_corner'],
                      pos_feat['orientation'], pos_feat['density']]
        pos_tensor = torch.tensor(pos_values, dtype=torch.float).unsqueeze(0).to(self.device)
        
        # 推理
        output = self.model(img_tensor, pos_tensor)
        probs = torch.softmax(output, dim=1)
        
        pred_idx = output.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
        
        return {
            'class': self.classes[pred_idx],
            'confidence': confidence,
            'all_probs': {c: p.item() for c, p in zip(self.classes, probs[0])},
        }
    
    def _crop_roi(self, image: np.ndarray, bbox: list) -> np.ndarray:
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        context = self.config.get('context', 0.2)
        pw, ph = w * (1 + context), h * (1 + context)
        
        h_img, w_img = image.shape[:2]
        x1 = max(0, int(cx - pw/2))
        y1 = max(0, int(cy - ph/2))
        x2 = min(w_img, int(cx + pw/2))
        y2 = min(h_img, int(cy + ph/2))
        
        return image[y1:y2, x1:x2]
    
    def batch_predict(self, image_path: str, bboxes: list) -> list:
        """批量预测"""
        return [self.predict(image_path, bbox) for bbox in bboxes]


if __name__ == '__main__':
    # 示例
    config = {
        'model_name': 'defect_classifier',
        'num_classes': 5,
        'backbone': 'efficientnet_b0',
        'crop_size': 256,
        'context': 0.2,
        'classes': ['scratch', 'dent', 'spot', 'crack', 'bubble'],
    }
    
    # 需要先训练模型
    print("Usage: python inference.py --model_path outputs/best_model.pth --image img.jpg --bbox 2500,1200,80,12")
