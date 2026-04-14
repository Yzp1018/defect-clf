"""
单元测试模块
"""
import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestLossFunctions(unittest.TestCase):
    """测试损失函数"""
    
    def test_focal_loss_basic(self):
        from utils.losses import FocalLoss
        
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        
        loss = loss_fn(logits, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # 标量
        self.assertGreater(loss.item(), 0)
    
    def test_focal_loss_with_weights(self):
        from utils.losses import FocalLoss
        
        alpha = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        
        loss = loss_fn(logits, targets)
        self.assertGreater(loss.item(), 0)
    
    def test_label_smoothing(self):
        from utils.losses import LabelSmoothingCrossEntropy
        
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        
        loss = loss_fn(logits, targets)
        self.assertGreater(loss.item(), 0)
    
    def test_combined_loss(self):
        from utils.losses import CombinedLoss
        
        loss_fn = CombinedLoss(gamma=2.0, smoothing=0.1, focal_weight=0.8)
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        
        loss = loss_fn(logits, targets)
        self.assertGreater(loss.item(), 0)
    
    def test_build_loss_fn(self):
        from utils.losses import build_loss_fn
        
        for name in ['ce', 'focal', 'smooth_ce', 'combined']:
            loss_fn = build_loss_fn(name=name)
            logits = torch.randn(4, 5)
            targets = torch.randint(0, 5, (4,))
            loss = loss_fn(logits, targets)
            self.assertGreater(loss.item(), 0)


class TestDataAugmentation(unittest.TestCase):
    """测试数据增强"""
    
    def test_augmentor_basic(self):
        from data.augment import DataAugmentor
        
        augmentor = DataAugmentor(img_size=256)
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        augmented = augmentor(img)
        self.assertEqual(augmented.shape, img.shape)
        self.assertEqual(augmented.dtype, np.uint8)
    
    def test_mixup(self):
        """测试 Mixup - 在缺陷分类场景中默认禁用"""
        from data.augment import mixup
        
        img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        mixed, label1, label2, lam = mixup(img1, img2, 0, 1, alpha=0.4)
        
        self.assertEqual(mixed.shape, img1.shape)
        self.assertEqual(label1, 0)
        self.assertEqual(label2, 1)
        # ⚠️ 在缺陷分类场景中，Mixup 默认禁用，返回原图和 lam=1.0
        self.assertEqual(lam, 1.0)
    
    def test_cutmix(self):
        """测试 CutMix - 在缺陷分类场景中默认禁用"""
        from data.augment import cutmix
        
        img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        mixed, label1, label2, lam = cutmix(img1, img2, 0, 1, beta=1.0)
        
        self.assertEqual(mixed.shape, img1.shape)
        # ⚠️ 在缺陷分类场景中，CutMix 默认禁用，返回原图和 lam=1.0
        self.assertEqual(lam, 1.0)


class TestModel(unittest.TestCase):
    """测试模型"""
    
    def test_defect_classifier_forward(self):
        from models.classifier import DefectClassifier
        
        model = DefectClassifier(num_classes=5)
        img = torch.randn(2, 3, 256, 256)
        pos = torch.randn(2, 11)
        
        output = model(img, pos)
        
        self.assertEqual(output.shape, (2, 5))
    
    def test_lightweight_classifier(self):
        from models.classifier import LightweightClassifier
        
        model = LightweightClassifier(num_classes=5)
        img = torch.randn(2, 3, 256, 256)
        pos = torch.randn(2, 11)
        
        output = model(img, pos)
        
        self.assertEqual(output.shape, (2, 5))
    
    def test_model_feature_extraction(self):
        from models.classifier import DefectClassifier
        
        model = DefectClassifier(num_classes=5)
        img = torch.randn(2, 3, 256, 256)
        pos = torch.randn(2, 11)
        
        features = model.get_feature(img, pos)
        
        self.assertIn('img_feat', features)
        self.assertIn('pos_feat', features)
        self.assertIn('fused', features)


class TestPositionFeatures(unittest.TestCase):
    """测试位置特征提取"""
    
    def test_position_extractor(self):
        from features.position import PositionFeatureExtractor
        
        extractor = PositionFeatureExtractor()
        bbox = (100, 100, 50, 50)  # x, y, w, h
        img_shape = (1000, 1000)
        
        features = extractor.extract(bbox, img_shape)
        
        self.assertIn('cx_norm', features)
        self.assertIn('cy_norm', features)
        self.assertIn('rel_area', features)
        self.assertIn('min_edge_dist', features)
        
        # 验证归一化值在合理范围
        self.assertGreaterEqual(features['cx_norm'], 0)
        self.assertLessEqual(features['cx_norm'], 1)
        self.assertGreaterEqual(features['cy_norm'], 0)
        self.assertLessEqual(features['cy_norm'], 1)


class TestVisualization(unittest.TestCase):
    """测试可视化工具"""
    
    def test_grad_cam_creation(self):
        from utils.visualization import GradCAM
        from models.classifier import DefectClassifier
        
        model = DefectClassifier(num_classes=5)
        grad_cam = GradCAM(model, target_layer='cnn')
        
        self.assertIsNotNone(grad_cam.target_layer)
    
    def test_feature_visualizer(self):
        from utils.visualization import FeatureVisualizer
        from models.classifier import DefectClassifier
        
        model = DefectClassifier(num_classes=5)
        visualizer = FeatureVisualizer(model, device='cpu')
        
        self.assertIsNotNone(visualizer.grad_cam)


if __name__ == '__main__':
    unittest.main(verbosity=2)
