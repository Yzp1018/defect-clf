"""
高分辨率图像处理测试模块
"""
import unittest
import numpy as np
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestHighResLoader(unittest.TestCase):
    """测试高分辨率图像加载器"""
    
    def test_highres_loader_creation(self):
        """测试加载器创建 (使用虚拟图像)"""
        from data.highres_loader import HighResImageLoader
        
        # 创建一个测试图像
        import cv2
        test_img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        test_path = '/tmp/test_large_image.jpg'
        cv2.imwrite(test_path, test_img)
        
        loader = HighResImageLoader(test_path, use_memory_map=True)
        
        self.assertEqual(loader.width, 1000)
        self.assertEqual(loader.height, 1000)
    
    def test_get_region(self):
        """测试区域裁剪"""
        from data.highres_loader import HighResImageLoader
        import cv2
        
        # 创建测试图像
        test_img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        test_path = '/tmp/test_region.jpg'
        cv2.imwrite(test_path, test_img)
        
        loader = HighResImageLoader(test_path)
        
        # 裁剪区域
        region = loader.get_region(x=100, y=100, w=256, h=256)
        
        self.assertEqual(region.shape, (256, 256, 3))
    
    def test_get_region_resize(self):
        """测试区域裁剪并调整大小"""
        from data.highres_loader import HighResImageLoader
        import cv2
        
        test_img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        test_path = '/tmp/test_resize.jpg'
        cv2.imwrite(test_path, test_img)
        
        loader = HighResImageLoader(test_path)
        
        # 裁剪并调整大小
        region = loader.get_region(x=100, y=100, w=200, h=200, resize_to=(256, 256))
        
        self.assertEqual(region.shape, (256, 256, 3))


class TestSlidingWindowDataset(unittest.TestCase):
    """测试滑动窗口数据集"""
    
    def test_window_generation(self):
        """测试窗口生成"""
        from data.highres_loader import SlidingWindowDataset
        import cv2
        
        # 创建测试图像
        test_img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        test_path = '/tmp/test_sliding.jpg'
        cv2.imwrite(test_path, test_img)
        
        ds = SlidingWindowDataset(
            img_path=test_path,
            window_size=256,
            stride=128,
        )
        
        # 验证窗口数量
        self.assertGreater(len(ds), 0)
        
        # 验证第一个窗口
        sample = ds[0]
        self.assertIn('image', sample)
        self.assertIn('window_x', sample)
        self.assertIn('window_y', sample)
        self.assertEqual(sample['image'].shape[1:], (256, 256))
    
    def test_window_coverage(self):
        """测试窗口覆盖范围"""
        from data.highres_loader import SlidingWindowDataset
        import cv2
        
        test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_path = '/tmp/test_coverage.jpg'
        cv2.imwrite(test_path, test_img)
        
        ds = SlidingWindowDataset(
            img_path=test_path,
            window_size=256,
            stride=256,  # 无重叠
        )
        
        # 应该生成 4 个窗口 (2x2)
        self.assertEqual(len(ds), 4)


class TestPredictionMerging(unittest.TestCase):
    """测试预测结果合并"""
    
    def test_weighted_avg_merge(self):
        """测试加权平均合并"""
        from data.highres_loader import SlidingWindowDataset
        import cv2
        
        test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_path = '/tmp/test_merge.jpg'
        cv2.imwrite(test_path, test_img)
        
        ds = SlidingWindowDataset(
            img_path=test_path,
            window_size=256,
            stride=128,
        )
        
        # 模拟预测结果
        n_windows = len(ds)
        predictions = np.random.randint(0, 5, n_windows)
        scores = np.random.uniform(0.5, 1.0, n_windows)
        
        # 合并
        pred_map, score_map = ds.merge_predictions(predictions, scores, strategy='weighted_avg')
        
        self.assertEqual(pred_map.shape, (512, 512))
        self.assertEqual(score_map.shape, (512, 512))
        self.assertGreater(score_map.max(), 0)
    
    def test_max_merge(self):
        """测试最大值合并"""
        from data.highres_loader import SlidingWindowDataset
        import cv2
        
        test_img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        test_path = '/tmp/test_max_merge.jpg'
        cv2.imwrite(test_path, test_img)
        
        ds = SlidingWindowDataset(
            img_path=test_path,
            window_size=256,
            stride=128,
        )
        
        n_windows = len(ds)
        predictions = np.random.randint(0, 5, n_windows)
        scores = np.random.uniform(0.5, 1.0, n_windows)
        
        pred_map, score_map = ds.merge_predictions(predictions, scores, strategy='max')
        
        self.assertEqual(pred_map.shape, (512, 512))
        self.assertGreater(score_map.max(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
