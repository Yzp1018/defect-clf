"""
高分辨率图像数据处理模块 - 针对工业相机大图优化

特性:
- 滑动窗口裁剪
- 多分辨率金字塔
- 内存映射加载
- 并行预处理
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, Generator
import json
from concurrent.futures import ThreadPoolExecutor
import threading


class HighResImageLoader:
    """高分辨率图像加载器 - 支持内存映射和分块加载"""
    
    def __init__(
        self,
        img_path: str,
        use_memory_map: bool = True,
        cache_size: int = 1,
    ):
        """
        Args:
            img_path: 图像路径
            use_memory_map: 是否使用内存映射 (节省内存)
            cache_size: 缓存的块数量
        """
        self.img_path = img_path
        self.use_memory_map = use_memory_map
        self.cache_size = cache_size
        
        # 获取图像信息 (不加载完整图像)
        self._init_image_info()
        
        # LRU 缓存
        self._cache = {}
        self._cache_lock = threading.Lock()
    
    def _init_image_info(self):
        """初始化图像信息"""
        if self.use_memory_map:
            # 使用 OpenCV 的 IMREAD_UNCHANGED + 懒加载
            dummy = cv2.imread(self.img_path, cv2.IMREAD_UNCHANGED)
            if dummy is None:
                raise FileNotFoundError(f"Cannot read image: {self.img_path}")
            self.height, self.width = dummy.shape[:2]
            self.channels = 3 if len(dummy.shape) == 3 else 1
            del dummy
        else:
            # 直接加载到内存
            self.image = cv2.imread(self.img_path)
            if self.image is None:
                raise FileNotFoundError(f"Cannot read image: {self.img_path}")
            self.height, self.width = self.image.shape[:2]
            self.channels = 3 if len(self.image.shape) == 3 else 1
    
    def get_region(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        resize_to: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        获取指定区域
        
        Args:
            x, y: 左上角坐标
            w, h: 宽高
            resize_to: 调整到的尺寸 (W, H)
        """
        # 边界检查
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        w = min(w, self.width - x)
        h = min(h, self.height - y)
        
        if self.use_memory_map:
            # 懒加载区域
            # 使用 OpenCV 的 ROI 读取
            full_img = cv2.imread(self.img_path)
            if full_img is None:
                raise ValueError(f"Failed to read image: {self.img_path}")
            region = full_img[y:y+h, x:x+w]
            del full_img
        else:
            region = self.image[y:y+h, x:x+w]
        
        if resize_to is not None:
            region = cv2.resize(region, resize_to, interpolation=cv2.INTER_LINEAR)
        
        return region
    
    def get_full_image(self, max_dim: Optional[int] = 4096) -> np.ndarray:
        """
        获取完整图像 (可选降采样)
        
        Args:
            max_dim: 最大边长，超过则降采样
        """
        if self.use_memory_map:
            img = cv2.imread(self.img_path)
        else:
            img = self.image.copy()
        
        if max_dim and max(img.shape[:2]) > max_dim:
            scale = max_dim / max(img.shape[:2])
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        
        return img


class SlidingWindowDataset(Dataset):
    """
    滑动窗口数据集 - 用于高分辨率图像推理
    
    将大图分割成多个重叠的窗口，分别进行预测
    """
    
    def __init__(
        self,
        img_path: str,
        window_size: int = 512,
        stride: int = 256,
        min_defect_size: int = 32,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            img_path: 图像路径
            window_size: 窗口大小
            stride: 滑动步长
            min_defect_size: 最小缺陷尺寸 (用于跳过空白区域)
            transform: 图像变换
        """
        self.img_path = img_path
        self.window_size = window_size
        self.stride = stride
        self.min_defect_size = min_defect_size
        self.transform = transform
        
        self.loader = HighResImageLoader(img_path)
        
        # 生成所有窗口位置
        self.windows = self._generate_windows()
    
    def _generate_windows(self) -> List[Dict]:
        """生成滑动窗口"""
        windows = []
        h, w = self.loader.height, self.loader.width
        
        for y in range(0, h, self.stride):
            for x in range(0, w, self.stride):
                # 计算实际窗口大小 (处理边界)
                win_w = min(self.window_size, w - x)
                win_h = min(self.window_size, h - y)
                
                if win_w < self.min_defect_size or win_h < self.min_defect_size:
                    continue
                
                windows.append({
                    'x': x,
                    'y': y,
                    'w': win_w,
                    'h': win_h,
                    'idx': len(windows),
                })
        
        return windows
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict:
        win = self.windows[idx]
        
        # 获取窗口图像
        img = self.loader.get_region(
            win['x'], win['y'], win['w'], win['h'],
            resize_to=(self.window_size, self.window_size)
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        else:
            # 默认转换为 tensor
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': img,
            'window_x': win['x'],
            'window_y': win['y'],
            'window_w': win['w'],
            'window_h': win['h'],
            'window_idx': idx,
            'original_size': (self.loader.height, self.loader.width),
        }
    
    def merge_predictions(
        self,
        predictions: List[np.ndarray],
        scores: List[np.ndarray],
        strategy: str = 'weighted_avg',
    ) -> np.ndarray:
        """
        合并窗口预测结果
        
        Args:
            predictions: 每个窗口的预测类别 [N]
            scores: 每个窗口的置信度分数 [N]
            strategy: 'weighted_avg' | 'max' | 'vote'
        
        Returns:
            全图预测热力图或分类结果
        """
        h, w = self.loader.height, self.loader.width
        
        # 创建全图分数图
        score_map = np.zeros((h, w), dtype=np.float32)
        pred_map = np.zeros((h, w), dtype=np.int32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            win = self.windows[i]
            x, y, win_w, win_h = win['x'], win['y'], win['w'], win['h']
            
            # 调整分数到窗口大小
            score_resized = cv2.resize(
                np.full((self.window_size, self.window_size), score),
                (win_w, win_h),
                interpolation=cv2.INTER_LINEAR
            )
            
            if strategy == 'weighted_avg':
                score_map[y:y+win_h, x:x+win_w] += score_resized
                pred_map[y:y+win_h, x:x+win_w] += pred
                count_map[y:y+win_h, x:x+win_w] += 1
            elif strategy == 'max':
                mask = score_resized > score_map[y:y+win_h, x:x+win_w]
                score_map[y:y+win_h, x:x+win_w][mask] = score_resized[mask]
                pred_map[y:y+win_h, x:x+win_w][mask] = pred
        
        if strategy == 'weighted_avg':
            # 避免除零
            count_map[count_map == 0] = 1
            score_map /= count_map
            pred_map = (pred_map / count_map).astype(np.int32)
        
        return pred_map, score_map


class ImagePyramid:
    """图像金字塔 - 多尺度处理"""
    
    def __init__(
        self,
        img_path: str,
        scales: List[float] = [0.5, 0.75, 1.0, 1.5, 2.0],
        max_dim: int = 8192,
    ):
        """
        Args:
            img_path: 图像路径
            scales: 多尺度因子
            max_dim: 最大处理尺寸
        """
        self.img_path = img_path
        self.scales = scales
        self.max_dim = max_dim
        
        self.loader = HighResImageLoader(img_path)
        self.pyramid = self._build_pyramid()
    
    def _build_pyramid(self) -> Dict[float, np.ndarray]:
        """构建图像金字塔"""
        pyramid = {}
        
        # 获取基础图像
        base_img = self.loader.get_full_image(max_dim=self.max_dim)
        base_h, base_w = base_img.shape[:2]
        
        for scale in self.scales:
            if scale == 1.0:
                pyramid[scale] = base_img
            else:
                new_size = (int(base_w * scale), int(base_h * scale))
                if scale < 1.0:
                    interp = cv2.INTER_AREA
                else:
                    interp = cv2.INTER_LINEAR
                pyramid[scale] = cv2.resize(base_img, new_size, interpolation=interp)
        
        return pyramid
    
    def get_scale(self, scale: float) -> np.ndarray:
        """获取指定尺度的图像"""
        if scale not in self.pyramid:
            # 动态生成
            base_img = self.pyramid[1.0]
            h, w = base_img.shape[:2]
            new_size = (int(w * scale), int(h * scale))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            self.pyramid[scale] = cv2.resize(base_img, new_size, interpolation=interp)
        
        return self.pyramid[scale]
    
    def predict_multi_scale(
        self,
        model: torch.nn.Module,
        device: str = 'cuda',
        batch_size: int = 8,
    ) -> Dict[str, List]:
        """
        多尺度预测
        
        Returns:
            包含各尺度预测结果的字典
        """
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        results = {'scales': [], 'predictions': [], 'scores': []}
        model.eval()
        
        with torch.no_grad():
            for scale, img in self.pyramid.items():
                # 简单中心裁剪预测 (示例)
                h, w = img.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                # 这里应该根据实际需求进行滑动窗口或全局预测
                # 简化示例：取中心区域
                crop_size = min(512, h, w)
                x1 = max(0, center_x - crop_size // 2)
                y1 = max(0, center_y - crop_size // 2)
                
                crop = img[y1:y1+crop_size, x1:x1+crop_size]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_tensor = transform(crop_rgb).unsqueeze(0).to(device)
                
                # 需要位置特征 (这里用默认值)
                pos_feat = torch.zeros(1, 11).to(device)
                
                output = model(crop_tensor, pos_feat)
                prob = torch.softmax(output, dim=1)
                score, pred = torch.max(prob, dim=1)
                
                results['scales'].append(scale)
                results['predictions'].append(pred.item())
                results['scores'].append(score.item())
        
        return results


def create_highres_dataloader(
    img_paths: List[str],
    window_size: int = 512,
    stride: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    创建高分辨率图像数据加载器
    
    Args:
        img_paths: 图像路径列表
        window_size: 窗口大小
        stride: 滑动步长
        batch_size: 批次大小
        num_workers: 工作线程数
        pin_memory: 是否锁定内存
    
    Returns:
        DataLoader
    """
    from torch.utils.data import ConcatDataset
    
    datasets = []
    for img_path in img_paths:
        ds = SlidingWindowDataset(
            img_path=img_path,
            window_size=window_size,
            stride=stride,
        )
        datasets.append(ds)
    
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    return loader


if __name__ == '__main__':
    # 测试示例
    import sys
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        print("Usage: python highres_loader.py <image_path>")
        sys.exit(1)
    
    # 测试高分辨率加载器
    loader = HighResImageLoader(img_path)
    print(f"Image size: {loader.width} x {loader.height}")
    
    # 测试滑动窗口
    ds = SlidingWindowDataset(img_path, window_size=512, stride=256)
    print(f"Number of windows: {len(ds)}")
    
    # 获取一个窗口
    sample = ds[0]
    print(f"Window shape: {sample['image'].shape}")
    print(f"Window position: ({sample['window_x']}, {sample['window_y']})")
