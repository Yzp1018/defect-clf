"""
数据增强模块 - 针对工业缺陷分类的特殊增强策略

注意：本场景所有样本都是缺陷样本，目标是缺陷类型分类
增强原则：
- 保持缺陷几何真实性（不改变缺陷形态逻辑）
- 保持缺陷尺度比例（不改变相对大小）
- 重点增强：噪声、模糊、光照变化、Cutout
- 谨慎使用：几何变换（仅当产品对称时）
- 禁用：Mixup/CutMix（会创造虚假缺陷样本）
"""
import cv2
import numpy as np
import random
from typing import Tuple, Optional, List


class DataAugmentor:
    """工业缺陷分类数据增强器
    
    支持的增强：
    - 光度变换：亮度、对比度、饱和度（模拟光照变化）
    - 噪声：高斯噪声、椒盐噪声（模拟相机噪声）
    - 模糊：高斯模糊、运动模糊（模拟相机抖动/对焦不准）
    - 遮挡：Cutout（强制模型关注局部特征）
    - 几何变换：仅在product_symmetric=True时启用
    """
    
    def __init__(
        self,
        img_size: int = 256,
        # 几何变换（默认禁用，除非产品对称）
        hflip_prob: float = 0.0,  # 水平翻转概率
        vflip_prob: float = 0.0,  # 垂直翻转概率
        rotate_prob: float = 0.0,  # 旋转概率
        rotate_range: Tuple[float, float] = (-5, 5),  # 小角度旋转
        product_symmetric: bool = False,  # 产品是否对称
        # 光度变换
        color_jitter_prob: float = 0.5,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        # 噪声
        noise_prob: float = 0.4,  # 提高噪声概率
        noise_std: float = 15,
        # 模糊
        blur_prob: float = 0.3,  # 提高模糊概率
        # 遮挡
        cutout_prob: float = 0.4,  # 提高Cutout概率
        cutout_ratio: float = 0.15,
        cutout_num: int = 2,  # 多个遮挡区域
    ):
        self.img_size = img_size
        self.product_symmetric = product_symmetric
        
        # 仅在产品对称时启用几何变换
        if product_symmetric:
            self.hflip_prob = hflip_prob
            self.vflip_prob = vflip_prob
            self.rotate_prob = rotate_prob
        else:
            self.hflip_prob = 0.0
            self.vflip_prob = 0.0
            self.rotate_prob = 0.0
        
        self.rotate_range = rotate_range
        self.color_jitter_prob = color_jitter_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.blur_prob = blur_prob
        self.cutout_prob = cutout_prob
        self.cutout_ratio = cutout_ratio
        self.cutout_num = cutout_num
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """应用数据增强
        
        Args:
            img: [H, W, 3] RGB 图像
        Returns:
            增强后的图像
            
        增强顺序：几何变换 → 光度变换 → 噪声 → 模糊 → 遮挡
        """
        # 几何变换（仅当产品对称时）
        if self.product_symmetric:
            if random.random() < self.hflip_prob:
                img = cv2.flip(img, 1)  # 水平翻转
            
            if random.random() < self.vflip_prob:
                img = cv2.flip(img, 0)  # 垂直翻转
            
            if random.random() < self.rotate_prob:
                angle = random.uniform(*self.rotate_range)
                img = self._rotate(img, angle)
        
        # 颜色/光度变换
        if random.random() < self.color_jitter_prob:
            img = self._color_jitter(img)
        
        # 噪声（模拟相机噪声）
        if random.random() < self.noise_prob:
            img = self._add_noise(img)
        
        # 模糊（模拟相机抖动/对焦不准）
        if random.random() < self.blur_prob:
            img = self._add_blur(img)
        
        # 遮挡（强制模型关注局部特征）
        if random.random() < self.cutout_prob:
            img = self._cutout(img)
        
        return img
    
    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        """颜色/光度抖动 - 模拟光照变化"""
        img = img.astype(np.float32)
        
        # 亮度调整
        alpha = random.uniform(*self.brightness_range)
        img = img * alpha
        img = np.clip(img, 0, 255)
        
        # 对比度调整
        alpha = random.uniform(*self.contrast_range)
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_mean = gray.mean()
        img = (img - gray_mean) * alpha + gray_mean
        img = np.clip(img, 0, 255)
        
        # 饱和度调整（适度，避免过度失真）
        alpha = random.uniform(0.8, 1.2)
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * alpha
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return img.astype(np.uint8)
    
    def _rotate(self, img: np.ndarray, angle: float) -> np.ndarray:
        """旋转图像（仅当产品对称时使用）"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        """添加高斯噪声 - 模拟相机传感器噪声"""
        noise = np.random.normal(0, self.noise_std, img.shape).astype(np.float32)
        img = img + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def _add_blur(self, img: np.ndarray) -> np.ndarray:
        """添加模糊 - 模拟相机抖动或对焦不准"""
        ksize = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        return img
    
    def _cutout(self, img: np.ndarray) -> np.ndarray:
        """Cutout 随机遮挡 - 强制模型关注局部特征
        
        使用多个小遮挡区域，而不是一个大区域
        """
        h, w = img.shape[:2]
        mask_h = int(h * self.cutout_ratio)
        mask_w = int(w * self.cutout_ratio)
        
        for _ in range(self.cutout_num):
            y = random.randint(0, max(0, h - mask_h))
            x = random.randint(0, max(0, w - mask_w))
            img[y:y+mask_h, x:x+mask_w] = 0  # 黑色遮挡
        
        return img


def mixup(
    img1: np.ndarray,
    img2: np.ndarray,
    label1: int,
    label2: int,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, int, int, float]:
    """Mixup 数据增强
    
    ⚠️ 警告：在缺陷分类场景中不推荐使用 Mixup
    原因：混合两个缺陷样本会创造虚假的缺陷形态，不符合物理现实
    
    Args:
        img1, img2: 两张图像
        label1, label2: 对应标签
        alpha: Beta 分布参数
    Returns:
        mixed_img, label1, label2, lam
        
    仅在以下场景考虑使用：
    - 缺陷类型之间可以物理共存（如划痕 + 污点）
    - 作为强正则化手段且验证集表现提升
    """
    # 默认禁用，返回原图
    return img1, label1, label2, 1.0


def cutmix(
    img1: np.ndarray,
    img2: np.ndarray,
    label1: int,
    label2: int,
    beta: float = 1.0,
) -> Tuple[np.ndarray, int, int, float]:
    """CutMix 数据增强
    
    ⚠️ 警告：在缺陷分类场景中不推荐使用 CutMix
    原因：从一个缺陷样本裁剪到另一个会创造虚假的缺陷位置和形态
    
    仅在以下场景考虑使用：
    - 缺陷位置不影响分类（纯纹理/形状分类）
    - 作为强正则化手段且验证集表现提升
    """
    # 默认禁用，返回原图
    return img1, label1, label2, 1.0


class AugmentWithMixup:
    """带 Mixup 的增强器
    
    ⚠️ 警告：在缺陷分类场景中不推荐使用
    仅作为实验性功能保留
    """
    
    def __init__(self, augmentor: DataAugmentor, mixup_alpha: float = 0.4):
        self.augmentor = augmentor
        self.mixup_alpha = mixup_alpha
    
    def __call__(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        label1: int,
        label2: int,
    ) -> Tuple[np.ndarray, int, int, float]:
        # 仅应用基础增强，禁用 Mixup
        img1_aug = self.augmentor(img1)
        return img1_aug, label1, label2, 1.0


# 导出工业场景推荐的增强配置
INDUSTRIAL_DEFAULT_CONFIG = {
    "product_symmetric": False,  # 大多数工业产品不对称
    "hflip_prob": 0.0,
    "vflip_prob": 0.0,
    "rotate_prob": 0.0,
    "color_jitter_prob": 0.5,
    "brightness_range": (0.8, 1.2),
    "contrast_range": (0.8, 1.2),
    "noise_prob": 0.4,
    "noise_std": 15,
    "blur_prob": 0.3,
    "cutout_prob": 0.4,
    "cutout_ratio": 0.15,
    "cutout_num": 2,
}
