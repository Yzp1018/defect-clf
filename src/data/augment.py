"""
数据增强模块 - 支持多种增强策略
"""
import cv2
import numpy as np
import random
from typing import Tuple, Optional, List


class DataAugmentor:
    """数据增强器
    
    支持的增强：
    - 几何变换：翻转、旋转、缩放
    - 颜色变换：亮度、对比度、饱和度、色调
    - 噪声：高斯噪声、椒盐噪声
    - 模糊：高斯模糊、运动模糊
    - 遮挡：Cutout, Mixup, CutMix
    """
    
    def __init__(
        self,
        img_size: int = 256,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.3,
        rotate_prob: float = 0.3,
        rotate_range: Tuple[float, float] = (-15, 15),
        color_jitter_prob: float = 0.5,
        noise_prob: float = 0.2,
        blur_prob: float = 0.2,
        cutout_prob: float = 0.3,
        cutout_ratio: float = 0.1,
    ):
        self.img_size = img_size
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate_prob = rotate_prob
        self.rotate_range = rotate_range
        self.color_jitter_prob = color_jitter_prob
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob
        self.cutout_prob = cutout_prob
        self.cutout_ratio = cutout_ratio
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """应用数据增强
        
        Args:
            img: [H, W, 3] RGB 图像
        Returns:
            增强后的图像
        """
        # 几何变换
        if random.random() < self.hflip_prob:
            img = cv2.flip(img, 1)  # 水平翻转
        
        if random.random() < self.vflip_prob:
            img = cv2.flip(img, 0)  # 垂直翻转
        
        if random.random() < self.rotate_prob:
            angle = random.uniform(*self.rotate_range)
            img = self._rotate(img, angle)
        
        # 颜色变换
        if random.random() < self.color_jitter_prob:
            img = self._color_jitter(img)
        
        # 噪声
        if random.random() < self.noise_prob:
            img = self._add_noise(img)
        
        # 模糊
        if random.random() < self.blur_prob:
            img = self._add_blur(img)
        
        # 遮挡
        if random.random() < self.cutout_prob:
            img = self._cutout(img)
        
        return img
    
    def _rotate(self, img: np.ndarray, angle: float) -> np.ndarray:
        """旋转图像"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        """颜色抖动"""
        img = img.astype(np.float32)
        
        # 亮度
        alpha = random.uniform(0.7, 1.3)
        img = img * alpha
        img = np.clip(img, 0, 255)
        
        # 对比度
        alpha = random.uniform(0.7, 1.3)
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_mean = gray.mean()
        img = (img - gray_mean) * alpha + gray_mean
        img = np.clip(img, 0, 255)
        
        # 饱和度
        alpha = random.uniform(0.7, 1.3)
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * alpha
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return img.astype(np.uint8)
    
    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img = img + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    
    def _add_blur(self, img: np.ndarray) -> np.ndarray:
        """添加模糊"""
        ksize = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        return img
    
    def _cutout(self, img: np.ndarray) -> np.ndarray:
        """Cutout 随机遮挡"""
        h, w = img.shape[:2]
        mask_h = int(h * self.cutout_ratio)
        mask_w = int(w * self.cutout_ratio)
        
        y = random.randint(0, h - mask_h)
        x = random.randint(0, w - mask_w)
        
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
    
    Args:
        img1, img2: 两张图像
        label1, label2: 对应标签
        alpha: Beta 分布参数
    Returns:
        mixed_img, label1, label2, lam
    """
    lam = np.random.beta(alpha, alpha)
    mixed_img = lam * img1 + (1 - lam) * img2
    return mixed_img.astype(np.uint8), label1, label2, lam


def cutmix(
    img1: np.ndarray,
    img2: np.ndarray,
    label1: int,
    label2: int,
    beta: float = 1.0,
) -> Tuple[np.ndarray, int, int, float]:
    """CutMix 数据增强
    
    Args:
        img1, img2: 两张图像
        label1, label2: 对应标签
        beta: Beta 分布参数
    Returns:
        mixed_img, label1, label2, lam
    """
    img_size = img1.shape[0]
    lam = np.random.beta(beta, beta)
    
    # 计算裁剪区域
    ratio = np.sqrt(1 - lam)
    cut_h = int(img_size * ratio)
    cut_w = int(img_size * ratio)
    
    cx = np.random.randint(0, img_size)
    cy = np.random.randint(0, img_size)
    
    x1 = np.clip(cx - cut_w // 2, 0, img_size)
    y1 = np.clip(cy - cut_h // 2, 0, img_size)
    x2 = np.clip(cx + cut_w // 2, 0, img_size)
    y2 = np.clip(cy + cut_h // 2, 0, img_size)
    
    # 应用 CutMix
    mixed_img = img1.copy()
    mixed_img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    
    # 计算实际 lam
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (img_size * img_size)
    
    return mixed_img, label1, label2, lam


class AugmentWithMixup:
    """带 Mixup 的增强器"""
    
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
        img1_aug = self.augmentor(img1)
        img2_aug = self.augmentor(img2)
        return mixup(img1_aug, img2_aug, label1, label2, self.mixup_alpha)
