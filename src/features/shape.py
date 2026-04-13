"""
形状特征提取器
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple


class ShapeFeatureExtractor:
    """形状特征提取"""
    
    def __init__(self):
        pass
    
    def extract(self, mask: np.ndarray) -> Dict:
        """提取形状特征
        
        Args:
            mask: 二值掩码 [H, W]
        Returns:
            特征字典
        """
        if mask.sum() == 0:
            return self._empty_features()
        
        # 几何特征
        area = int(mask.sum())
        perimeter = self._get_perimeter(mask)
        bounding_box = cv2.boundingRect(mask)
        bbox_area = bounding_box[2] * bounding_box[3]
        
        # 圆形度 (0-1, 1为完美圆形)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # 矩形度 (0-1)
        rectangle = cv2.minAreaRect(cv2.findNonZero(mask.astype(np.uint8)))
        rectangle_area = rectangle[1][0] * rectangle[1][1]
        rectness = area / rectangle_area if rectangle_area > 0 else 0
        
        # 紧凑度
        compactness = perimeter ** 2 / area if area > 0 else 0
        
        # 长宽比
        aspect_ratio = bounding_box[2] / bounding_box[3] if bounding_box[3] > 0 else 0
        
        # 离心率 (拟合椭圆)
        moments = cv2.moments(mask)
        if moments['mu20'] + moments['mu02'] > 0:
            ecc = ((moments['mu20'] - moments['mu02']) ** 2 + 4 * moments['mu11'] ** 2) / \
                  ((moments['mu20'] + moments['mu02']) ** 2)
        else:
            ecc = 0
        
        # Hu矩 (7个不变矩)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        # 轮廓特征
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = contours[0]
            # 傅里叶描述子 (前10个)
            fourier = self._fourier_descriptors(contour, n=10)
            # 轮廓复杂度
            convex_hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(convex_hull)
            solidity = area / hull_area if hull_area > 0 else 0
        else:
            fourier = np.zeros(10)
            solidity = 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'rectness': rectness,
            'compactness': compactness,
            'aspect_ratio': aspect_ratio,
            'eccentricity': ecc,
            'solidity': solidity,
            'hu_moments': hu_moments,
            'fourier': fourier,
        }
    
    def _get_perimeter(self, mask: np.ndarray) -> float:
        """计算周长"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            return cv2.arcLength(contours[0], closed=True)
        return 0
    
    def _fourier_descriptors(self, contour: np.ndarray, n: int = 10) -> np.ndarray:
        """傅里叶描述子"""
        if len(contour) < 3:
            return np.zeros(n)
        
        # 转换为复数形式
        complex_contour = contour.squeeze().astype(np.complex64)
        
        # DFT
        fft = np.fft.fft(complex_contour)
        
        # 取前n个频率分量 (去直流)
        descriptors = np.abs(fft[1:n+1])
        
        # 归一化
        if descriptors.max() > 0:
            descriptors = descriptors / descriptors.max()
        
        return descriptors
    
    def _empty_features(self) -> Dict:
        """空特征"""
        return {
            'area': 0, 'perimeter': 0, 'circularity': 0,
            'rectness': 0, 'compactness': 0, 'aspect_ratio': 0,
            'eccentricity': 0, 'solidity': 0,
            'hu_moments': np.zeros(7),
            'fourier': np.zeros(10),
        }


class TextureFeatureExtractor:
    """纹理特征提取"""
    
    def __init__(self):
        pass
    
    def extract(self, image: np.ndarray) -> Dict:
        """提取纹理特征
        
        Args:
            image: 灰度图 [H, W]
        Returns:
            特征字典
        """
        # LBP (Local Binary Pattern)
        lbp = self._lbp(image)
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256), density=True)
        
        # GLCM (Gray Level Co-occurrence Matrix)
        glcm = self._glcm(image)
        
        # 梯度特征
        grad_mag, grad_dir = self._gradient(image)
        
        return {
            'lbp_hist': lbp_hist,
            'glcm_contrast': glcm[0],
            'glcm_homogeneity': glcm[1],
            'glcm_energy': glcm[2],
            'grad_mean': grad_mag.mean(),
            'grad_std': grad_mag.std(),
        }
    
    def _lbp(self, image: np.ndarray, radius: int = 1, points: int = 8) -> np.ndarray:
        """简化LBP"""
        # 简化实现
        h, w = image.shape
        lbp = np.zeros((h-2*radius, w-2*radius), dtype=np.uint8)
        
        for i in range(radius, h-radius):
            for j in range(radius, w-radius):
                center = image[i, j]
                code = 0
                for k in range(points):
                    angle = 2 * np.pi * k / points
                    x = int(j + radius * np.cos(angle))
                    y = int(i - radius * np.sin(angle))
                    if image[y, x] >= center:
                        code |= (1 << k)
                lbp[i-radius, j-radius] = code
        
        return lbp
    
    def _glcm(self, image: np.ndarray, distances=[1, 2], angles=[0, np.pi/4, np.pi/2]) -> Tuple:
        """简化GLCM"""
        # 量化到8级
        quant = (image // 32).astype(np.uint8)
        
        # 简化: 只计算一个方向
        h, w = quant.shape
        glcm = np.zeros((8, 8))
        
        for i in range(h-1):
            for j in range(w-1):
                glcm[quant[i, j], quant[i+1, j]] += 1
        
        # 归一化
        glcm = glcm / glcm.sum() if glcm.sum() > 0 else glcm
        
        # 特征
        contrast = np.sum(glcm * (np.abs(np.arange(8)[:, None] - np.arange(8)) ** 2))
        homogeneity = np.sum(glcm / (1 + np.abs(np.arange(8)[:, None] - np.arange(8))))
        energy = np.sum(glcm ** 2)
        
        return contrast, homogeneity, energy
    
    def _gradient(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """梯度"""
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_dir = np.arctan2(grad_y, grad_x)
        return grad_mag, grad_dir
