"""
位置特征提取器
"""
import numpy as np
from typing import Dict, Tuple


class PositionFeatureExtractor:
    """位置特征提取"""
    
    REGIONS = ['center', 'top', 'bottom', 'left', 'right', 
               'top_left', 'top_right', 'bottom_left', 'bottom_right']
    
    def __init__(self, img_size: Tuple[int, int] = None):
        self.img_size = img_size
    
    def extract(self, bbox: Tuple[int, int, int, int], 
                img_shape: Tuple[int, int] = None) -> Dict:
        """提取位置特征
        
        Args:
            bbox: (x, y, w, h) 缺陷边界框
            img_shape: (H, W) 图像尺寸
        Returns:
            特征字典
        """
        x, y, w, h = bbox
        H, W = img_shape if img_shape else self.img_size
        
        if H is None or W is None:
            raise ValueError("img_size or img_shape must be provided")
        
        # 归一化坐标
        cx = (x + w/2) / W  # 中心X (0-1)
        cy = (y + h/2) / H  # 中心Y (0-1)
        
        # 相对尺寸
        rel_w = w / W
        rel_h = h / H
        rel_area = rel_w * rel_h
        
        # 到各边缘的距离 (归一化)
        dist_left = x / W
        dist_right = (W - x - w) / W
        dist_top = y / H
        dist_bottom = (H - y - h) / H
        min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)
        
        # 对角距离
        corner_dists = [
            np.sqrt((x/W)**2 + (y/H)**2),  # 左上
            np.sqrt(((W-x-w)/W)**2 + (y/H)**2),  # 右上
            np.sqrt((x/W)**2 + ((H-y-h)/H)**2),  # 左下
            np.sqrt(((W-x-w)/W)**2 + ((H-y-h)/H)**2),  # 右下
        ]
        min_corner_dist = min(corner_dists)
        
        # 区域分类
        region = self._get_region(cx, cy)
        
        # 边缘检测 (是否靠近边缘)
        is_edge = min_edge_dist < 0.1
        is_corner = min_corner_dist < 0.15
        
        # 方向 (水平/垂直/对角)
        orientation = self._get_orientation(w, h)
        
        # 密度分布 (缺陷在ROI内的分布)
        density = self._get_density(bbox, img_shape)
        
        return {
            'cx_norm': cx,
            'cy_norm': cy,
            'rel_width': rel_w,
            'rel_height': rel_h,
            'rel_area': rel_area,
            'dist_left': dist_left,
            'dist_right': dist_right,
            'dist_top': dist_top,
            'dist_bottom': dist_bottom,
            'min_edge_dist': min_edge_dist,
            'min_corner_dist': min_corner_dist,
            'region': self._encode_region(region),
            'is_edge': float(is_edge),
            'is_corner': float(is_corner),
            'orientation': orientation,
            'density': density,
        }
    
    def _get_region(self, cx: float, cy: float) -> str:
        """区域分类"""
        # 三分法
        x_zone = 'left' if cx < 0.33 else ('right' if cx > 0.66 else 'center')
        y_zone = 'top' if cy < 0.33 else ('bottom' if cy > 0.66 else 'center')
        
        if x_zone == 'center' and y_zone == 'center':
            return 'center'
        return f"{y_zone}_{x_zone}"
    
    def _encode_region(self, region: str) -> np.ndarray:
        """区域编码为one-hot"""
        encoding = np.zeros(len(self.REGIONS))
        if region in self.REGIONS:
            encoding[self.REGIONS.index(region)] = 1
        return encoding
    
    def _get_orientation(self, w: int, h: int) -> int:
        """方向分类: 0=水平, 1=垂直, 2=方形"""
        ratio = w / h if h > 0 else 1
        if ratio > 1.5:
            return 0  # 水平
        elif ratio < 0.67:
            return 1  # 垂直
        return 2  # 方形
    
    def _get_density(self, bbox: Tuple, img_shape: Tuple) -> float:
        """密度 (缺陷面积/ROI面积)"""
        x, y, w, h = bbox
        H, W = img_shape
        roi_area = (h * 2) * (w * 2)  # 扩展ROI
        defect_area = w * h
        return defect_area / roi_area if roi_area > 0 else 0


class SpatialRelationExtractor:
    """空间关系特征 (多缺陷时)"""
    
    def __init__(self):
        pass
    
    def extract(self, bboxes: list) -> np.ndarray:
        """提取多缺陷间关系特征
        
        Args:
            bboxes: [(x,y,w,h), ...]
        Returns:
            关系特征向量
        """
        n = len(bboxes)
        if n < 2:
            return np.zeros(10)
        
        # 计算中心点
        centers = np.array([(b[0]+b[2]/2, b[1]+b[3]/2) for b in bboxes])
        
        # 中心点方差 (分散程度)
        center_std = centers.std(axis=0).mean()
        
        # 最近邻距离
        nn_dist = self._nearest_neighbor_dist(centers)
        
        # 聚类程度
        cluster_score = 1 / (1 + center_std)
        
        # 方向分布
        dir_variance = self._direction_variance(centers)
        
        return np.array([
            center_std, nn_dist, cluster_score, dir_variance, n,
            centers[:, 0].mean(), centers[:, 1].mean(),
            centers[:, 0].std(), centers[:, 1].std(), n / max(centers.max(), 1)
        ])
    
    def _nearest_neighbor_dist(self, points: np.ndarray) -> float:
        """最近邻距离"""
        from scipy.spatial.distance import cdist
        if len(points) < 2:
            return 0
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        return dists.min(axis=1).mean()
    
    def _direction_variance(self, centers: np.ndarray) -> float:
        """方向方差"""
        if len(centers) < 2:
            return 0
        vectors = centers[1:] - centers[0]
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        return angles.std()
