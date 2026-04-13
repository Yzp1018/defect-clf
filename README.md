# 工业缺陷分类系统

## 问题描述

- **输入**: 工业相机拍摄的高分辨率图像 (4000-7000万像素)
- **特点**: 缺陷像素占比极低 (<1%)
- **已知**: 缺陷位置坐标 (x, y, width, height)
- **目标**: 基于位置和形状自动分类缺陷

---

## 方案设计

### 核心思路

```
高分辨率原图 → 坐标裁剪 → 缺陷ROI → 特征提取 → 分类器 → 类别
```

### 1. 数据预处理

#### 1.1 ROI 裁剪
- 根据已知坐标裁剪缺陷区域
- 使用多尺度裁剪: 1x, 1.5x, 2x 感受野
- 保留上下文信息

#### 1.2 图像增强
- 自适应直方图均衡化
- 去噪 (BM3D / Non-local means)
- 对比度增强

### 2. 特征工程

#### 2.1 位置特征
- 相对位置: (x/w, y/h) 归一化
- 区域分布: 边缘/中心/角落
- 密度分布: 缺陷到边缘的距离

#### 2.2 形状特征
- 几何: 面积、周长、圆形度、矩形度
- 轮廓: 傅里叶描述子、 Hu矩
- 纹理: LBP、GLCM、梯度方向

#### 2.3 深度特征
- 预训练 CNN 特征 (ResNet/EfficientNet)
- 注意力机制聚焦缺陷区域

### 3. 模型架构

```
┌─────────────────────────────────────┐
│         高分辨率图像 (4000×12000)    │
└──────────────┬──────────────────────┘
               │ 坐标裁剪
               ▼
┌─────────────────────────────────────┐
│      缺陷ROI (256×256 / 512×512)    │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌─────────────┐
│ 形状特征分支 │  │ 位置特征分支 │
│ (CNN/Transformer)│  │ (MLP)      │
└──────┬──────┘  └──────┬──────┘
       └───────┬───────┘
               ▼
┌─────────────────────────────────────┐
│        多模态融合 + 分类头           │
└──────────────┬──────────────────────┘
               ▼
         缺陷类别
```

### 4. 类别定义 (示例)

| 类别 | 特征描述 |
|------|----------|
| 划痕 | 细长线性，方向随机 |
| 凹陷 | 局部纹理异常，边缘模糊 |
| 污点 | 斑点状，边缘清晰 |
| 裂纹 | 树状或线性，方向固定 |
| 气泡 | 圆形或椭圆形，内部灰度低 |

---

## 实现代码

### 目录结构

```
defect-clf/
├── src/
│   ├── data/           # 数据加载
│   ├── features/       # 特征提取
│   ├── models/         # 模型定义
│   ├── train.py        # 训练脚本
│   └── inference.py    # 推理
├── configs/            # 配置文件
├── data/               # 数据目录
└── notebooks/          # 分析notebook
```

### 核心代码示例

```python
# 1. 缺陷ROI裁剪
class DefectCropper:
    def __init__(self, scales=[1.0, 1.5, 2.0]):
        self.scales = scales
    
    def crop(self, image, bbox, context=0.1):
        x, y, w, h = bbox
        # 添加上下文
        cx, cy = x + w/2, y + h/2
        pw, ph = w * (1+context), h * (1+context)
        
        crops = []
        for scale in self.scales:
            sw, sh = pw * scale, ph * scale
            crop = image[cy-sh//2:cy+sh//2, cx-sw//2:cx+sw//2]
            crops.append(crop)
        return crops

# 2. 形状特征提取
class ShapeFeatureExtractor:
    def extract(self, mask):
        # 几何特征
        area = np.sum(mask)
        perimeter = self.get_perimeter(mask)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Hu矩
        moments = cv2.moments(mask)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'hu_moments': hu_moments,
        }

# 3. 位置特征
class PositionFeatureExtractor:
    def extract(self, bbox, img_shape):
        h, w = img_shape
        x, y, bw, bh = bbox
        
        # 归一化中心
        cx, cy = (x + bw/2) / w, (y + bh/2) / h
        
        # 边缘距离
        dist_left = x / w
        dist_right = (w - x - bw) / w
        dist_top = y / h
        dist_bottom = (h - y - bh) / h
        min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)
        
        return {
            'cx_norm': cx,
            'cy_norm': cy,
            'min_edge_dist': min_edge_dist,
            'region': self.get_region(cx, cy),
        }

# 4. 模型定义
class DefectClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        # 图像分支
        self.cnn = timm.create_model('efficientnet_b0', pretrained=True, num_classes=128)
        
        # 位置MLP
        self.pos_mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, img, pos_feat):
        img_feat = self.cnn(img)
        pos_feat = self.pos_mlp(pos_feat)
        fused = torch.cat([img_feat, pos_feat], dim=-1)
        return self.fusion(fused)
```

---

## 训练策略

### 1. 数据不平衡
- 使用 focal loss
- SMOTE 过采样
- 类别权重

### 2. 弱监督
- 多实例学习 (MIL)
- 标注噪声容忍

### 3. 迁移学习
- ImageNet 预训练
- 工业缺陷数据微调

---

## 部署优化

### 1. 高分辨率处理
- 分块推理 (Tiling)
- 滑窗检测
- 坐标引导裁剪

### 2. 推理加速
- TensorRT 优化
- INT8 量化
- 批处理

---

## 评估指标

- Precision / Recall / F1
- 混淆矩阵
- 每类指标
- 推理速度 (FPS)
