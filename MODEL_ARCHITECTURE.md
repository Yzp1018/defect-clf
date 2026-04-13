# 模型架构详解

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        输入层                                        │
├─────────────────────────────────────────────────────────────────────┤
│  高分辨率图像 (4000×12000)                                          │
│         ↓                                                           │
│  根据bbox坐标裁剪缺陷ROI (+20%上下文)                               │
│         ↓                                                           │
│  Resize → 256×256×3                                                │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     双分支特征提取                                   │
├──────────────────────────────┬──────────────────────────────────────┤
│      图像分支 (CNN)          │         位置分支 (MLP)               │
│  ┌─────────────────────┐    │    ┌─────────────────────────┐        │
│  │  EfficientNet-B0    │    │    │  输入: 11维特征         │        │
│  │  (预训练ImageNet)   │    │    │  - cx_norm, cy_norm     │        │
│  │                     │    │    │  - rel_w, rel_h         │        │
│  │  Input: 256×256×3   │    │    │  - rel_area             │        │
│  │  Output: 1280维     │    │    │  - min_edge_dist        │        │
│  └─────────────────────┘    │    │  - min_corner_dist      │        │
│         ↓                   │    │  - is_edge, is_corner   │        │
│  GlobalAvgPool              │    │  - orientation          │        │
│  1280 维向量                │    │  - density              │        │
│         ↓                   │    └─────────────────────────┘        │
│  1280 维特征                │              ↓                        │
│                             │    ┌─────────────────────────┐        │
│                             │    │  Linear(11→64)          │        │
│                             │    │  ReLU                   │        │
│                             │    │  Linear(64→64)          │        │
│                             │    │  ReLU                   │        │
│                             │    └─────────────────────────┘        │
│                             │              ↓                        │
│                             │    64 维位置特征                      │
└─────────────────────────────┴──────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        特征融合                                      │
├─────────────────────────────────────────────────────────────────────┤
│  Concat([1280, 64]) → 1344 维                                      │
│         ↓                                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Linear(1344 → 256)                                         │   │
│  │  ReLU                                                      │   │
│  │  Dropout(0.3)                                              │   │
│  │                                                             │   │
│  │  Linear(256 → 128)                                         │   │
│  │  ReLU                                                      │   │
│  │  Dropout(0.15)                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         ↓                                                            │
│  128 维融合特征                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        分类头                                        │
├─────────────────────────────────────────────────────────────────────┤
│  Linear(128 → num_classes) → logits                                │
│         ↓                                                            │
│  Softmax → 概率分布                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 图像分支 (CNN Backbone)

### 2.1 骨干网络选择

| 模型 | 参数量 | Top-1 | 推理速度 | 推荐场景 |
|------|--------|-------|----------|----------|
| EfficientNet-B0 | 5.3M | 77.1% | 快 | 默认/生产 |
| EfficientNet-B1 | 7.8M | 79.2% | 中 | 精度优先 |
| EfficientNet-B2 | 9.2M | 80.5% | 中 | 精度优先 |
| ResNet-50 | 25.6M | 76.2% | 中 | 通用 |
| MobileNetV3 | 5.4M | 75.2% | 最快 | 边缘设备 |

### 2.2 EfficientNet 结构

```
EfficientNet-B0:
├── Stage 1: Conv3x3, 32 → 16
├── Stage 2-9: MBConv blocks (swish + se)
│   ├── Block1: 16→16, stride=1, expand=1
│   ├── Block2: 16→24, stride=2, expand=6
│   ├── Block3: 24→40, stride=2, expand=6
│   ├── Block4: 40→80, stride=2, expand=6
│   ├── Block5: 80→112, stride=1, expand=6
│   ├── Block6: 112→192, stride=2, expand=6
│   ├── Block7: 192→320, stride=1, expand=6
├── Stage 10: Conv1x1, 320→1280
└── GlobalAvgPool → 1280
```

### 2.3 为什么用 EfficientNet?

1. **参数效率高**: 5.3M 参数达到 77% ImageNet 准确率
2. **多尺度特征**: 复合缩放同时考虑深度/宽度/分辨率
3. **迁移学习**: ImageNet 预训练对工业缺陷有帮助
4. **可扩展**: B0→B7 系列满足不同精度需求

---

## 3. 位置特征分支 (MLP)

### 3.1 输入特征 (11维)

| 特征名 | 维度 | 说明 | 示例值 |
|--------|------|------|--------|
| cx_norm | 1 | 归一化中心X (0-1) | 0.35 |
| cy_norm | 1 | 归一化中心Y (0-1) | 0.42 |
| rel_width | 1 | 相对宽度 (bbox_w / img_w) | 0.01 |
| rel_height | 1 | 相对高度 (bbox_h / img_h) | 0.005 |
| rel_area | 1 | 相对面积 | 0.00005 |
| min_edge_dist | 1 | 到最近边缘的距离 | 0.15 |
| min_corner_dist | 1 | 到最近角落的距离 | 0.2 |
| is_edge | 1 | 是否靠近边缘 (0/1) | 0 |
| is_corner | 1 | 是否靠近角落 (0/1) | 1 |
| orientation | 1 | 方向: 0=水平, 1=垂直, 2=方形 | 0 |
| density | 1 | 缺陷密度 | 0.3 |

### 3.2 MLP 结构

```
位置特征 (11维)
       ↓
┌──────────────────┐
│ Linear(11 → 64) │  # 扩展维度
│ ReLU            │
│ Dropout(0.2)    │
└──────────────────┘
       ↓
┌──────────────────┐
│ Linear(64 → 64) │  # 进一步抽象
│ ReLU            │
└──────────────────┘
       ↓
   64维向量
```

---

## 4. 特征融合策略

### 4.1 简单 concat + MLP

```python
# 代码实现
img_feat = self.cnn(img)        # [B, 1280]
pos_feat = self.pos_mlp(pos)    # [B, 64]

fused = torch.cat([img_feat, pos_feat], dim=-1)  # [B, 1344]

fused = self.fusion(fused)  # [B, 128]
logits = self.classifier(fused)  # [B, num_classes]
```

### 4.2 融合层详解

```python
self.fusion = nn.Sequential(
    # 第一层: 1344 → 256
    nn.Linear(1344, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    # 第二层: 256 → 128
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.15),
)
```

---

## 5. 损失函数

### 5.1 交叉熵损失

```python
criterion = nn.CrossEntropyLoss()

# 如果类别不平衡，添加权重
class_weights = torch.tensor([1.0, 2.0, 1.5, 1.0, 1.0])  # 根据数据分布调整
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### 5.2 Focal Loss (推荐用于不平衡数据)

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

---

## 6. 优化器与学习率

### 6.1 AdamW

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,           # 初始学习率
    weight_decay=1e-4, # L2正则化
    betas=(0.9, 0.999)
)
```

### 6.2 学习率调度

```python
# Cosine Annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=50,       # 周期
    eta_min=1e-6    # 最小学习率
)
```

### 6.3 学习率曲线

```
lr
 1e-3 ┤                                            ╭───
 1e-4 ┤                         ╭─────────────     ╱
 1e-5 ┤        ╭───────────────╯               ╰──
 1e-6 ┤───────╯
      └──────────────────────────────────────────────→ epoch
         0      10     20     30     40     50
```

---

## 7. 数据增强

### 7.1 训练时增强

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### 7.2 验证时增强

```python
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

---

## 8. 训练流程

```python
for epoch in range(epochs):
    # Train
    model.train()
    for batch in train_loader:
        img, pos, label = batch['image'], batch['position'], batch['label']
        
        optimizer.zero_grad()
        output = model(img, pos)
        loss = criterion(output, label)
        
        loss.backward()
        optimizer.step()
    
    # Validate
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in val_loader:
            img, pos, label = batch['image'], batch['position'], batch['label']
            output = model(img, pos)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        
        acc = 100 * correct / total
        print(f'Epoch {epoch}: Val Acc = {acc:.2f}%')
    
    scheduler.step()
```

---

## 9. 模型变体

### 9.1 轻量级模型 (边缘设备)

```python
class LightweightClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        # 轻量CNN
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 128x128
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
        )  # 输出 64 维
        
        # 融合分类
        self.classifier = nn.Sequential(
            nn.Linear(64 + 11, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
```

### 9.2 多尺度模型

```python
# 输入多个尺度的图像 [1x, 1.5x, 2x]
# 提取特征后注意力加权融合
# 适合不同大小的缺陷
```

---

## 10. 推理部署

### 10.1 单张推理

```python
# 1. 加载模型
model = build_model('defect_classifier', num_classes=5)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model'])
model.eval()

# 2. 准备输入
image = load_and_preprocess('defect.jpg', bbox=[100, 200, 50, 10])
position = extract_position_features(bbox, img_shape)

# 3. 推理
with torch.no_grad():
    output = model(image, position)
    probs = torch.softmax(output, dim=1)
    pred_class = classes[output.argmax(1)]
    confidence = probs[0, pred_idx]

print(f"预测: {pred_class}, 置信度: {confidence:.2%}")
```

### 10.2 TensorRT 优化 (可选)

```python
# 转换到 TensorRT 以加速推理
import torch2trt

model_trt = torch2trt(model, [torch.ones(1, 3, 256, 256).cuda()])
torch.save(model_trt.state_dict(), 'model_trt.pth')
```

---

## 11. 完整训练命令

```bash
python src/train.py \
    --data_dir data \
    --output_dir outputs \
    --model defect_classifier \
    --backbone efficientnet_b0 \
    --num_classes 5 \
    --crop_size 256 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.0001 \
    --weight_decay 0.0001 \
    --num_workers 4
```

---

## 12. 预期效果

| 指标 | 数值 |
|------|------|
| 参数量 | ~6M |
| 训练时间 (RTX 3080) | ~10 min |
| 推理速度 | ~100 FPS |
| 预期准确率 (5类) | 85-95% |
