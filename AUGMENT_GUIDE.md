# 工业缺陷分类数据增强指南

## 核心原则

**您的场景特点：**
- 所有样本都是缺陷产品（无正常样本）
- 已知缺陷位置坐标（中心点 + 宽高）
- 目标：缺陷类型分类（如划痕、凹陷、污点等）

### ⚠️ 关键约束

1. **保持缺陷几何真实性** - 不改变缺陷形态的物理逻辑
2. **保持缺陷尺度比例** - 不改变缺陷相对大小
3. **禁止创造虚假样本** - Mixup/CutMix 会混合两个缺陷，产生不真实的缺陷形态

---

## 推荐的增强策略

### ✅ 强烈推荐（已实现）

| 增强类型 | 概率 | 作用 | 物理意义 |
|---------|------|------|---------|
| **噪声** | 40% | 高斯噪声 (σ=15) | 模拟相机传感器噪声 |
| **模糊** | 30% | 高斯模糊 (k=3/5/7) | 模拟相机抖动/对焦不准 |
| **光照变化** | 50% | 亮度/对比度调整 | 模拟产线光照波动 |
| **Cutout** | 40% | 2 个 15% 遮挡块 | 强制模型关注局部特征 |

### ⚠️ 谨慎使用

| 增强类型 | 默认 | 使用条件 |
|---------|------|---------|
| **水平翻转** | 禁用 | 仅当产品左右对称时启用 |
| **垂直翻转** | 禁用 | 仅当产品上下对称时启用 |
| **旋转** | 禁用 | 仅当产品旋转对称或缺陷方向无关时启用 |

### ❌ 禁止使用

| 增强类型 | 原因 |
|---------|------|
| **Mixup** | 混合两个缺陷样本会创造虚假的缺陷形态 |
| **CutMix** | 从一个缺陷裁剪到另一个会产生不真实的位置关系 |

---

## 使用方法

### 方式 1：使用推荐配置（默认）

```python
from src.data.augment import DataAugmentor, INDUSTRIAL_DEFAULT_CONFIG

# 直接使用工业场景推荐配置
augmenter = DataAugmentor(**INDUSTRIAL_DEFAULT_CONFIG)
augmented_img = augmenter(original_img)
```

### 方式 2：自定义配置

```python
from src.data.augment import DataAugmentor

# 非对称产品（大多数工业场景）
augmenter = DataAugmentor(
    product_symmetric=False,  # ⚠️ 重要：默认禁用几何变换
    hflip_prob=0.0,
    vflip_prob=0.0,
    rotate_prob=0.0,
    
    # 光度变换
    color_jitter_prob=0.5,
    brightness_range=(0.8, 1.2),
    contrast_range=(0.8, 1.2),
    
    # 噪声和模糊
    noise_prob=0.4,
    noise_std=15,
    blur_prob=0.3,
    
    # Cutout
    cutout_prob=0.4,
    cutout_ratio=0.15,
    cutout_num=2,
)
```

### 方式 3：对称产品（特殊情况）

```python
# 仅当产品物理对称时启用几何变换
augmenter = DataAugmentor(
    product_symmetric=True,   # ✅ 启用几何变换
    hflip_prob=0.5,           # 50% 概率水平翻转
    vflip_prob=0.3,           # 30% 概率垂直翻转
    rotate_prob=0.2,          # 20% 概率小角度旋转
    rotate_range=(-5, 5),     # ±5 度旋转
    
    # 其他增强保持不变
    noise_prob=0.4,
    blur_prob=0.3,
    cutout_prob=0.4,
)
```

---

## YAML 配置

在 `configs/config.yaml` 中已包含优化后的增强配置：

```yaml
augmentation:
  product_symmetric: false  # ⚠️ 根据产品特性调整
  hflip_prob: 0.0
  vflip_prob: 0.0
  rotate_prob: 0.0
  
  color_jitter_prob: 0.5
  brightness_range: [0.8, 1.2]
  contrast_range: [0.8, 1.2]
  
  noise_prob: 0.4
  noise_std: 15
  
  blur_prob: 0.3
  
  cutout_prob: 0.4
  cutout_ratio: 0.15
  cutout_num: 2
```

训练时自动加载：
```bash
python src/train.py --config configs/config.yaml
```

---

## 为什么这样设计？

### 1. 噪声和模糊是核心增强
- **工业相机特性**：传感器噪声、镜头抖动、对焦误差是常见变异
- **提升鲁棒性**：模型学会忽略噪声，关注缺陷本质特征
- **物理真实**：完全符合产线实际情况

### 2. Cutout 强制局部关注
- **缺陷通常很小**：高分辨率图像中缺陷可能只占<1%像素
- **防止背景依赖**：避免模型通过背景纹理"作弊"分类
- **提升泛化**：类似 Dropout，防止过拟合

### 3. 几何变换需谨慎
- **缺陷有方向性**：划痕的方向、裂纹的走向可能包含分类信息
- **产品不对称**：大多数工业零件有明确的上下/左右方向
- **例外情况**：圆形对称零件（如轴承、晶圆）可启用翻转/旋转

### 4. 禁用 Mixup/CutMix
- **物理不真实**：两个缺陷混合后产生的形态在现实中不存在
- **标签模糊**：混合样本的类别归属不明确
- **实验验证**：在缺陷分类任务中通常降低精度

---

## 效果验证

运行测试：
```bash
python tests/test_modules.py
```

预期输出：
```
✓ 增强器创建成功
✓ 几何变换已禁用（非对称模式）
✓ 重点增强已启用：噪声、模糊、Cutout
✓ 所有测试通过
```

---

## 调优建议

### 如果训练集准确率>>验证集准确率（过拟合）
- ✅ 增加 `noise_prob` 到 0.5-0.6
- ✅ 增加 `cutout_prob` 到 0.5
- ✅ 增加 `cutout_num` 到 3-4
- ✅ 增加 `noise_std` 到 20-25

### 如果训练集准确率≈验证集准确率但都低（欠拟合）
- ✅ 减少 `noise_prob` 到 0.2-0.3
- ✅ 减少 `blur_prob` 到 0.1-0.2
- ✅ 检查模型容量是否足够

### 如果产品确实对称
- ✅ 设置 `product_symmetric=true`
- ✅ 启用 `hflip_prob=0.5`
- ✅ 可选启用 `rotate_prob=0.2-0.3`

---

## 常见问题

**Q: 我的产品是圆形对称的，可以启用翻转吗？**  
A: 是的，设置 `product_symmetric=true` 并调整翻转概率。

**Q: 缺陷非常小（<50 像素），如何增强？**  
A: 增加 `cutout_num` 到 3-4，减小 `cutout_ratio` 到 0.05-0.1。

**Q: 产线光照变化很大，如何调整？**  
A: 扩大 `brightness_range` 到 (0.6, 1.4)，增加 `color_jitter_prob` 到 0.7。

**Q: 可以使用预训练的 ImageNet 权重吗？**  
A: 可以，但建议使用 Focal Loss 处理类别不平衡，并加强噪声/模糊增强。
