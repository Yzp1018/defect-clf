# 高分辨率图像处理指南

本指南介绍如何使用本项目处理工业相机的高分辨率照片（4K-32K+）。

## 核心特性

### 1. 滑动窗口推理
将超大图像分割成重叠的小窗口，分别进行预测后合并结果。

```bash
python src/inference_highres.py \
    --image_path /path/to/8k_image.jpg \
    --checkpoint outputs/best_model.pth \
    --window_size 512 \
    --stride 256 \
    --output_dir results/
```

### 2. 内存优化加载
支持懒加载和内存映射，避免一次性加载整张大图到内存。

```python
from data.highres_loader import HighResImageLoader

loader = HighResImageLoader('large_image.jpg', use_memory_map=True)
print(f"Image size: {loader.width} x {loader.height}")

# 只加载感兴趣区域
region = loader.get_region(x=1000, y=2000, w=512, h=512)
```

### 3. 多尺度金字塔
构建图像金字塔处理不同大小的缺陷。

```python
from data.highres_loader import ImagePyramid

pyramid = ImagePyramid('large_image.jpg', scales=[0.5, 1.0, 2.0])

# 获取特定尺度
img_2x = pyramid.get_scale(2.0)
```

## 使用场景

### 场景 1: 已知缺陷位置 (推荐)
如果已有缺陷的 bbox 标注，直接使用原有数据集：

```python
from data.dataset import DefectDataset

dataset = DefectDataset(
    data_dir='data',
    split='test',
    crop_size=256,
    use_context=0.2,  # 添加 20% 上下文
)
```

**优点**: 
- 精确裁剪缺陷区域
- 训练和推理一致性好
- 可以提取位置特征

### 场景 2: 未知缺陷位置 (全图扫描)
对整张图进行滑动窗口扫描：

```python
from data.highres_loader import SlidingWindowDataset, create_highres_dataloader

dataset = SlidingWindowDataset(
    img_path='large_image.jpg',
    window_size=512,
    stride=256,  # 50% 重叠
)

loader = create_highres_dataloader(
    img_paths=['large_image.jpg'],
    batch_size=16,
    num_workers=4,
)
```

**优点**:
- 无需预知缺陷位置
- 可以发现新缺陷
- 生成全图热力图

### 场景 3: 多尺度检测
结合图像金字塔和滑动窗口：

```python
from data.highres_loader import ImagePyramid

pyramid = ImagePyramid(
    'large_image.jpg',
    scales=[0.5, 0.75, 1.0, 1.5, 2.0],
    max_dim=8192,
)

results = pyramid.predict_multi_scale(model, device='cuda')
```

## 性能优化建议

### 1. 窗口大小选择
| 图像尺寸 | 推荐窗口 | 步长 | 显存需求 |
|---------|---------|------|---------|
| 4K (3840×2160) | 512 | 256 | 4GB |
| 8K (7680×4320) | 512 | 256 | 4GB |
| 12K+ | 256 | 128 | 2GB |

### 2. 批处理大小
根据显存调整：
- 4GB: batch_size=8
- 8GB: batch_size=16
- 16GB+: batch_size=32

### 3. 工作线程数
```python
# CPU 核心数较少时
num_workers = 2

# CPU 核心数充足时
num_workers = 8
```

## 输出结果说明

推理完成后生成以下文件：

```
results/
├── image_result.jpg      # 带标注的结果图
├── image_heatmap.jpg     # 缺陷热力图
└── image_results.json    # JSON 格式详细结果
```

JSON 结果格式：
```json
{
  "image_path": "large_image.jpg",
  "image_size": [7680, 4320],
  "defects": [
    {
      "bbox": [1234, 567, 128, 96],
      "area": 12288,
      "class": 0,
      "class_name": "scratch",
      "confidence": 0.92,
      "contour": [...]
    }
  ],
  "summary": {
    "total_defects": 5,
    "by_class": {
      "scratch": 3,
      "dent": 2
    },
    "high_confidence_count": 4
  }
}
```

## 训练适配

如果使用滑动窗口推理，建议训练时也采用类似策略：

### 方法 1: 从大图裁剪训练样本
```python
# 在数据集中随机裁剪多个窗口
crop_size = 512
use_context = 0.2
```

### 方法 2: 多尺度训练
```yaml
# configs/config.yaml
training:
  crop_sizes: [256, 384, 512]  # 随机选择
```

## 常见问题

### Q: 显存不足怎么办？
A: 减小 `window_size` 或 `batch_size`，或使用更轻量的模型：
```bash
python inference_highres.py \
    --checkpoint outputs/lightweight_model.pth \
    --window_size 256 \
    --batch_size 8
```

### Q: 推理速度太慢？
A: 
1. 增大 `stride` (减少窗口数量)
2. 使用 GPU 加速
3. 增加 `num_workers`
4. 先用低分辨率筛查，再对可疑区域精细分析

### Q: 小缺陷检测效果差？
A:
1. 使用多尺度金字塔
2. 减小窗口大小
3. 训练时使用 Focal Loss
4. 增加正样本权重

## 最佳实践

1. **优先使用已知位置**: 如果有 bbox 标注，直接用 `DefectDataset`
2. **合理设置阈值**: 根据业务需求调整 `confidence_threshold`
3. **后处理优化**: 使用形态学操作连接相邻缺陷
4. **结果验证**: 人工抽检高置信度和低置信度样本
5. **持续迭代**: 将误检/漏检样本加入训练集

## 扩展功能

### ONNX 导出 (加速推理)
```python
import torch

model.eval()
dummy_img = torch.randn(1, 3, 512, 512)
dummy_pos = torch.randn(1, 11)

torch.onnx.export(
    model,
    (dummy_img, dummy_pos),
    "model.onnx",
    input_names=['image', 'position'],
    output_names=['output'],
    dynamic_axes={'image': {0: 'batch'}, 'position': {0: 'batch'}}
)
```

### TensorRT 部署
```bash
# 转换 ONNX 到 TensorRT
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```
