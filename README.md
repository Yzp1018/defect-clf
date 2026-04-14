# 工业缺陷分类系统 - 完善版

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**工业表面缺陷分类系统** - 专为高分辨率工业相机图像设计

## 🎯 核心特性

### 针对高分辨率图像的优化
- **滑动窗口推理**: 支持 4K-32K+ 超大图像，自动分割成可处理的小窗口
- **内存映射加载**: 懒加载机制，避免 OOM 错误
- **多尺度金字塔**: 检测不同大小的缺陷
- **高效批处理**: 并行预处理和 GPU 加速

### 先进的模型架构
- **双分支融合**: CNN 图像特征 + MLP 位置特征
- **多模型支持**: 标准版、多尺度版、轻量版
- **类别不平衡处理**: Focal Loss、类别权重自动计算
- **数据增强**: Mixup、CutMix、Cutout 等高级策略

### 完善的工具链
- **训练脚本**: 支持 YAML 配置、断点续训
- **推理脚本**: 高分辨率专用推理器，自动生成热力图
- **可视化工具**: Grad-CAM、t-SNE、特征分布分析
- **单元测试**: 覆盖核心模块

## 📦 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 推理 (已知缺陷位置)
```bash
python src/inference.py \
    --image_path data/test/sample.jpg \
    --checkpoint outputs/best_model.pth
```

### 推理 (高分辨率全图扫描)
```bash
python src/inference_highres.py \
    --image_path /path/to/8k_image.jpg \
    --checkpoint outputs/best_model.pth \
    --window_size 512 \
    --stride 256 \
    --output_dir results/
```

### 训练
```bash
python src/train.py \
    --config configs/config.yaml \
    --data_dir data \
    --output_dir outputs
```

## 📚 文档

- [高分辨率图像处理指南](HIGHRES_GUIDE.md) - 详细的使用说明和最佳实践
- [数据格式说明](DATA_FORMAT.md) - 标注文件格式
- [模型架构](MODEL_ARCHITECTURE.md) - 网络结构详解

## 🔧 使用场景

### 场景 1: 已知缺陷位置 (推荐)
如果有 bbox 标注，直接使用 `DefectDataset`:
```python
from src.data.dataset import DefectDataset

dataset = DefectDataset(
    data_dir='data',
    split='test',
    crop_size=256,
    use_context=0.2,
)
```

### 场景 2: 未知缺陷位置 (全图扫描)
对整张高分辨率图进行滑动窗口扫描:
```python
from src.data.highres_loader import SlidingWindowDataset

dataset = SlidingWindowDataset(
    img_path='large_image.jpg',
    window_size=512,
    stride=256,
)
```

### 场景 3: 多尺度检测
```python
from src.data.highres_loader import ImagePyramid

pyramid = ImagePyramid('image.jpg', scales=[0.5, 1.0, 2.0])
results = pyramid.predict_multi_scale(model, device='cuda')
```

## 📊 项目结构

```
workspace/
├── src/
│   ├── data/
│   │   ├── dataset.py          # 缺陷数据集 (已知位置)
│   │   ├── highres_loader.py   # 高分辨率加载器 (新)
│   │   └── augment.py          # 数据增强
│   ├── models/
│   │   └── classifier.py       # 分类模型
│   ├── utils/
│   │   ├── losses.py           # 损失函数 (Focal Loss 等)
│   │   └── visualization.py    # 可视化工具
│   ├── train.py                # 训练脚本
│   ├── inference.py            # 标准推理
│   └── inference_highres.py    # 高分辨率推理 (新)
├── tests/
│   ├── test_modules.py         # 核心模块测试
│   └── test_highres.py         # 高分辨率测试 (新)
├── configs/
│   └── config.yaml             # 配置文件
├── HIGHRES_GUIDE.md            # 高分辨率指南 (新)
└── README.md                   # 本文件
```

## 🎓 核心改进

相比基础版本，本项目完善了以下内容:

| 模块 | 改进内容 |
|------|---------|
| **损失函数** | ✅ Focal Loss、Label Smoothing、组合损失 |
| **数据增强** | ✅ Mixup、CutMix、Cutout、几何/颜色变换 |
| **高分辨率** | ✅ 滑动窗口、内存映射、多尺度金字塔 |
| **训练** | ✅ YAML 配置、类别权重自动计算 |
| **可视化** | ✅ Grad-CAM、t-SNE、预测案例 |
| **测试** | ✅ 21 个单元测试用例 |
| **文档** | ✅ 详细使用指南和 API 文档 |

## 📈 性能建议

| 图像尺寸 | 窗口大小 | 步长 | 批大小 | 显存需求 |
|---------|---------|------|--------|---------|
| 4K | 512 | 256 | 16 | 4GB |
| 8K | 512 | 256 | 16 | 4GB |
| 12K+ | 256 | 128 | 8 | 2GB |

## 🔍 输出示例

推理完成后生成:
```
results/
├── image_result.jpg      # 带标注的结果图
├── image_heatmap.jpg     # 缺陷热力图
└── image_results.json    # JSON 详细结果
```

JSON 格式:
```json
{
  "image_size": [7680, 4320],
  "defects": [
    {
      "bbox": [1234, 567, 128, 96],
      "class_name": "scratch",
      "confidence": 0.92,
      "area": 12288
    }
  ],
  "summary": {
    "total_defects": 5,
    "by_class": {"scratch": 3, "dent": 2}
  }
}
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## 📄 许可证

MIT License
