# 训练数据集

包含训练用的标注数据。

## 文件格式

```json
{
  "annotations": [
    {"id": 1, "image_name": "img_001.jpg", "bbox": [cx, cy, w, h], "class": "scratch"}
  ],
  "classes": ["scratch", "dent", "spot", "crack", "bubble"]
}
```

## 标注格式

- **bbox**: `[cx, cy, w, h]` - 中心点X, 中心点Y, 宽度, 高度 (像素)
- **class**: 缺陷类别
- **image_name**: 对应 images/ 目录下的图像文件名
