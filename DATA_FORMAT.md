# 数据格式说明

## 1. 目录结构

```
data/
├── images/           # 高分辨率原始图像
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── annotations/      # 标注文件
    ├── train.json
    ├── val.json
    └── test.json
```

---

## 2. 标注格式

### bbox 格式: [cx, cy, w, h] (中心点 + 宽高)

```json
{
  "annotations": [
    {"id": 1, "image_name": "img_001.jpg", "bbox": [cx, cy, width, height], "class": "scratch"}
  ],
  "classes": ["scratch", "dent", "spot", "crack", "bubble"]
}
```

### 坐标说明

```
图像坐标系 (左上角为原点, y向下为正):

      cx (中心点X)
      ←──────→
      ┌───────┐  ↑ 
      │       │  │ h (高)
      │   ●   │  ↓
      │       │  
      └───────┘
      ←─ w →

bbox = [cx, cy, w, h]
     = [中心X, 中心Y, 宽, 高]
     = [1000, 500, 80, 12]
```

---

## 3. 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| id | int | ✅ | 唯一标识 |
| image_name | string | ✅ | 图像文件名 |
| bbox | array[4] | ✅ | **[cx, cy, width, height]** |
| class | string | ✅ | 缺陷类别 |

---

## 4. 类别定义

| 类别 | 特征描述 |
|------|----------|
| scratch | 划痕 - 细长线条 |
| dent | 凹陷 - 不规则凹坑 |
| spot | 污点 - 斑点 |
| crack | 裂纹 - 树状/直线 |
| bubble | 气泡 - 圆形/椭圆 |

---

## 5. 数据集示例

### train.json

```json
{
  "annotations": [
    {"id": 1, "image_name": "img_001.jpg", "bbox": [2500, 1200, 80, 12], "class": "scratch"},
    {"id": 2, "image_name": "img_001.jpg", "bbox": [5120, 3210, 45, 45], "class": "dent"},
    {"id": 3, "image_name": "img_002.jpg", "bbox": [1810, 810, 25, 25], "class": "spot"},
    {"id": 4, "image_name": "img_002.jpg", "bbox": [6250, 1510, 150, 8], "class": "crack"},
    {"id": 5, "image_name": "img_003.jpg", "bbox": [3020, 2520, 35, 35], "class": "bubble"},
    {"id": 6, "image_name": "img_003.jpg", "bbox": [4520, 1810, 60, 8], "class": "scratch"},
    {"id": 7, "image_name": "img_004.jpg", "bbox": [1220, 3520, 40, 40], "class": "dent"},
    {"id": 8, "image_name": "img_005.jpg", "bbox": [7010, 920, 20, 20], "class": "spot"},
    {"id": 9, "image_name": "img_005.jpg", "bbox": [5520, 2810, 120, 6], "class": "crack"},
    {"id": 10, "image_name": "img_006.jpg", "bbox": [2020, 1520, 30, 30], "class": "bubble"}
  ],
  "classes": ["scratch", "dent", "spot", "crack", "bubble"]
}
```

### val.json

```json
{
  "annotations": [
    {"id": 101, "image_name": "img_v001.jpg", "bbox": [2820, 1410, 70, 10], "class": "scratch"},
    {"id": 102, "image_name": "img_v001.jpg", "bbox": [4820, 2920, 38, 38], "class": "dent"},
    {"id": 103, "image_name": "img_v002.jpg", "bbox": [1520, 620, 22, 22], "class": "spot"}
  ],
  "classes": ["scratch", "dent", "spot", "crack", "bubble"]
}
```

---

## 6. 训练命令

```bash
python src/train.py --data_dir data --epochs 50
```
