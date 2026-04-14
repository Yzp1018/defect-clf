"""
高分辨率图像推理脚本 - 针对工业相机大图优化

使用方法:
    python inference_highres.py \
        --image_path /path/to/highres_image.jpg \
        --checkpoint outputs/best_model.pth \
        --window_size 512 \
        --stride 256 \
        --output_dir results/
"""
import os
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Tuple

from data.highres_loader import (
    HighResImageLoader,
    SlidingWindowDataset,
    ImagePyramid,
    create_highres_dataloader,
)
from models.classifier import build_model


class HighResInference:
    """高分辨率图像推理器"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        window_size: int = 512,
        stride: int = 256,
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            checkpoint_path: 模型检查点路径
            device: 计算设备
            window_size: 滑动窗口大小
            stride: 滑动步长
            confidence_threshold: 置信度阈值
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        
        # 加载模型
        self.model = self._load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 类别映射 (从检查点加载或默认)
        self.idx_to_class = getattr(self.model, 'idx_to_class', {
            0: 'scratch', 1: 'dent', 2: 'spot', 3: 'crack', 4: 'bubble'
        })
    
    def _load_checkpoint(self, checkpoint_path: str) -> torch.nn.Module:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        config = checkpoint.get('config', {})
        num_classes = config.get('num_classes', 5)
        backbone = config.get('backbone', 'efficientnet_b0')
        
        model = build_model(
            name=config.get('model', 'defect_classifier'),
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False,
        )
        
        # 处理可能的 DataParallel 包装
        state_dict = checkpoint['model']
        if any(k.startswith('module.') for k in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        return model
    
    @torch.no_grad()
    def predict_single_image(
        self,
        img_path: str,
        save_visualization: bool = True,
        output_dir: str = 'results',
    ) -> Dict:
        """
        预测单张高分辨率图像
        
        Args:
            img_path: 图像路径
            save_visualization: 是否保存可视化结果
            output_dir: 输出目录
        
        Returns:
            预测结果字典
        """
        # 创建滑动窗口数据集
        dataset = SlidingWindowDataset(
            img_path=img_path,
            window_size=self.window_size,
            stride=self.stride,
        )
        
        loader = create_highres_dataloader(
            img_paths=[img_path],
            window_size=self.window_size,
            stride=self.stride,
            batch_size=16,
            num_workers=4,
        )
        
        # 收集所有预测
        all_predictions = []
        all_scores = []
        all_positions = []
        
        for batch in tqdm(loader, desc="Inference"):
            img = batch['image'].to(self.device)
            
            # 需要位置特征 (这里使用中心位置的默认特征)
            # 实际应用中应该根据窗口位置计算真实的位置特征
            batch_size = img.shape[0]
            pos_feat = torch.zeros(batch_size, 11).to(self.device)
            
            # 前向传播
            output = self.model(img, pos_feat)
            prob = torch.softmax(output, dim=1)
            score, pred = torch.max(prob, dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_scores.extend(score.cpu().numpy())
            all_positions.append({
                'x': batch['window_x'].numpy(),
                'y': batch['window_y'].numpy(),
                'w': batch['window_w'].numpy(),
                'h': batch['window_h'].numpy(),
            })
        
        # 合并位置信息
        window_info = {}
        for pos in all_positions:
            for i in range(len(pos['x'])):
                idx = len(window_info)
                window_info[idx] = {
                    'x': int(pos['x'][i]),
                    'y': int(pos['y'][i]),
                    'w': int(pos['w'][i]),
                    'h': int(pos['h'][i]),
                }
        
        # 生成全图预测
        pred_map, score_map = dataset.merge_predictions(
            predictions=np.array(all_predictions),
            scores=np.array(all_scores),
            strategy='weighted_avg',
        )
        
        # 提取缺陷区域
        defects = self._extract_defects(
            pred_map=pred_map,
            score_map=score_map,
            window_info=window_info,
        )
        
        # 保存结果
        if save_visualization:
            self._save_results(
                img_path=img_path,
                pred_map=pred_map,
                score_map=score_map,
                defects=defects,
                output_dir=output_dir,
            )
        
        return {
            'image_path': img_path,
            'image_size': (dataset.loader.height, dataset.loader.width),
            'predictions': all_predictions,
            'scores': all_scores,
            'defects': defects,
            'summary': self._generate_summary(defects),
        }
    
    def _extract_defects(
        self,
        pred_map: np.ndarray,
        score_map: np.ndarray,
        window_info: Dict,
        min_area: int = 100,
    ) -> List[Dict]:
        """
        从预测图中提取缺陷区域
        
        Args:
            pred_map: 预测类别图
            score_map: 置信度图
            window_info: 窗口信息
            min_area: 最小缺陷面积
        
        Returns:
            缺陷列表
        """
        defects = []
        
        # 找到高置信度区域
        mask = score_map > self.confidence_threshold
        
        # 形态学操作连接相邻区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            mask_closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 计算区域内的主要类别和平均分数
            region_mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.drawContours(region_mask, [cnt], -1, 255, -1)
            region_score = score_map[region_mask > 0].mean()
            region_pred = pred_map[region_mask > 0]
            main_pred = np.bincount(region_pred).argmax()
            
            defects.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'area': int(area),
                'class': int(main_pred),
                'class_name': self.idx_to_class.get(main_pred, f'class_{main_pred}'),
                'confidence': float(region_score),
                'contour': cnt.tolist(),
            })
        
        # 按置信度排序
        defects.sort(key=lambda d: d['confidence'], reverse=True)
        
        return defects
    
    def _generate_summary(self, defects: List[Dict]) -> Dict:
        """生成缺陷统计摘要"""
        summary = {
            'total_defects': len(defects),
            'by_class': {},
            'high_confidence_count': 0,
        }
        
        for defect in defects:
            cls_name = defect['class_name']
            summary['by_class'][cls_name] = summary['by_class'].get(cls_name, 0) + 1
            
            if defect['confidence'] > 0.8:
                summary['high_confidence_count'] += 1
        
        return summary
    
    def _save_results(
        self,
        img_path: str,
        pred_map: np.ndarray,
        score_map: np.ndarray,
        defects: List[Dict],
        output_dir: str,
    ):
        """保存可视化结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载原图
        loader = HighResImageLoader(img_path)
        img = loader.get_full_image(max_dim=4096)
        
        # 调整预测图到原图大小
        h, w = img.shape[:2]
        pred_resized = cv2.resize(pred_map, (w, h), interpolation=cv2.INTER_NEAREST)
        score_resized = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 创建热力图
        heatmap = cv2.applyColorMap(
            (score_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # 叠加
        overlay = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
        
        # 绘制缺陷框
        for defect in defects[:20]:  # 最多显示 20 个
            x, y, w, h = defect['bbox']
            color = (0, 255, 0) if defect['confidence'] > 0.8 else (0, 128, 255)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            label = f"{defect['class_name']}: {defect['confidence']:.2f}"
            cv2.putText(
                overlay, label, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        # 保存
        base_name = Path(img_path).stem
        cv2.imwrite(str(output_path / f"{base_name}_result.jpg"), overlay)
        cv2.imwrite(str(output_path / f"{base_name}_heatmap.jpg"), heatmap)
        
        # 保存 JSON 结果
        result_json = {
            'image_path': img_path,
            'image_size': [img.shape[1], img.shape[0]],
            'defects': defects,
            'summary': self._generate_summary(defects),
        }
        
        with open(output_path / f"{base_name}_results.json", 'w') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='高分辨率图像缺陷检测推理')
    parser.add_argument('--image_path', type=str, required=True, help='输入图像路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--window_size', type=int, default=512, help='滑动窗口大小')
    parser.add_argument('--stride', type=int, default=256, help='滑动步长')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = HighResInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        window_size=args.window_size,
        stride=args.stride,
        confidence_threshold=args.confidence_threshold,
    )
    
    # 执行推理
    results = inferencer.predict_single_image(
        img_path=args.image_path,
        save_visualization=True,
        output_dir=args.output_dir,
    )
    
    # 打印摘要
    print("\n=== 检测结果摘要 ===")
    print(f"图像尺寸：{results['image_size']}")
    print(f"总缺陷数：{results['summary']['total_defects']}")
    print("各类缺陷数量:")
    for cls_name, count in results['summary']['by_class'].items():
        print(f"  {cls_name}: {count}")
    print(f"高置信度缺陷 (>0.8): {results['summary']['high_confidence_count']}")


if __name__ == '__main__':
    main()
