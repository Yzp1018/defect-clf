"""
特征可视化工具 - Grad-CAM, t-SNE, 特征分布分析
"""
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class GradCAM:
    """Grad-CAM 可视化 - 显示模型关注区域"""
    
    def __init__(self, model: torch.nn.Module, target_layer: str = 'cnn'):
        self.model = model
        self.target_layer_name = target_layer
        self.target_layer = None
        self.gradients = None
        self.activations = None
        
        # 注册 hook
        self._register_hook()
    
    def _register_hook(self):
        """注册前向和后向 hook"""
        # 找到目标层
        for name, module in self.model.named_modules():
            if name.endswith(self.target_layer_name):
                self.target_layer = module
                module.register_forward_hook(self._forward_hook)
                module.register_backward_hook(self._backward_hook)
                break
        
        if self.target_layer is None:
            raise ValueError(f"Target layer '{self.target_layer_name}' not found")
    
    def _forward_hook(self, module, input, output):
        self.activations = output
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate(
        self,
        img: torch.Tensor,
        pos: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """生成 Grad-CAM 热力图
        
        Args:
            img: [1, 3, H, W] 输入图像
            pos: [1, pos_feat_dim] 位置特征
            target_class: 目标类别，None 则使用预测类别
        Returns:
            heatmap: [H, W, 3] RGB 热力图
        """
        self.model.eval()
        
        # 前向传播
        output = self.model(img, pos)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        
        # 计算 Grad-CAM
        gradients = self.gradients.cpu().detach().numpy()  # [1, C, H', W']
        activations = self.activations.cpu().detach().numpy()
        
        # 全局平均池化梯度
        weights = gradients.mean(axis=(2, 3), keepdims=True)  # [1, C, 1, 1]
        
        # 加权激活
        cam = (weights * activations).sum(axis=1)  # [1, H', W']
        cam = np.maximum(cam, 0)  # ReLU
        
        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # 上采样到原图尺寸
        img_np = img[0].cpu().permute(1, 2, 0).numpy()
        h, w = img_np.shape[:2]
        cam_resized = cv2.resize(cam[0], (w, h))
        
        # 应用 colormap
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 叠加到原图
        overlay = heatmap * 0.5 + (img_np * 255).astype(np.uint8) * 0.5
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay, heatmap


class FeatureVisualizer:
    """特征可视化工具集"""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.grad_cam = GradCAM(model)
    
    @torch.no_grad()
    def visualize_grad_cam(
        self,
        img: torch.Tensor,
        pos: torch.Tensor,
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """生成并保存 Grad-CAM 可视化"""
        overlay, heatmap = self.grad_cam.generate(img, pos)
        
        if save_path:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(img[0].cpu().permute(1, 2, 0).numpy())
            plt.title('Input Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap)
            plt.title('Grad-CAM Heatmap')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
        
        return overlay
    
    @torch.no_grad()
    def extract_features(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_samples: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """提取融合特征用于 t-SNE 可视化
        
        Returns:
            features: [N, feat_dim]
            labels: [N]
            class_names: 类别名称列表
        """
        self.model.eval()
        all_features = []
        all_labels = []
        class_names = []
        
        count = 0
        for batch in dataloader:
            if count >= max_samples:
                break
            
            img = batch['image'].to(self.device)
            pos = batch['position'].to(self.device)
            label = batch['label'].cpu().numpy()
            class_name = batch.get('class_name', ['unknown'] * len(label))
            
            # 获取特征
            features = self.model.get_feature(img, pos)['fused']  # [B, feat_dim]
            
            all_features.append(features.cpu().numpy())
            all_labels.append(label)
            
            if not class_names:
                class_names = list(set(class_name))
            
            count += len(label)
        
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        
        return features, labels, class_names
    
    def plot_tsne(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None,
        perplexity: float = 30,
    ):
        """t-SNE 可视化特征分布"""
        # 降维到 2D
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        # 绘图
        plt.figure(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            color_name = class_names[i] if i < len(class_names) else f'Class {label}'
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[colors[i]],
                label=color_name,
                alpha=0.6,
                s=50,
            )
        
        plt.legend()
        plt.title('t-SNE Visualization of Features')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved t-SNE plot to {save_path}")
        
        plt.show()
    
    def analyze_feature_distribution(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """分析特征分布统计"""
        plt.figure(figsize=(15, 5))
        
        # 特征均值分布
        plt.subplot(1, 3, 1)
        feature_means = features.mean(axis=0)
        plt.hist(feature_means, bins=50, alpha=0.7)
        plt.title('Feature Mean Distribution')
        plt.xlabel('Mean Value')
        plt.ylabel('Frequency')
        
        # 特征标准差分布
        plt.subplot(1, 3, 2)
        feature_stds = features.std(axis=0)
        plt.hist(feature_stds, bins=50, alpha=0.7, color='orange')
        plt.title('Feature Std Distribution')
        plt.xlabel('Std Value')
        plt.ylabel('Frequency')
        
        # 类别特征中心距离
        plt.subplot(1, 3, 3)
        unique_labels = np.unique(labels)
        centers = []
        for label in unique_labels:
            mask = labels == label
            center = features[mask].mean(axis=0)
            centers.append(center)
        
        centers = np.array(centers)
        if len(centers) > 1:
            # 计算类间距离
            distances = []
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    distances.append(dist)
            plt.hist(distances, bins=30, alpha=0.7, color='green')
            plt.title('Inter-class Center Distances')
            plt.xlabel('Distance')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved feature distribution analysis to {save_path}")
        
        plt.show()


def visualize_model_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int = 10,
    save_dir: str = 'visualizations',
    device: str = 'cpu',
):
    """可视化模型预测结果（正确/错误案例）"""
    from PIL import Image
    
    model.eval()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    correct_count = 0
    error_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if correct_count >= num_samples and error_count >= num_samples:
                break
            
            img = batch['image'].to(device)
            pos = batch['position'].to(device)
            label = batch['label'].cpu().numpy()
            pred = model(img, pos).argmax(dim=1).cpu().numpy()
            
            for i in range(len(label)):
                is_correct = label[i] == pred[i]
                
                if is_correct and correct_count >= num_samples:
                    continue
                if not is_correct and error_count >= num_samples:
                    continue
                
                # 保存图像
                img_np = img[i].cpu().permute(1, 2, 0).numpy()
                status = 'correct' if is_correct else 'error'
                count = correct_count if is_correct else error_count
                
                filename = f"{status}_{count}_pred{pred[i]}_true{label[i]}.png"
                filepath = Path(save_dir) / filename
                
                plt.figure(figsize=(4, 4))
                plt.imshow(img_np)
                plt.title(f"Pred: {pred[i]}, True: {label[i]}")
                plt.axis('off')
                plt.savefig(filepath, dpi=100, bbox_inches='tight')
                plt.close()
                
                if is_correct:
                    correct_count += 1
                else:
                    error_count += 1
    
    print(f"Saved {correct_count} correct and {error_count} error cases to {save_dir}")
