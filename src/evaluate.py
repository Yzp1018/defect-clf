"""
评估脚本
"""
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from data.dataset import DefectDataset
from torch.utils.data import DataLoader
from models.classifier import build_model


def evaluate(model_path: str, data_dir: str, config: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据
    test_ds = DefectDataset(data_dir, split='test')
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 模型
    model = build_model(
        config['model_name'],
        num_classes=config['num_classes'],
        backbone=config['backbone'],
    ).to(device)
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    
    # 预测
    preds, targets = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            img = batch['image'].to(device)
            pos = batch['position'].to(device)
            
            output = model(img, pos)
            preds.extend(output.argmax(dim=1).cpu().numpy())
            targets.extend(batch['label'].numpy())
    
    # 报告
    print(classification_report(targets, preds, target_names=test_ds.classes))
    
    # 混淆矩阵
    cm = confusion_matrix(targets, preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_ds.classes, yticklabels=test_ds.classes)
    plt.xlabel('Pred')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data')
    args = parser.parse_args()
    
    config = {
        'model_name': 'defect_classifier',
        'num_classes': 5,
        'backbone': 'efficientnet_b0',
    }
    evaluate(args.model_path, args.data_dir, config)
