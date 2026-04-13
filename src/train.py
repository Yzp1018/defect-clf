"""
训练脚本
"""
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from data.dataset import DefectDataset
from models.classifier import build_model


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 数据
        self.train_loader = self._build_loader('train')
        self.val_loader = self._build_loader('val')
        
        # 模型
        self.model = build_model(
            config.model,
            num_classes=config.num_classes,
            backbone=config.backbone,
        ).to(self.device)
        
        # 损失函数 (处理不平衡)
        self.criterion = self._build_criterion()
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        
        # 最佳模型
        self.best_acc = 0
    
    def _build_loader(self, split: str):
        dataset = DefectDataset(
            data_dir=self.config.data_dir,
            split=split,
            crop_size=self.config.crop_size,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(split == 'train'),
            num_workers=self.config.num_workers,
        )
    
    def _build_criterion(self):
        """带类别权重的交叉熵"""
        return nn.CrossEntropyLoss(weight=None)  # 可添加 weights
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        preds, targets = [], []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            img = batch['image'].to(self.device)
            pos = batch['position'].to(self.device)
            label = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(img, pos)
            loss = self.criterion(output, label)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            preds.extend(output.argmax(dim=1).cpu().numpy())
            targets.extend(label.cpu().numpy())
        
        acc = np.mean(np.array(preds) == np.array(targets))
        return total_loss / len(self.train_loader), acc
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        preds, targets = [], []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            img = batch['image'].to(self.device)
            pos = batch['position'].to(self.device)
            label = batch['label'].to(self.device)
            
            output = self.model(img, pos)
            loss = self.criterion(output, label)
            
            total_loss += loss.item()
            preds.extend(output.argmax(dim=1).cpu().numpy())
            targets.extend(label.cpu().numpy())
        
        acc = np.mean(np.array(preds) == np.array(targets))
        return total_loss / len(self.val_loader), acc, preds, targets
    
    def train(self):
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, preds, targets = self.validate()
            
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # 保存最佳
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint('best_model.pth')
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch+1}.pth')
        
        print(f"\nBest Val Acc: {self.best_acc:.4f}")
    
    def save_checkpoint(self, name: str):
        path = os.path.join(self.config.output_dir, name)
        os.makedirs(self.config.output_dir, exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'config': self.config.__dict__,
        }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--model', type=str, default='defect_classifier')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
