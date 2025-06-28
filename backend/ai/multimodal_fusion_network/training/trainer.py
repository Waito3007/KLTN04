"""
Module huấn luyện mô hình.
"""
import os
import json
import logging
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from models.multimodal_fusion_model import EnhancedMultimodalFusionModel
from dataloader_multimodal import MultimodalCommitDataset

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CommitModelTrainer:
    """Class huấn luyện mô hình commit."""
    
    def __init__(
        self,
        model: EnhancedMultimodalFusionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        task_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[str] = None
    ):
        """
        Khởi tạo trainer.
        
        Args:
            model: Mô hình cần huấn luyện
            train_loader: DataLoader cho tập huấn luyện
            val_loader: DataLoader cho tập validation
            task_weights: Dict trọng số cho các task (mặc định là bằng nhau)
            learning_rate: Learning rate
            weight_decay: Weight decay
            device: Thiết bị sử dụng ('cuda' hoặc 'cpu')
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Xác định thiết bị
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Chuyển mô hình tới thiết bị
        self.model.to(self.device)
        
        # Lấy danh sách các task
        self.task_names = list(self.model.task_heads.keys())
        
        # Thiết lập trọng số cho các task
        if task_weights is None:
            self.task_weights = {task: 1.0 for task in self.task_names}
        else:
            self.task_weights = task_weights
        
        # Thiết lập các loss function cho các task
        self.loss_functions = {}
        for task_name, task_config in model.config['task_heads'].items():
            if task_config.get('type', 'regression') == 'regression':
                self.loss_functions[task_name] = nn.MSELoss()
            else:
                # Nếu là multi-label (multi-hot vector), dùng BCEWithLogitsLoss
                # Giả sử mọi classification đều là multi-label nếu output là vector
                # Nếu task_config có 'num_classes' và >1, dùng BCEWithLogitsLoss
                if 'num_classes' in task_config and task_config['num_classes'] > 2:
                    self.loss_functions[task_name] = nn.BCEWithLogitsLoss()
                else:
                    self.loss_functions[task_name] = nn.CrossEntropyLoss()
        
        # Thiết lập optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Lưu trữ lịch sử training
        self.history = {
            'train_loss': [],
            'val_loss': {},
            'val_accuracy': {}
        }
        
        for task in self.task_names:
            self.history['val_loss'][task] = []
            self.history['val_accuracy'][task] = []
    
    def train_epoch(self) -> float:
        """
        Huấn luyện một epoch.
        
        Returns:
            Trung bình loss trên tập huấn luyện
        """
        self.model.train()
        epoch_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            # Chuyển dữ liệu tới thiết bị
            text = batch['text'].to(self.device)
            metadata = batch['metadata'].to(self.device)
            labels = {task: batch['labels'][task].to(self.device) for task in self.task_names}
            
            # Xóa gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(text, metadata)
            
            # Tính loss cho từng task và tổng hợp
            batch_loss = None
            for task_name in self.task_names:
                task_loss = self.loss_functions[task_name](outputs[task_name], labels[task_name])
                weighted_loss = self.task_weights.get(task_name, 1.0) * task_loss
                if batch_loss is None:
                    batch_loss = weighted_loss
                else:
                    batch_loss = batch_loss + weighted_loss
            if batch_loss is None:
                continue  # Không có task nào, bỏ qua batch này
            
            # Backward và optimize
            batch_loss.backward()
            self.optimizer.step()
            
            # Cập nhật loss
            epoch_loss += batch_loss.item()
            
            # Cập nhật progress bar
            progress_bar.set_postfix({'loss': batch_loss.item()})
        
        # Tính trung bình loss
        avg_epoch_loss = epoch_loss / num_batches
        
        return avg_epoch_loss
    
    def evaluate(self) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Đánh giá mô hình trên tập validation.
        
        Returns:
            Tuple (val_loss, task_losses, task_accuracies)
        """
        self.model.eval()
        val_loss = 0.0
        task_losses = {task: 0 for task in self.task_names}
        task_correct = {task: 0 for task in self.task_names}
        task_total = {task: 0 for task in self.task_names}
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                # Chuyển dữ liệu tới thiết bị
                text = batch['text'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                labels = {task: batch['labels'][task].to(self.device) for task in self.task_names}
                
                # Forward pass
                outputs = self.model(text, metadata)
                
                # Tính loss và accuracy cho từng task
                batch_loss = torch.tensor(0.0, device=self.device)
                for task_name in self.task_names:
                    task_config = self.model.config['task_heads'][task_name]
                    
                    # Tính loss
                    task_loss = self.loss_functions[task_name](outputs[task_name], labels[task_name])
                    task_losses[task_name] += task_loss.item()
                    
                    # Áp dụng trọng số và cộng vào tổng loss
                    batch_loss += self.task_weights.get(task_name, 1.0) * task_loss
                    
                    # Tính accuracy
                    if task_config.get('type', 'classification') == 'classification':
                        predictions = torch.argmax(outputs[task_name], dim=1)
                        task_correct[task_name] += (predictions == labels[task_name]).sum().item()
                        task_total[task_name] += labels[task_name].size(0)
                
                # Cập nhật val_loss
                val_loss += batch_loss.item()
        
        # Tính trung bình loss và accuracy
        avg_val_loss = val_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}
        
        task_accuracies = {}
        for task in self.task_names:
            task_config = self.model.config['task_heads'][task]
            if task_config.get('type', 'classification') == 'classification' and task_total[task] > 0:
                task_accuracies[task] = task_correct[task] / task_total[task]
            else:
                task_accuracies[task] = 0.0
        
        return avg_val_loss, avg_task_losses, task_accuracies
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 5,
        checkpoint_dir: Optional[str] = None,
        log_interval: int = 1
    ) -> Dict[str, Any]:
        """
        Huấn luyện mô hình.
        
        Args:
            num_epochs: Số epochs
            early_stopping_patience: Số epochs chờ trước khi dừng sớm
            checkpoint_dir: Thư mục lưu checkpoint (nếu None thì không lưu)
            log_interval: Số epochs giữa các lần log
            
        Returns:
            Dict lịch sử huấn luyện
        """
        logger.info(f"Bắt đầu huấn luyện trên {self.device}")
        logger.info(f"Số epochs: {num_epochs}")
        logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        logger.info(f"Các task: {self.task_names}")
        
        best_val_loss = float('inf')
        early_stopping_counter = 0
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # Huấn luyện một epoch
            train_loss = self.train_epoch()
            
            # Đánh giá trên tập validation
            val_loss, task_losses, task_accuracies = self.evaluate()
            
            # Cập nhật lịch sử
            self.history['train_loss'].append(train_loss)
            for task in self.task_names:
                self.history['val_loss'][task].append(task_losses[task])
                self.history['val_accuracy'][task].append(task_accuracies[task])
            
            # Log kết quả
            if epoch % log_interval == 0:
                epoch_time = time.time() - epoch_start_time
                logger.info(f"Epoch {epoch}/{num_epochs} - {epoch_time:.2f}s - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
                
                for task in self.task_names:
                    task_config = self.model.config['task_heads'][task]
                    if task_config.get('type', 'classification') == 'classification':
                        logger.info(f"  {task}: Loss = {task_losses[task]:.4f}, Accuracy = {task_accuracies[task]:.4f}")
                    else:
                        logger.info(f"  {task}: Loss = {task_losses[task]:.4f}")
            
            # Lưu checkpoint nếu cần
            if checkpoint_dir is not None:
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                    self._save_checkpoint(checkpoint_path, epoch, val_loss)
                    logger.info(f"Đã lưu checkpoint tốt nhất với val_loss = {val_loss:.4f}")
                
                # Lưu checkpoint mỗi 10 epochs
                if epoch % 10 == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt')
                    self._save_checkpoint(checkpoint_path, epoch, val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    logger.info(f"Early stopping sau {epoch} epochs")
                    break
        
        total_time = time.time() - start_time
        logger.info(f"Đã hoàn thành huấn luyện sau {total_time:.2f}s")
        
        return self.history
    
    def _save_checkpoint(self, path: str, epoch: int, val_loss: float) -> None:
        """
        Lưu checkpoint của mô hình.
        
        Args:
            path: Đường dẫn lưu checkpoint
            epoch: Epoch hiện tại
            val_loss: Validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'model_config': self.model.config
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> int:
        """
        Tải checkpoint của mô hình.
        
        Args:
            path: Đường dẫn đến checkpoint
            
        Returns:
            Epoch của checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        logger.info(f"Đã tải checkpoint từ epoch {checkpoint['epoch']} với val_loss = {checkpoint['val_loss']:.4f}")
        
        return checkpoint['epoch']
    
    def plot_loss_curves(self, save_path: Optional[str] = None) -> None:
        """
        Vẽ đồ thị loss theo epoch.
        
        Args:
            save_path: Đường dẫn lưu đồ thị (nếu None thì hiển thị)
        """
        plt.figure(figsize=(12, 6))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Vẽ train loss
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        
        # Vẽ val loss cho từng task
        colors = ['r', 'g', 'm', 'c', 'y', 'k']
        for i, task in enumerate(self.task_names):
            color = colors[i % len(colors)]
            plt.plot(epochs, self.history['val_loss'][task], f'{color}-', label=f'{task} Val Loss')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Đã lưu đồ thị loss vào {save_path}")
        else:
            plt.show()
    
    def plot_accuracy_curves(self, save_path: Optional[str] = None) -> None:
        """
        Vẽ đồ thị accuracy theo epoch.
        
        Args:
            save_path: Đường dẫn lưu đồ thị (nếu None thì hiển thị)
        """
        plt.figure(figsize=(12, 6))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Vẽ accuracy cho từng task
        colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
        for i, task in enumerate(self.task_names):
            task_config = self.model.config['task_heads'][task]
            if task_config.get('type', 'classification') == 'classification':
                color = colors[i % len(colors)]
                plt.plot(epochs, self.history['val_accuracy'][task], f'{color}-', label=f'{task} Accuracy')
        
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Đã lưu đồ thị accuracy vào {save_path}")
        else:
            plt.show()


def train_multimodal_fusion_model(
    model_config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    task_weights: Optional[Dict[str, float]] = None,
    checkpoint_dir: Optional[str] = None,
    device: Optional[str] = None,
    early_stopping_patience: int = 5,
    log_interval: int = 1
) -> Tuple[EnhancedMultimodalFusionModel, Dict[str, Any]]:
    """
    Huấn luyện mô hình fusion đa phương thức.
    
    Args:
        model_config: Cấu hình mô hình
        train_loader: DataLoader cho tập huấn luyện
        val_loader: DataLoader cho tập validation
        num_epochs: Số epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        task_weights: Dict trọng số cho các task
        checkpoint_dir: Thư mục lưu checkpoint
        device: Thiết bị sử dụng ('cuda' hoặc 'cpu')
        early_stopping_patience: Số epochs chờ trước khi dừng sớm
        log_interval: Số epochs giữa các lần log
        
    Returns:
        Tuple (model, history)
    """
    # Tạo model
    model = EnhancedMultimodalFusionModel(model_config)
    
    # Tạo trainer
    trainer = CommitModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_weights=task_weights,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device
    )
    
    # Huấn luyện model
    history = trainer.train(
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir=checkpoint_dir,
        log_interval=log_interval
    )
    
    # Vẽ đồ thị kết quả
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        trainer.plot_loss_curves(os.path.join(checkpoint_dir, 'loss_curves.png'))
        trainer.plot_accuracy_curves(os.path.join(checkpoint_dir, 'accuracy_curves.png'))
    
    return model, history


if __name__ == "__main__":
    ()
    # Ví dụ sử dụng
    # Giả sử đã có model_config, train_loader, val_loader
    
    # model_config = {...}
    # train_loader = ...
    # val_loader = ...
    
    # task_weights = {
    #     'risk_prediction': 1.0,
    #     'complexity_prediction': 1.0,
    #     'hotspot_prediction': 1.0,
    #     'urgency_prediction': 1.0,
    #     'completeness_prediction': 1.0,
    #     'estimated_effort': 0.5
    # }
    
    # model, history = train_multimodal_fusion_model(
    #     model_config=model_config,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=50,
    #     learning_rate=1e-3,
    #     weight_decay=1e-5,
    #     task_weights=task_weights,
    #     checkpoint_dir='checkpoints',
    #     device='cuda',
    #     early_stopping_patience=5,
    #     log_interval=1
    # )

    # Kiểm tra dữ liệu đầu vào
    train_ds = MultimodalCommitDataset('data/processed/train.json')
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    for batch in train_loader:
        print(batch['text'], batch['labels'], batch['meta'])
