"""
Multi-Task Trainer for Multi-Modal Fusion Network
Triển khai Joint Multi-Task Learning với Dynamic Loss Weighting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
from pathlib import Path
import json
import logging
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class MultiModalDataset(Dataset):
    """
    Dataset cho Multi-Modal Fusion Network
    """
    
    def __init__(self, samples: List[Dict], text_processor, metadata_processor, 
                 label_encoders: Dict[str, Any]):
        """
        Args:
            samples: List of samples with text, metadata, and labels
            text_processor: TextProcessor instance
            metadata_processor: MetadataProcessor instance
            label_encoders: Dict mapping task names to label encoders
        """
        self.samples = samples
        self.text_processor = text_processor
        self.metadata_processor = metadata_processor
        self.label_encoders = label_encoders
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Process text
        text = sample.get('text', '')
        if self.text_processor.method == "lstm":
            text_input = self.text_processor.encode_text_lstm(text)
            attention_mask = None
        else:
            text_encoding = self.text_processor.encode_text_transformer(text)
            text_input = text_encoding['input_ids']
            attention_mask = text_encoding['attention_mask']
        
        # Process metadata
        metadata_input = self.metadata_processor.process_sample(sample)
        
        # Process labels
        labels = {}
        for task_name, label_value in sample.get('labels', {}).items():
            if task_name in self.label_encoders:
                try:
                    encoded_label = self.label_encoders[task_name].transform([label_value])[0]
                    labels[task_name] = torch.tensor(encoded_label, dtype=torch.long)
                except ValueError:
                    # Handle unknown labels
                    labels[task_name] = torch.tensor(0, dtype=torch.long)
        
        result = {
            'text_input': text_input,
            'metadata_input': metadata_input,
            'labels': labels
        }
        
        if attention_mask is not None:
            result['attention_mask'] = attention_mask
        
        return result

def collate_fn(batch):
    """
    Custom collate function for DataLoader
    """
    # Stack text inputs
    text_inputs = torch.stack([item['text_input'] for item in batch])
    
    # Stack attention masks if present
    attention_masks = None
    if 'attention_mask' in batch[0]:
        attention_masks = torch.stack([item['attention_mask'] for item in batch])
    
    # Stack metadata inputs
    metadata_keys = batch[0]['metadata_input'].keys()
    metadata_inputs = {}
    for key in metadata_keys:
        metadata_inputs[key] = torch.stack([item['metadata_input'][key] for item in batch])
    
    # Collect labels
    task_names = batch[0]['labels'].keys()
    labels = {}
    for task_name in task_names:
        labels[task_name] = torch.stack([item['labels'][task_name] for item in batch])
    
    result = {
        'text_input': text_inputs,
        'metadata_input': metadata_inputs,
        'labels': labels
    }
    
    if attention_masks is not None:
        result['attention_mask'] = attention_masks
    
    return result

class DynamicLossWeighting:
    """
    Dynamic Loss Weighting cho Multi-Task Learning
    """
    
    def __init__(self, task_names: List[str], method: str = "uncertainty", alpha: float = 0.5):
        """
        Args:
            task_names: List of task names
            method: "uncertainty", "gradnorm", "equal"
            alpha: Learning rate for weight updates
        """
        self.task_names = task_names
        self.method = method
        self.alpha = alpha
        
        # Initialize weights
        self.weights = {task: 1.0 for task in task_names}
        self.loss_history = {task: [] for task in task_names}
        self.prev_losses = {task: 0.0 for task in task_names}
        
        if method == "uncertainty":
            # Learnable uncertainty parameters
            self.log_vars = nn.Parameter(torch.zeros(len(task_names)))
    
    def compute_weighted_loss(self, losses: Dict[str, torch.Tensor], 
                            model: nn.Module = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss
        """
        if self.method == "equal":
            # Equal weighting
            total_loss = sum(losses.values())
            return total_loss, self.weights
        
        elif self.method == "uncertainty":
            # Uncertainty weighting (Kendall et al.)
            total_loss = 0
            for i, (task, loss) in enumerate(losses.items()):
                precision = torch.exp(-self.log_vars[i])
                total_loss += precision * loss + self.log_vars[i]
            
            return total_loss, self.weights
        
        elif self.method == "gradnorm":
            # GradNorm (Chen et al.)
            if model is None:
                # Fallback to equal weighting
                total_loss = sum(losses.values())
                return total_loss, self.weights
            
            # Compute weighted loss
            weighted_losses = []
            for task in self.task_names:
                weighted_losses.append(self.weights[task] * losses[task])
            
            total_loss = sum(weighted_losses)
            
            # Update weights based on gradient norms (simplified version)
            self._update_gradnorm_weights(losses, model)
            
            return total_loss, self.weights
    
    def _update_gradnorm_weights(self, losses: Dict[str, torch.Tensor], model: nn.Module):
        """
        Update weights using GradNorm algorithm (simplified)
        """
        # This is a simplified version - full GradNorm requires more complex implementation
        for task in self.task_names:
            current_loss = losses[task].item()
            self.loss_history[task].append(current_loss)
            
            if len(self.loss_history[task]) > 1:
                # Simple heuristic: increase weight if loss is increasing
                loss_change = current_loss - self.prev_losses[task]
                if loss_change > 0:
                    self.weights[task] = min(self.weights[task] * 1.1, 5.0)
                else:
                    self.weights[task] = max(self.weights[task] * 0.95, 0.1)
            
            self.prev_losses[task] = current_loss

class MultiTaskTrainer:
    """
    Multi-Task Trainer cho Multi-Modal Fusion Network
    """
    
    def __init__(self, 
                 model: nn.Module,
                 task_configs: Dict[str, int],
                 loss_weighting_method: str = "uncertainty",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 save_dir: str = "./models/multimodal_fusion"):
        """
        Args:
            model: MultiModalFusionNetwork instance
            task_configs: Dict mapping task names to number of classes
            loss_weighting_method: "uncertainty", "gradnorm", "equal"
            device: Training device
            save_dir: Directory to save models and logs
        """
        self.model = model.to(device)
        self.task_configs = task_configs
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss weighting
        self.loss_weighting = DynamicLossWeighting(
            task_names=list(task_configs.keys()),
            method=loss_weighting_method
        )
          # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging"""
        log_file = self.save_dir / "training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer) -> Dict[str, float]:
        """
        Train one epoch
        """
        self.model.train()
        epoch_losses = defaultdict(list)
        epoch_accuracies = defaultdict(list)
        
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            text_input = batch['text_input'].to(self.device)
            metadata_input = {k: v.to(self.device) for k, v in batch['metadata_input'].items()}
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(text_input, metadata_input, attention_mask)
            
            # Compute losses for each task
            task_losses = {}
            task_accuracies = {}
            
            for task_name, logits in outputs.items():
                if task_name in labels:
                    task_loss = self.criterion(logits, labels[task_name])
                    task_losses[task_name] = task_loss
                    
                    # Compute accuracy
                    predictions = torch.argmax(logits, dim=1)
                    accuracy = (predictions == labels[task_name]).float().mean()
                    task_accuracies[task_name] = accuracy
                    
                    epoch_losses[task_name].append(task_loss.item())
                    epoch_accuracies[task_name].append(accuracy.item())
            
            # Compute weighted loss
            total_loss, loss_weights = self.loss_weighting.compute_weighted_loss(task_losses, self.model)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Log progress
            if batch_idx % 100 == 0:
                self.logger.info(f"Batch {batch_idx}/{total_batches} - Total Loss: {total_loss.item():.4f}")
                for task_name, loss in task_losses.items():
                    self.logger.info(f"  {task_name}: Loss={loss.item():.4f}, Acc={task_accuracies[task_name].item():.4f}")
        
        # Compute epoch averages
        epoch_results = {}
        for task_name in self.task_configs.keys():
            if task_name in epoch_losses:
                epoch_results[f"{task_name}_loss"] = np.mean(epoch_losses[task_name])
                epoch_results[f"{task_name}_accuracy"] = np.mean(epoch_accuracies[task_name])
        
        epoch_results["total_loss"] = sum(epoch_results[f"{task}_loss"] for task in self.task_configs.keys() if f"{task}_loss" in epoch_results)
        
        return epoch_results
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate one epoch
        """
        self.model.eval()
        epoch_losses = defaultdict(list)
        epoch_accuracies = defaultdict(list)
        
        all_predictions = defaultdict(list)
        all_labels = defaultdict(list)
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                text_input = batch['text_input'].to(self.device)
                metadata_input = {k: v.to(self.device) for k, v in batch['metadata_input'].items()}
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                outputs = self.model(text_input, metadata_input, attention_mask)
                
                # Compute losses and metrics
                for task_name, logits in outputs.items():
                    if task_name in labels:
                        task_loss = self.criterion(logits, labels[task_name])
                        epoch_losses[task_name].append(task_loss.item())
                        
                        # Predictions and accuracy
                        predictions = torch.argmax(logits, dim=1)
                        accuracy = (predictions == labels[task_name]).float().mean()
                        epoch_accuracies[task_name].append(accuracy.item())
                        
                        # Store for detailed metrics
                        all_predictions[task_name].extend(predictions.cpu().numpy())
                        all_labels[task_name].extend(labels[task_name].cpu().numpy())
        
        # Compute epoch averages
        epoch_results = {}
        for task_name in self.task_configs.keys():
            if task_name in epoch_losses:
                epoch_results[f"{task_name}_loss"] = np.mean(epoch_losses[task_name])
                epoch_results[f"{task_name}_accuracy"] = np.mean(epoch_accuracies[task_name])
                
                # Compute F1 score
                if task_name in all_predictions:
                    f1 = f1_score(all_labels[task_name], all_predictions[task_name], average='weighted')
                    epoch_results[f"{task_name}_f1"] = f1
        
        epoch_results["total_loss"] = sum(epoch_results[f"{task}_loss"] for task in self.task_configs.keys() if f"{task}_loss" in epoch_results)
        
        return epoch_results, all_predictions, all_labels
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              num_epochs: int = 50,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-5,
              patience: int = 10,
              save_best: bool = True) -> Dict[str, List[float]]:
        """
        Main training loop
        """
        # Optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience//2, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Tasks: {list(self.task_configs.keys())}")
        self.logger.info(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Training...")
            train_results = self.train_epoch(train_loader, optimizer)
            
            # Validation
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - Validation...")
            val_results, val_predictions, val_labels = self.validate_epoch(val_loader)
            
            # Update learning rate
            scheduler.step(val_results['total_loss'])
            
            # Log results
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
            self.logger.info(f"Train Loss: {train_results['total_loss']:.4f}, Val Loss: {val_results['total_loss']:.4f}")
            
            for task_name in self.task_configs.keys():
                if f"{task_name}_accuracy" in train_results and f"{task_name}_accuracy" in val_results:
                    self.logger.info(f"  {task_name}: Train Acc={train_results[f'{task_name}_accuracy']:.4f}, "
                                   f"Val Acc={val_results[f'{task_name}_accuracy']:.4f}")
            
            # Save history
            for key, value in train_results.items():
                self.train_history[key].append(value)
            for key, value in val_results.items():
                self.val_history[key].append(value)
            
            # Early stopping and model saving
            if val_results['total_loss'] < best_val_loss:
                best_val_loss = val_results['total_loss']
                patience_counter = 0
                
                if save_best:
                    self.save_model(epoch, val_results, "best_model.pth")
                    self.logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_model(epoch, val_results, f"checkpoint_epoch_{epoch+1}.pth")
        
        # Save final model
        self.save_model(epoch, val_results, "final_model.pth")
        
        # Save training history
        self._save_training_history()
        
        # Generate training plots
        self._plot_training_history()
        
        return {
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history)
        }
    
    def save_model(self, epoch: int, metrics: Dict[str, float], filename: str):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'task_configs': self.task_configs,
            'metrics': metrics,
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history)
        }
        
        # Save loss weighting parameters if using uncertainty method
        if hasattr(self.loss_weighting, 'log_vars'):
            checkpoint['loss_weighting_log_vars'] = self.loss_weighting.log_vars.data
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def _save_training_history(self):
        """
        Save training history to JSON
        """
        history = {
            'train_history': {k: [float(x) for x in v] for k, v in self.train_history.items()},
            'val_history': {k: [float(x) for x in v] for k, v in self.val_history.items()}
        }
        
        with open(self.save_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    def _plot_training_history(self):
        """
        Plot training history
        """
        plt.style.use('seaborn-v0_8')
        
        # Plot losses
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Total loss
        axes[0, 0].plot(self.train_history['total_loss'], label='Train')
        axes[0, 0].plot(self.val_history['total_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Task-specific losses
        task_names = list(self.task_configs.keys())
        if len(task_names) > 0:
            for i, task_name in enumerate(task_names[:3]):  # Show first 3 tasks
                row = (i + 1) // 2
                col = (i + 1) % 2
                if row < 2 and col < 2:
                    train_key = f"{task_name}_loss"
                    val_key = f"{task_name}_loss"
                    if train_key in self.train_history and val_key in self.val_history:
                        axes[row, col].plot(self.train_history[train_key], label='Train')
                        axes[row, col].plot(self.val_history[val_key], label='Validation')
                        axes[row, col].set_title(f'{task_name} Loss')
                        axes[row, col].set_xlabel('Epoch')
                        axes[row, col].set_ylabel('Loss')
                        axes[row, col].legend()
                        axes[row, col].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "training_losses.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot accuracies
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Accuracies', fontsize=16)
        
        for i, task_name in enumerate(task_names[:4]):  # Show first 4 tasks
            row = i // 2
            col = i % 2
            train_key = f"{task_name}_accuracy"
            val_key = f"{task_name}_accuracy"
            
            if train_key in self.train_history and val_key in self.val_history:
                axes[row, col].plot(self.train_history[train_key], label='Train')
                axes[row, col].plot(self.val_history[val_key], label='Validation')
                axes[row, col].set_title(f'{task_name} Accuracy')
                axes[row, col].set_xlabel('Epoch')
                axes[row, col].set_ylabel('Accuracy')
                axes[row, col].legend()
                axes[row, col].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / "training_accuracies.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Training plots saved successfully")
    
    def train_step(self, text_input, metadata_input, targets, optimizer=None):
        """
        Perform a single training step
        Args:
            text_input: Text input tensor
            metadata_input: Metadata input dict
            targets: Target labels dict
            optimizer: Optional optimizer (creates AdamW if None)
        Returns:
            float: Total loss value
        """
        # Create optimizer if not provided
        if optimizer is None:
            optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        
        self.model.train()
        
        # Forward pass
        optimizer.zero_grad()
        outputs = self.model(text_input, metadata_input)
        
        # Compute losses for each task
        task_losses = {}
        for task_name, logits in outputs.items():
            if task_name in targets:
                task_loss = self.criterion(logits, targets[task_name])
                task_losses[task_name] = task_loss
        
        # Compute weighted loss
        total_loss, loss_weights = self.loss_weighting.compute_weighted_loss(task_losses, self.model)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
