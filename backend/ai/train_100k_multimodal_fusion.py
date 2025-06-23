#!/usr/bin/env python3
"""
Training Script cho 100K Dataset Multimodal Fusion
==================================================

Script này sẽ train mô hình multimodal fusion với 100K samples từ dataset lớn.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, f1_score
from collections import Counter
import logging

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'multimodal_fusion'))

# Import multimodal components
from multimodal_fusion.data_preprocessing.text_processor import TextProcessor
from multimodal_fusion.data_preprocessing.metadata_processor import MetadataProcessor
from multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_logs/100k_multimodal_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Large100KDataset(Dataset):
    """Dataset class cho 100K training data"""
    
    def __init__(self, data, text_processor, metadata_processor, split='train'):
        self.data = data
        self.text_processor = text_processor
        self.metadata_processor = metadata_processor
        self.split = split
        
        # Task mapping
        self.task_configs = {
            'risk_prediction': {'labels': ['low', 'high'], 'type': 'classification'},
            'complexity_prediction': {'labels': ['simple', 'medium', 'complex'], 'type': 'classification'},
            'hotspot_prediction': {'labels': ['low', 'medium', 'high'], 'type': 'classification'},
            'urgency_prediction': {'labels': ['normal', 'urgent'], 'type': 'classification'}
        }
        
        logger.info(f"Created {split} dataset with {len(data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Process text
        text_features = self.text_processor.encode_text_lstm(sample['text'])
          # Process metadata
        metadata_features = self.metadata_processor.process_sample(sample['metadata'])
        
        # Process labels
        labels = {}
        for task, config in self.task_configs.items():
            label_str = sample['labels'][task]
            label_idx = config['labels'].index(label_str)
            labels[task] = torch.tensor(label_idx, dtype=torch.long)
        
        return {
            'text': text_features,  # Already returns torch.long from encode_text_lstm
            'metadata': metadata_features,
            'labels': labels,
            'sample_id': idx
        }

def custom_collate_fn(batch):
    """Custom collate function để handle metadata dict"""
    collated = {
        'text': torch.stack([item['text'] for item in batch]),
        'labels': {},
        'sample_ids': [item['sample_id'] for item in batch]
    }
    
    # Handle labels
    for task in ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']:
        collated['labels'][task] = torch.stack([item['labels'][task] for item in batch])
    
    # Handle metadata - collect all metadata dicts
    metadata_batch = {}
    first_metadata = batch[0]['metadata']
    
    for key in first_metadata.keys():
        if isinstance(first_metadata[key], torch.Tensor):
            metadata_batch[key] = torch.stack([item['metadata'][key] for item in batch])
        else:
            metadata_batch[key] = [item['metadata'][key] for item in batch]
    
    collated['metadata'] = metadata_batch
    return collated

class MultimodalTrainer100K:
    """Trainer cho 100K dataset"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create directories
        Path("trained_models/multimodal_fusion_100k").mkdir(parents=True, exist_ok=True)
        Path("training_logs").mkdir(exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def load_data(self, data_file="training_data/improved_100k_multimodal_training.json"):
        """Load 100K training data"""
        logger.info(f"Loading data from: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.train_data = data['train_data']
        self.val_data = data['val_data']
        
        logger.info(f"Loaded {len(self.train_data)} training samples")
        logger.info(f"Loaded {len(self.val_data)} validation samples")
        
        # Print label distribution
        self._print_label_distribution()
        
        return self.train_data, self.val_data
    
    def _print_label_distribution(self):
        """Print label distribution"""
        logger.info("Label distribution:")
        
        for task in ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']:
            train_labels = [sample['labels'][task] for sample in self.train_data]
            counter = Counter(train_labels)
            logger.info(f"  {task}: {dict(counter)}")
    
    def setup_processors(self):
        """Setup text and metadata processors"""
        logger.info("Setting up processors...")
        
        # Initialize processors
        self.text_processor = TextProcessor()
        self.metadata_processor = MetadataProcessor()
        
        # Collect all samples for fitting
        all_samples = self.train_data + self.val_data
        
        # Fit text processor
        texts = [sample['text'] for sample in all_samples]
        self.text_processor.fit(texts)
        
        # Fit metadata processor
        metadata_list = [sample['metadata'] for sample in all_samples]
        self.metadata_processor.fit(metadata_list)
        
        logger.info(f"Text vocabulary size: {len(self.text_processor.vocab)}")
        logger.info("Processors setup complete")
    
    def create_dataloaders(self):
        """Create data loaders"""
        logger.info("Creating data loaders...")
        
        # Create datasets
        train_dataset = Large100KDataset(
            self.train_data, 
            self.text_processor, 
            self.metadata_processor, 
            split='train'
        )
        
        val_dataset = Large100KDataset(
            self.val_data, 
            self.text_processor, 
            self.metadata_processor, 
            split='validation'
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=custom_collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=custom_collate_fn        )
        
        logger.info(f"Created train loader with {len(self.train_loader)} batches")
        logger.info(f"Created val loader with {len(self.val_loader)} batches")
    
    def setup_model(self):
        """Setup multimodal fusion model"""
        logger.info("Setting up model...")
        
        # Get feature dimensions from processors
        feature_dims = self.metadata_processor.get_feature_dimensions()
        
        # Model configuration
        model_config = {
            'text_encoder': {
                'vocab_size': len(self.text_processor.vocab),
                'embedding_dim': 128,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.3
            },            'metadata_encoder': {
                'categorical_dims': {
                    'author_encoded': feature_dims['author_vocab_size'],
                    'season_encoded': feature_dims['season_vocab_size']
                },
                'numerical_features': ['numerical_features'],
                'embedding_dim': 64,
                'hidden_dim': 128,
                'dropout': 0.3
            },
            'fusion': {
                'hidden_dim': 256,
                'dropout': 0.4
            },
            'task_heads': {
                'risk_prediction': {'num_classes': 2, 'type': 'classification'},
                'complexity_prediction': {'num_classes': 3, 'type': 'classification'},
                'hotspot_prediction': {'num_classes': 3, 'type': 'classification'},
                'urgency_prediction': {'num_classes': 2, 'type': 'classification'}
            }
        }
        
        # Create model
        self.model = MultiModalFusionNetwork(model_config)
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model created with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def setup_training(self):
        """Setup optimizer, scheduler, loss functions"""
        logger.info("Setting up training components...")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
          # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Loss functions
        self.loss_functions = {}
        for task in ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']:
            self.loss_functions[task] = nn.CrossEntropyLoss()
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        logger.info("Training setup complete")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        task_losses = {task: 0.0 for task in ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']}
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            text = batch['text'].to(self.device)
            metadata = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in batch['metadata'].items()}
            labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(text, metadata)
                    
                    # Calculate losses
                    losses = {}
                    for task in outputs.keys():
                        losses[task] = self.loss_functions[task](outputs[task], labels[task])
                    
                    # Combined loss
                    total_batch_loss = sum(losses.values())
                
                # Backward pass
                self.scaler.scale(total_batch_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(text, metadata)
                
                # Calculate losses
                losses = {}
                for task in outputs.keys():
                    losses[task] = self.loss_functions[task](outputs[task], labels[task])
                
                # Combined loss
                total_batch_loss = sum(losses.values())
                
                # Backward pass
                total_batch_loss.backward()
                self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_batch_loss.item()
            for task, loss in losses.items():
                task_losses[task] += loss.item()
            
            num_batches += 1
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                           f"Loss: {total_batch_loss.item():.4f}")
        
        # Calculate average losses
        avg_total_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}
        
        return avg_total_loss, avg_task_losses
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        task_losses = {task: 0.0 for task in ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']}
        task_predictions = {task: [] for task in ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']}
        task_targets = {task: [] for task in ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                text = batch['text'].to(self.device)
                metadata = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in batch['metadata'].items()}
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                
                # Forward pass
                outputs = self.model(text, metadata)
                
                # Calculate losses
                losses = {}
                for task in outputs.keys():
                    losses[task] = self.loss_functions[task](outputs[task], labels[task])
                    
                    # Collect predictions and targets
                    preds = torch.argmax(outputs[task], dim=1)
                    task_predictions[task].extend(preds.cpu().numpy())
                    task_targets[task].extend(labels[task].cpu().numpy())
                
                total_batch_loss = sum(losses.values())
                
                # Accumulate losses
                total_loss += total_batch_loss.item()
                for task, loss in losses.items():
                    task_losses[task] += loss.item()
                
                num_batches += 1
        
        # Calculate metrics
        avg_total_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}
        
        # Calculate accuracies
        task_accuracies = {}
        for task in task_predictions.keys():
            accuracy = accuracy_score(task_targets[task], task_predictions[task])
            task_accuracies[task] = accuracy
        
        overall_accuracy = np.mean(list(task_accuracies.values()))
        
        return avg_total_loss, avg_task_losses, task_accuracies, overall_accuracy
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = "trained_models/multimodal_fusion_100k/latest_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = "trained_models/multimodal_fusion_100k/best_model_100k.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(1, self.config['num_epochs'] + 1):
            logger.info(f"\nEpoch {epoch}/{self.config['num_epochs']}")
            
            # Train
            train_loss, train_task_losses = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_task_losses, val_accuracies, overall_acc = self.validate()
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Overall Val Accuracy: {overall_acc:.4f}")
            
            for task in val_accuracies.keys():
                logger.info(f"  {task}: {val_accuracies[task]:.4f}")
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Check for best model
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if overall_acc > self.best_val_acc:
                self.best_val_acc = overall_acc
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")

def main():
    """Main function"""
    print("🚀 TRAINING MULTIMODAL FUSION WITH 100K DATASET")
    print("=" * 70)
    
    # Training configuration
    config = {
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'patience': 7,
        'num_workers': 4 if torch.cuda.is_available() else 0
    }
    
    # Create trainer
    trainer = MultimodalTrainer100K(config)
    
    try:
        # Load data
        trainer.load_data()
        
        # Setup processors
        trainer.setup_processors()
        
        # Create data loaders
        trainer.create_dataloaders()
        
        # Setup model
        trainer.setup_model()
        
        # Setup training
        trainer.setup_training()
        
        # Train
        trainer.train()
        
        print("\n🎉 Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
