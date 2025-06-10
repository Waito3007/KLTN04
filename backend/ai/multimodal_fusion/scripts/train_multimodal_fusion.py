#!/usr/bin/env python3
"""
Training Script for Multimodal Fusion Model
Complete training pipeline for commit analysis with text + metadata fusion
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from collections import Counter
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def custom_collate_fn(batch):
    """Custom collate function to handle metadata dict structure"""
    collated = {}
    
    # Handle text features (simple tensors)
    collated['text_features'] = torch.stack([item['text_features'] for item in batch])
    
    # Handle metadata features (dict of tensors)
    metadata_keys = batch[0]['metadata_features'].keys()
    collated['metadata_features'] = {}
    for key in metadata_keys:
        collated['metadata_features'][key] = torch.stack([item['metadata_features'][key] for item in batch])
    
    # Handle labels (list of dicts)
    collated['labels'] = [item['labels'] for item in batch]
    
    # Handle text (list of strings)
    collated['text'] = [item['text'] for item in batch]
    
    return collated

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "ai"))

# Import multimodal components
from ai.multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
from ai.multimodal_fusion.data_preprocessing.text_processor import TextProcessor
from ai.multimodal_fusion.data_preprocessing.metadata_processor import MetadataProcessor
# from ai.multimodal_fusion.training.multitask_trainer import MultiTaskTrainer
# from ai.multimodal_fusion.losses.multi_task_losses import MultiTaskLoss
# from ai.multimodal_fusion.evaluation.metrics_calculator import MetricsCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multimodal_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultimodalDataset(Dataset):
    """Dataset for multimodal fusion training"""
    
    def __init__(self, samples, text_processor, metadata_processor, max_length=512):
        self.samples = samples
        self.text_processor = text_processor
        self.metadata_processor = metadata_processor
        self.max_length = max_length
        
        # Define label mappings for multimodal tasks
        self.task_labels = {
            'risk_prediction': {'low': 0, 'high': 1},
            'complexity_prediction': {'simple': 0, 'medium': 1, 'complex': 2},
            'hotspot_prediction': {'very_low': 0, 'low': 1, 'medium': 2, 'high': 3, 'very_high': 4},
            'urgency_prediction': {'normal': 0, 'urgent': 1}
        }
        
        logger.info(f"Created dataset with {len(samples)} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
          # Process text
        commit_text = sample.get('text', '') or sample.get('message', '')
        text_features = self.text_processor.encode_text_lstm(commit_text)
          # Process metadata
        metadata = self._extract_metadata(sample)
        metadata_features = self.metadata_processor.process_sample(sample)
        
        # Generate multimodal task labels
        labels = self._generate_multimodal_labels(sample, commit_text, metadata)
        
        return {
            'text_features': text_features,
            'metadata_features': metadata_features,
            'labels': labels,
            'text': commit_text
        }
    
    def _extract_metadata(self, sample):
        """Extract metadata features from sample"""
        return {
            'author': sample.get('author', 'unknown'),
            'files_changed': len(sample.get('files_changed', [])),
            'additions': sample.get('additions', 0),
            'deletions': sample.get('deletions', 0),
            'time_of_day': sample.get('time_of_day', 12),  # hour
            'day_of_week': sample.get('day_of_week', 1),   # 1-7
            'commit_size': sample.get('additions', 0) + sample.get('deletions', 0),
            'is_merge': 'merge' in sample.get('text', '').lower()
        }
    
    def _generate_multimodal_labels(self, sample, text, metadata):
        """Generate labels for multimodal tasks based on commit analysis"""
        labels = {}
        
        # Risk prediction (high/low) - based on commit patterns
        risk_score = self._calculate_risk_score(text, metadata)
        labels['risk_prediction'] = 1 if risk_score > 0.5 else 0
        
        # Complexity prediction (simple/medium/complex) - based on changes
        complexity = self._calculate_complexity(text, metadata)
        labels['complexity_prediction'] = complexity
        
        # Hotspot prediction (very_low to very_high) - based on file patterns
        hotspot = self._calculate_hotspot_score(text, metadata)
        labels['hotspot_prediction'] = hotspot
        
        # Urgency prediction (normal/urgent) - based on keywords
        urgency = self._calculate_urgency(text, metadata)
        labels['urgency_prediction'] = urgency
        
        return labels
    
    def _calculate_risk_score(self, text, metadata):
        """Calculate risk score from commit text and metadata"""
        risk_keywords = ['fix', 'bug', 'error', 'crash', 'security', 'vulnerability', 'critical']
        risk_score = 0.0
        
        text_lower = text.lower()
        for keyword in risk_keywords:
            if keyword in text_lower:
                risk_score += 0.2
        
        # Add metadata-based risk
        if metadata['commit_size'] > 1000:  # Large commits are risky
            risk_score += 0.2
        if metadata['files_changed'] > 10:  # Many files changed
            risk_score += 0.1
            
        return min(risk_score, 1.0)
    
    def _calculate_complexity(self, text, metadata):
        """Calculate complexity level (0=simple, 1=medium, 2=complex)"""
        commit_size = metadata['commit_size']
        files_changed = metadata['files_changed']
        
        if commit_size < 50 and files_changed <= 2:
            return 0  # simple
        elif commit_size < 500 and files_changed <= 10:
            return 1  # medium
        else:
            return 2  # complex
    
    def _calculate_hotspot_score(self, text, metadata):
        """Calculate hotspot prediction (0-4 scale)"""
        # Based on files changed and commit frequency patterns
        files_changed = metadata['files_changed']
        
        if files_changed <= 1:
            return 0  # very_low
        elif files_changed <= 3:
            return 1  # low
        elif files_changed <= 7:
            return 2  # medium
        elif files_changed <= 15:
            return 3  # high
        else:
            return 4  # very_high
    
    def _calculate_urgency(self, text, metadata):
        """Calculate urgency (0=normal, 1=urgent)"""
        urgent_keywords = ['urgent', 'critical', 'hotfix', 'emergency', 'asap', 'immediately']
        text_lower = text.lower()
        
        for keyword in urgent_keywords:
            if keyword in text_lower:
                return 1
        
        # Large commits on weekends might be urgent
        if metadata['day_of_week'] in [6, 7] and metadata['commit_size'] > 500:
            return 1
            
        return 0

def load_training_data(data_file):
    """Load and prepare training data"""
    logger.info(f"Loading training data from {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different data formats
    if 'data' in data:
        samples = data['data']
    elif isinstance(data, list):
        samples = data
    else:
        samples = [data]
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples

def setup_model_and_training(device, vocab_size=10000):
    """Setup model, optimizer, and training components"""
      # Model configuration
    config = {
        'text_encoder': {
            'vocab_size': vocab_size,
            'embedding_dim': 768,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.1,
            'max_length': 512,
            'method': 'lstm'
        },        'metadata_encoder': {
            'categorical_dims': {'author_encoded': 1000, 'season_encoded': 4},  # vocab sizes to match processor output
            'numerical_features': ['numerical_features'],  # single tensor from processor
            'embedding_dim': 128,
            'hidden_dim': 128,
            'dropout': 0.1
        },
        'fusion': {
            'method': 'cross_attention',
            'fusion_dim': 256,
            'dropout': 0.1
        },
        'task_heads': {
            'risk_prediction': {'num_classes': 2},
            'complexity_prediction': {'num_classes': 3},
            'hotspot_prediction': {'num_classes': 5},
            'urgency_prediction': {'num_classes': 2}
        }
    }
    
    # Initialize model
    model = MultiModalFusionNetwork(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")
      # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
      # Loss functions
    loss_fns = {
        'risk_prediction': nn.CrossEntropyLoss().to(device),
        'complexity_prediction': nn.CrossEntropyLoss().to(device),
        'hotspot_prediction': nn.CrossEntropyLoss().to(device),
        'urgency_prediction': nn.CrossEntropyLoss().to(device)
    }
    
    return model, optimizer, scheduler, loss_fns, config

def train_epoch(model, train_loader, optimizer, loss_fns, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    task_losses = {task: 0 for task in loss_fns.keys()}
    task_correct = {task: 0 for task in loss_fns.keys()}
    task_total = {task: 0 for task in loss_fns.keys()}
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Move data to device
        text_features = batch['text_features'].to(device)
        
        # Handle metadata features (dict of tensors from custom collate)
        metadata_features = {}
        for key, value in batch['metadata_features'].items():
            metadata_features[key] = value.to(device)
        
        # Forward pass
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(text_features, metadata_features)
                
                # Calculate losses
                batch_loss = 0
                for task, loss_fn in loss_fns.items():
                    labels = torch.tensor([sample[task] for sample in batch['labels']], device=device)
                    task_loss = loss_fn(outputs[task], labels)
                    batch_loss += task_loss
                    task_losses[task] += task_loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs[task], 1)
                    task_correct[task] += (predicted == labels).sum().item()
                    task_total[task] += labels.size(0)
            
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(text_features, metadata_features)
            
            # Calculate losses
            batch_loss = 0
            for task, loss_fn in loss_fns.items():
                labels = torch.tensor([sample[task] for sample in batch['labels']], device=device)
                task_loss = loss_fn(outputs[task], labels)
                batch_loss += task_loss
                task_losses[task] += task_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs[task], 1)
                task_correct[task] += (predicted == labels).sum().item()
                task_total[task] += labels.size(0)
            
            batch_loss.backward()
            optimizer.step()
        
        total_loss += batch_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{batch_loss.item():.4f}",
            'avg_loss': f"{total_loss/len(progress_bar):.4f}"
        })
    
    # Calculate average metrics
    avg_loss = total_loss / len(train_loader)
    task_accuracies = {task: task_correct[task] / task_total[task] for task in loss_fns.keys()}
    avg_accuracy = sum(task_accuracies.values()) / len(task_accuracies)
    
    return avg_loss, task_accuracies, avg_accuracy

def validate_epoch(model, val_loader, loss_fns, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    task_losses = {task: 0 for task in loss_fns.keys()}
    task_correct = {task: 0 for task in loss_fns.keys()}
    task_total = {task: 0 for task in loss_fns.keys()}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
            text_features = batch['text_features'].to(device)
            
            # Handle metadata features (dict of tensors from custom collate)
            metadata_features = {}
            for key, value in batch['metadata_features'].items():
                metadata_features[key] = value.to(device)
            
            # Forward pass
            outputs = model(text_features, metadata_features)
            
            # Calculate losses and metrics
            batch_loss = 0
            for task, loss_fn in loss_fns.items():
                labels = torch.tensor([sample[task] for sample in batch['labels']], device=device)
                task_loss = loss_fn(outputs[task], labels)
                batch_loss += task_loss
                task_losses[task] += task_loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs[task], 1)
                task_correct[task] += (predicted == labels).sum().item()
                task_total[task] += labels.size(0)
            
            total_loss += batch_loss.item()
    
    # Calculate average metrics
    avg_loss = total_loss / len(val_loader)
    task_accuracies = {task: task_correct[task] / task_total[task] for task in loss_fns.keys()}
    avg_accuracy = sum(task_accuracies.values()) / len(task_accuracies)
    
    return avg_loss, task_accuracies, avg_accuracy

def main():
    """Main training function"""
    logger.info("🚀 MULTIMODAL FUSION MODEL TRAINING")
    logger.info("=" * 60)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🔧 Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
    
    # Paths
    data_file = Path(__file__).parent.parent.parent / "training_data" / "sample_preview.json"
    output_dir = Path(__file__).parent.parent.parent / "trained_models" / "multimodal_fusion"
    log_dir = Path(__file__).parent.parent.parent / "training_logs" / "multimodal_fusion"
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if not data_file.exists():
        logger.error(f"❌ Dataset not found: {data_file}")
        return
    
    samples = load_training_data(data_file)
    
    # Take a subset for training (adjust as needed)
    if len(samples) > 10000:
        samples = samples[:10000]
        logger.info(f"Using subset of {len(samples)} samples for training")
      # Initialize processors
    text_processor = TextProcessor()
    metadata_processor = MetadataProcessor()
    
    # Build vocabulary from sample texts
    texts = [sample.get('text', '') or sample.get('message', '') for sample in samples]
    text_processor.build_vocab(texts, vocab_size=10000)
    
    # Fit metadata processor with samples
    logger.info("🔧 Fitting metadata processor...")
    metadata_processor.fit(samples)
    
    # Create dataset
    dataset = MultimodalDataset(samples, text_processor, metadata_processor)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"📊 Train samples: {len(train_dataset)}")
    logger.info(f"📊 Val samples: {len(val_dataset)}")
      # Data loaders
    batch_size = 32 if device.type == 'cuda' else 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    
    # Setup model and training
    model, optimizer, scheduler, loss_fns, config = setup_model_and_training(
        device, vocab_size=len(text_processor.vocab)
    )
    
    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Training loop
    num_epochs = 20
    best_val_accuracy = 0
    patience = 5
    patience_counter = 0
    
    logger.info(f"🏃 Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        logger.info(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_accuracies, train_avg_acc = train_epoch(
            model, train_loader, optimizer, loss_fns, device, scaler
        )
        
        # Validate
        val_loss, val_accuracies, val_avg_acc = validate_epoch(
            model, val_loader, loss_fns, device
        )
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Log metrics
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_avg_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_avg_acc:.4f}")
        
        for task in loss_fns.keys():
            logger.info(f"  {task}: Train {train_accuracies[task]:.4f}, Val {val_accuracies[task]:.4f}")
        
        # Save best model
        if val_avg_acc > best_val_accuracy:
            best_val_accuracy = val_avg_acc
            patience_counter = 0
            
            # Save model
            save_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'epoch': epoch + 1,
                'best_val_accuracy': best_val_accuracy,
                'val_accuracies': val_accuracies,
                'text_processor': text_processor,
                'metadata_processor': metadata_processor
            }
            
            torch.save(save_dict, output_dir / 'best_multimodal_fusion_model.pth')
            logger.info(f"💾 Saved best model (Val Acc: {best_val_accuracy:.4f})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"⏹️ Early stopping after {patience} epochs without improvement")
            break
    
    logger.info(f"\n🎉 Training completed!")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
    
    # Save final model
    final_save_dict = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_epoch': epoch + 1,
        'final_val_accuracy': val_avg_acc,
        'best_val_accuracy': best_val_accuracy,
        'text_processor': text_processor,
        'metadata_processor': metadata_processor
    }
    
    torch.save(final_save_dict, output_dir / 'final_multimodal_fusion_model.pth')
    logger.info(f"💾 Saved final model")

if __name__ == "__main__":
    main()
