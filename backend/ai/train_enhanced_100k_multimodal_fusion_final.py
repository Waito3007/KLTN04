#!/usr/bin/env python3
"""
Enhanced 100K Multimodal Fusion Training Script - Final Version
This script handles the new data format structure and enhanced text processing.
"""

import os
import sys
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_100k_training_final.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def label_to_numeric(label_str):
    """Convert string labels to numeric values"""
    if isinstance(label_str, str):
        label_str = label_str.lower().strip()
        if label_str in ['low', 'simple']:
            return 0
        elif label_str in ['medium', 'moderate']:
            return 1
        elif label_str in ['high', 'complex']:
            return 2
        else:
            return 0  # Default to low
    return 0

class Enhanced100KDataset(Dataset):
    """Enhanced dataset for 100K samples with new data format"""
    
    def __init__(self, data, text_processor, metadata_processor):
        self.data = data
        self.text_processor = text_processor
        self.metadata_processor = metadata_processor
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        try:
            sample = self.data[idx]
            
            # Extract text - handle new format
            text = sample.get('text', sample.get('commit_message', ''))
            if not text:
                text = "Empty commit message"
            
            # Process text with enhanced features
            text_encoded = self.text_processor.encode(text)
            
            # Extract enhanced text features if available
            if hasattr(self.text_processor, 'extract_enhanced_features'):
                try:
                    enhanced_features = self.text_processor.extract_enhanced_features(text)
                    feature_values = []
                    for feature_name, feature_value in enhanced_features.items():
                        if isinstance(feature_value, (list, np.ndarray)):
                            feature_values.extend(feature_value)
                        else:
                            feature_values.append(float(feature_value))
                    
                    # Ensure we have exactly 18 features
                    while len(feature_values) < 18:
                        feature_values.append(0.0)
                    feature_values = feature_values[:18]
                    
                    enhanced_text_features = torch.tensor(feature_values, dtype=torch.float32)
                except Exception as e:
                    logger.warning(f"Error extracting enhanced features: {e}")
                    enhanced_text_features = torch.zeros(18, dtype=torch.float32)
            else:
                enhanced_text_features = torch.zeros(18, dtype=torch.float32)
            
            # Extract metadata from new structure
            metadata = sample.get('metadata', {})
            metadata_dict = {
                'author': metadata.get('author', 'unknown'),
                'files_changed': len(metadata.get('files_mentioned', [])) if isinstance(metadata.get('files_mentioned'), list) else 1,
                'insertions': 0,  # Not available in new format
                'deletions': 0,   # Not available in new format  
                'hour_of_day': 12,  # Default value
                'day_of_week': 1   # Default value
            }
            
            # Combine base metadata features with enhanced text features
            base_metadata = self.metadata_processor.process(metadata_dict)
            enhanced_values = enhanced_text_features.tolist() if len(enhanced_text_features.shape) > 0 else [0.0] * 18
            
            # Combine: base metadata (5 features) + enhanced text features (18 features) = 23 total
            combined_numerical = base_metadata['numerical_features'].tolist() + enhanced_values
            
            metadata_input = {
                'numerical_features': torch.tensor(combined_numerical, dtype=torch.float32),
                'author': base_metadata['author']
            }
            
            # Extract labels from new structure
            labels = sample.get('labels', {})
            labels_tensor = torch.tensor([
                label_to_numeric(labels.get('risk_prediction', 'low')),
                label_to_numeric(labels.get('complexity_prediction', 'simple')),
                label_to_numeric(labels.get('hotspot_prediction', 'low')),
                label_to_numeric(labels.get('urgency_prediction', 'low'))
            ], dtype=torch.long)
            
            return {
                'text_encoded': text_encoded,
                'metadata_input': metadata_input,
                'labels': labels_tensor
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # Return default values on error
            return {
                'text_encoded': torch.zeros(128, dtype=torch.long),
                'metadata_input': {
                    'numerical_features': torch.zeros(23, dtype=torch.float32),  # 5 base + 18 enhanced
                    'author': torch.tensor(0, dtype=torch.long)
                },
                'labels': torch.zeros(4, dtype=torch.long)
            }

def enhanced_collate_fn(batch):
    """Enhanced collate function for DataLoader"""
    text_encoded = torch.stack([item['text_encoded'] for item in batch])
    
    # Handle metadata input dict
    metadata_input = {
        'numerical_features': torch.stack([item['metadata_input']['numerical_features'] for item in batch]),
        'author': torch.stack([item['metadata_input']['author'] for item in batch])
    }
    
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'text_encoded': text_encoded,
        'metadata_input': metadata_input,
        'labels': labels
    }

def train_enhanced_100k_model():
    """Main training function with enhanced text processing"""
    
    logger.info("Starting Enhanced 100K Multimodal Fusion Training...")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    data_path = os.path.join(current_dir, 'training_data', 'improved_100k_multimodal_training.json')
    if not os.path.exists(data_path):
        logger.error(f"Training data not found at {data_path}")
        return
    
    logger.info("Loading training data...")
    with open(data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # Handle data format - check if already split
    if isinstance(full_data, dict) and 'train_data' in full_data and 'val_data' in full_data:
        logger.info("Found pre-split data (train_data/val_data)")
        train_data = full_data['train_data']
        val_data = full_data['val_data']
        all_data = train_data + val_data
    else:
        # Handle other data formats
        if isinstance(full_data, dict):
            if 'training_data' in full_data:
                all_data = full_data['training_data']
            elif 'samples' in full_data:
                all_data = full_data['samples']
            else:
                # If it's a dict with other structure, convert to list
                all_data = list(full_data.values()) if isinstance(list(full_data.values())[0], dict) else full_data
        else:
            all_data = full_data
        
        # Split data if not already split
        split_idx = int(0.8 * len(all_data))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
    
    logger.info(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Extract texts for vocabulary building
    texts = []
    for sample in all_data:
        if isinstance(sample, dict):
            text = sample.get('text', sample.get('commit_message', ''))
            if text:
                texts.append(text)
        elif isinstance(sample, str):
            texts.append(sample)
        else:
            continue
    
    logger.info(f"Loaded {len(texts)} commit messages for vocabulary building")
    
    # Initialize enhanced text processor
    logger.info("Initializing Enhanced Text Processor...")
    try:
        from multimodal_fusion.data_preprocessing.minimal_enhanced_text_processor import MinimalEnhancedTextProcessor
        text_processor = MinimalEnhancedTextProcessor(
            method="lstm",
            vocab_size=10000,
            max_length=128,
            enable_sentiment=True,
            enable_advanced_cleaning=True
        )
        logger.info("Enhanced Text Processor initialized")
    except ImportError as e:
        logger.error(f"Failed to import enhanced text processor: {e}")
        # Fallback to basic processor
        from multimodal_fusion.data_preprocessing.text_processor import TextProcessor
        text_processor = TextProcessor(method="lstm", vocab_size=10000, max_length=128)
        logger.info("Using basic text processor as fallback")
    
    # Fit text processor
    logger.info("Fitting text processor to training data...")
    text_processor.fit(texts)
    
    # Initialize metadata processor
    logger.info("Initializing Metadata Processor...")
    from multimodal_fusion.data_preprocessing.metadata_processor import MetadataProcessor
    metadata_processor = MetadataProcessor()
    
    # Create metadata samples for fitting based on new data structure
    metadata_samples = []
    for sample in all_data:
        if isinstance(sample, dict) and 'metadata' in sample:
            metadata = sample['metadata']
            metadata_samples.append({
                'author': metadata.get('author', 'unknown'),
                'files_changed': len(metadata.get('files_mentioned', [])) if isinstance(metadata.get('files_mentioned'), list) else 1,
                'insertions': 0,  # Not available in new format
                'deletions': 0,   # Not available in new format
                'hour_of_day': 12,  # Default value
                'day_of_week': 1    # Default value
            })
    
    if metadata_samples:
        logger.info("Fitting metadata processor...")
        metadata_processor.fit(metadata_samples)
    else:
        logger.warning("No metadata samples found for fitting")
    
    # Create datasets
    train_dataset = Enhanced100KDataset(train_data, text_processor, metadata_processor)
    val_dataset = Enhanced100KDataset(val_data, text_processor, metadata_processor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=enhanced_collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=enhanced_collate_fn,
        num_workers=0
    )
    
    # Initialize model
    logger.info("Initializing Enhanced Multimodal Fusion Model...")
    
    # Calculate dimensions
    sample_batch = next(iter(train_loader))
    text_dim = 64  # LSTM hidden dimension (fixed from model config)
    metadata_dim = sample_batch['metadata_input']['numerical_features'].shape[-1]
    
    logger.info(f"Text features dimension: {text_dim}")
    logger.info(f"Metadata features dimension: {metadata_dim}")
    
    # Model configuration matching the expected format
    model_config = {
        'text_encoder': {
            'vocab_size': text_processor.vocab_size,
            'embedding_dim': 128,
            'hidden_dim': 64,
            'num_layers': 2,
            'method': 'lstm'
        },
        'metadata_encoder': {
            'categorical_dims': {
                'author': 1000  # Simplified author encoding
            },
            'numerical_features': ['files_changed', 'insertions', 'deletions', 'hour_of_day', 'day_of_week'] + 
                                [f'enhanced_feature_{i}' for i in range(18)],
            'hidden_dim': metadata_dim
        },
        'fusion': {
            'method': 'cross_attention',
            'fusion_dim': 256
        },
        'task_heads': {
            'risk_prediction': {'num_classes': 3},      # low, medium, high
            'complexity_prediction': {'num_classes': 3}, # simple, moderate, complex  
            'hotspot_prediction': {'num_classes': 3},   # low, medium, high
            'urgency_prediction': {'num_classes': 3}    # low, medium, high
        }
    }
    
    from multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
    model = MultiModalFusionNetwork(config=model_config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move batch to device
                text_encoded = batch['text_encoded'].to(device)
                metadata_input = {k: v.to(device) for k, v in batch['metadata_input'].items()}
                labels = batch['labels'].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(text_encoded, metadata_input)
                
                # Calculate loss for each task
                total_loss = 0
                task_names = ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']
                
                for i, task_name in enumerate(task_names):
                    if task_name in outputs:
                        task_logits = outputs[task_name]
                        task_labels = labels[:, i]
                        task_loss = criterion(task_logits, task_labels)
                        total_loss += task_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
                train_batches += 1
                
                # Log progress
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss.item():.4f}")
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    # Move batch to device
                    text_encoded = batch['text_encoded'].to(device)
                    metadata_input = {k: v.to(device) for k, v in batch['metadata_input'].items()}
                    labels = batch['labels'].to(device)
                    
                    # Forward pass
                    outputs = model(text_encoded, metadata_input)
                    
                    # Calculate loss
                    total_loss = 0
                    task_names = ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']
                    
                    for i, task_name in enumerate(task_names):
                        if task_name in outputs:
                            task_logits = outputs[task_name]
                            task_labels = labels[:, i]
                            task_loss = criterion(task_logits, task_labels)
                            total_loss += task_loss
                            
                            # Calculate accuracy for this task
                            predicted = torch.argmax(task_logits, dim=1)
                            val_correct += (predicted == task_labels).sum().item()
                            val_total += task_labels.size(0)
                    
                    val_loss += total_loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Val Accuracy: {val_accuracy:.4f}")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(current_dir, 'models', 'enhanced_100k_multimodal_fusion_best.pth')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'model_config': model_config,
                'text_processor': text_processor,
                'metadata_processor': metadata_processor
            }, model_path)
            
            logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        logger.info("-" * 80)
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    return model, text_processor, metadata_processor

if __name__ == "__main__":
    try:
        model, text_processor, metadata_processor = train_enhanced_100k_model()
        logger.info("Enhanced 100K training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
