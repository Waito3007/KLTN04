"""
Enhanced 100K Training Script with NLTK Support - Fixed Version
Trains the multimodal fusion model with enhanced text processing capabilities
"""

import os
import sys
import torch
import torch.nn as nn
import json
import logging
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Setup logging with Unicode support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_100k_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Enhanced100KDataset(Dataset):
    """Dataset class for 100K enhanced training data"""
    
    def __init__(self, data, text_processor, metadata_processor, device='cpu'):
        self.data = data
        self.text_processor = text_processor
        self.metadata_processor = metadata_processor
        self.device = device
        
        logger.info(f"Initialized Enhanced100KDataset with {len(data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        try:
            # Extract fields from new data format
            commit_message = sample.get('text', '')
            metadata = sample.get('metadata', {})
            labels = sample.get('labels', {})
            
            # Process text with enhanced features
            text_encoded = self.text_processor.encode_text_lstm(commit_message)
            
            # Extract enhanced text features
            enhanced_features = self.text_processor.extract_enhanced_features(commit_message)
            
            # Convert enhanced features to tensor
            feature_values = []
            feature_keys = [
                'length', 'word_count', 'char_count', 'digit_count', 'upper_count', 'punctuation_count',
                'has_commit_type', 'has_bug_keywords', 'has_feature_keywords', 'has_doc_keywords',
                'has_technical_keywords', 'has_ui_keywords', 'has_testing_keywords',
                'avg_word_length', 'max_word_length', 'unique_word_ratio'
            ]
            
            # Add sentiment features if available
            if 'sentiment_polarity' in enhanced_features:
                feature_keys.extend(['sentiment_polarity', 'sentiment_subjectivity'])
            
            for key in feature_keys:
                val = enhanced_features.get(key, 0)
                if isinstance(val, bool):
                    val = float(val)
                elif isinstance(val, str):
                    val = 1.0 if val == 'positive' else (-1.0 if val == 'negative' else 0.0)
                feature_values.append(float(val))
            
            enhanced_text_features = torch.tensor(feature_values, dtype=torch.float32)
            
            # Process metadata - extract from nested metadata dict
            metadata_dict = {
                'author': metadata.get('author', 'unknown'),
                'files_changed': metadata.get('files_mentioned', []),  # Use files_mentioned as proxy
                'insertions': 0,  # Not available in new format
                'deletions': 0,   # Not available in new format
                'hour_of_day': 12,  # Default value
                'day_of_week': 1,   # Default value
                'is_merge': False,  # Default value
                'commit_size': 'medium',  # Default value
                'message_length': metadata.get('message_length', len(commit_message)),
                'word_count': metadata.get('word_count', len(commit_message.split())),
                'has_scope': metadata.get('has_scope', False),
                'is_conventional': metadata.get('is_conventional', False),
                'has_breaking': metadata.get('has_breaking', False)
            }
            
            # Create basic metadata features manually for compatibility
            files_changed_count = len(metadata_dict['files_changed']) if isinstance(metadata_dict['files_changed'], list) else 1
            metadata_features = torch.tensor([
                float(files_changed_count),
                float(metadata_dict.get('insertions', 0)),
                float(metadata_dict.get('deletions', 0)),
                float(metadata_dict.get('hour_of_day', 12) / 24.0),
                float(metadata_dict.get('day_of_week', 1) / 7.0),
                float(metadata_dict.get('is_merge', False)),
                1.0 if metadata_dict.get('commit_size') == 'small' else 0.0,
                1.0 if metadata_dict.get('commit_size') == 'medium' else 0.0,
                1.0 if metadata_dict.get('commit_size') == 'large' else 0.0,
                hash(metadata_dict.get('author', 'unknown')) % 1000 / 1000.0  # Simple author encoding
            ], dtype=torch.float32)
            
            # Labels - convert string labels to numeric
            def label_to_numeric(label_str):
                if label_str in ['low', 'simple']:
                    return 0
                elif label_str in ['medium', 'moderate']:
                    return 1
                elif label_str in ['high', 'complex']:
                    return 2
                else:
                    return 0  # Default to low
            
            labels_tensor = torch.tensor([
                label_to_numeric(labels.get('risk_prediction', 'low')),
                label_to_numeric(labels.get('complexity_prediction', 'simple')),
                label_to_numeric(labels.get('hotspot_prediction', 'low')),
                label_to_numeric(labels.get('urgency_prediction', 'low'))
            ], dtype=torch.long)
            
            return {
                'text_encoded': text_encoded,
                'enhanced_text_features': enhanced_text_features,
                'metadata_features': metadata_features,
                'labels': labels_tensor
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # Return default values on error
            return {
                'text_encoded': torch.zeros(128, dtype=torch.long),
                'enhanced_text_features': torch.zeros(18, dtype=torch.float32),  # Adjusted size
                'metadata_features': torch.zeros(10, dtype=torch.float32),
                'labels': torch.zeros(4, dtype=torch.long)
            }

def enhanced_collate_fn(batch):
    """Enhanced collate function for DataLoader"""
    text_encoded = torch.stack([item['text_encoded'] for item in batch])
    enhanced_text_features = torch.stack([item['enhanced_text_features'] for item in batch])
    metadata_features = torch.stack([item['metadata_features'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'text_encoded': text_encoded,
        'enhanced_text_features': enhanced_text_features,
        'metadata_features': metadata_features,
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
                all_data = list(full_data.values()) if all(isinstance(v, dict) for v in full_data.values()) else [full_data]
        else:
            all_data = full_data
        
        # Split data manually if not pre-split
        train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42, stratify=None)

    logger.info(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Extract text data for vocabulary building from all samples
    texts = []
    for sample in all_data:
        if isinstance(sample, dict):
            # Use 'text' field for new data format, fallback to 'commit_message' for old format
            texts.append(sample.get('text', sample.get('commit_message', '')))
        elif isinstance(sample, str):
            texts.append(sample)
        else:
            logger.warning(f"Unexpected sample format: {type(sample)}")
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
                'hour_of_day': 12,  # Default
                'day_of_week': 1,   # Default
                'is_merge': False,  # Default
                'commit_size': 'medium'  # Default
            })
        else:
            # Fallback for old format
            metadata_samples.append({
                'author': sample.get('author', 'unknown'),
                'files_changed': sample.get('files_changed', 1),
                'insertions': sample.get('insertions', 0),
                'deletions': sample.get('deletions', 0),
                'hour_of_day': sample.get('hour_of_day', 12),
                'day_of_week': sample.get('day_of_week', 1),
                'is_merge': sample.get('is_merge', False),
                'commit_size': sample.get('commit_size', 'medium')
            })
    
    metadata_processor.fit(metadata_samples)

    # Data is already split or was split above
    logger.info("Using data splits...")
    logger.info(f"Final - Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = Enhanced100KDataset(train_data, text_processor, metadata_processor, device)
    val_dataset = Enhanced100KDataset(val_data, text_processor, metadata_processor, device)
    
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
    try:
        from multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
    except ImportError as e:
        logger.error(f"Could not import MultiModalFusionNetwork: {e}")
        return
    
    # Get enhanced feature dimensions
    sample_batch = next(iter(train_loader))
    enhanced_text_feature_dim = sample_batch['enhanced_text_features'].shape[1]
    metadata_feature_dim = sample_batch['metadata_features'].shape[1]
    
    logger.info(f"Enhanced text features dimension: {enhanced_text_feature_dim}")
    logger.info(f"Metadata features dimension: {metadata_feature_dim}")
    
    # Model configuration using the new config-based approach
    model_config = {
        'text_encoder': {
            'vocab_size': text_processor.get_vocab_size(),
            'embedding_dim': 128,
            'hidden_dim': 64,
            'num_layers': 2,
            'method': 'lstm'
        },
        'metadata_encoder': {
            'categorical_dims': {
                'author': 1000  # Simplified author encoding
            },
            'numerical_features': ['files_changed', 'insertions', 'deletions', 'hour_of_day', 'day_of_week'],
            'embedding_dim': 64,
            'hidden_dim': metadata_feature_dim
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
    
    model = MultiModalFusionNetwork(config=model_config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training parameters
    epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Training history
    train_history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'learning_rate': []
    }
    
    # Create output directory
    output_dir = os.path.join(current_dir, 'trained_models', 'enhanced_multimodal_fusion_100k')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting training loop...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move batch to device
                text_encoded = batch['text_encoded'].to(device)
                enhanced_text_features = batch['enhanced_text_features'].to(device)
                metadata_features = batch['metadata_features'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                
                # Prepare metadata input as dict for the model - combine enhanced features with basic metadata
                combined_features = torch.cat([metadata_features, enhanced_text_features], dim=1)
                metadata_input = {
                    'numerical_features': combined_features,
                    'author': torch.zeros(text_encoded.size(0), dtype=torch.long).to(device)  # Dummy author
                }
                
                outputs = model(text_encoded, metadata_input)
                
                # Calculate loss for all tasks
                total_loss = 0
                correct_predictions = 0
                total_predictions = 0
                
                task_names = ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']
                for task_idx, task_name in enumerate(task_names):
                    if task_name in outputs:
                        task_output = outputs[task_name]
                        task_labels = labels[:, task_idx]
                        task_loss = criterion(task_output, task_labels)
                        total_loss += task_loss
                        
                        # Calculate accuracy
                        _, predicted = torch.max(task_output.data, 1)
                        correct_predictions += (predicted == task_labels).sum().item()
                        total_predictions += task_labels.size(0)
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
                train_correct += correct_predictions
                train_total += total_predictions
                
                if batch_idx % 100 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                              f"Loss: {total_loss.item():.4f}, LR: {current_lr:.2e}")
                    
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    text_encoded = batch['text_encoded'].to(device)
                    enhanced_text_features = batch['enhanced_text_features'].to(device)
                    metadata_features = batch['metadata_features'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Prepare metadata input as dict for the model
                    combined_features = torch.cat([metadata_features, enhanced_text_features], dim=1)
                    metadata_input = {
                        'numerical_features': combined_features,
                        'author': torch.zeros(text_encoded.size(0), dtype=torch.long).to(device)
                    }
                    
                    outputs = model(text_encoded, metadata_input)
                    
                    total_loss = 0
                    correct_predictions = 0
                    total_predictions = 0
                    
                    task_names = ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']
                    for task_idx, task_name in enumerate(task_names):
                        if task_name in outputs:
                            task_output = outputs[task_name]
                            task_labels = labels[:, task_idx]
                            task_loss = criterion(task_output, task_labels)
                            total_loss += task_loss
                            
                            _, predicted = torch.max(task_output.data, 1)
                            correct_predictions += (predicted == task_labels).sum().item()
                            total_predictions += task_labels.size(0)
                    
                    val_loss += total_loss.item()
                    val_correct += correct_predictions
                    val_total += total_predictions
                    
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = train_correct / max(train_total, 1)
        val_accuracy = val_correct / max(val_total, 1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        train_history['train_accuracy'].append(train_accuracy)
        train_history['val_accuracy'].append(val_accuracy)
        train_history['learning_rate'].append(current_lr)
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            model_path = os.path.join(output_dir, 'best_enhanced_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_history': train_history,
                'text_processor_vocab': text_processor.vocab,
                'model_config': model_config
            }, model_path)
            logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Save final training history
    history_path = os.path.join(output_dir, 'enhanced_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    
    logger.info("Enhanced training completed successfully!")
    logger.info(f"Models and history saved to: {output_dir}")
    
    return model, train_history

if __name__ == "__main__":
    try:
        model, history = train_enhanced_100k_model()
        print("Enhanced 100K training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
