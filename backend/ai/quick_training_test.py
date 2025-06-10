"""
Quick Training Test - Small Dataset
Tests the multimodal model training on a small subset to verify functionality
"""

import os
import sys
import torch
import torch.nn as nn
import json
import logging
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickTestDataset(Dataset):
    """Quick test dataset class"""
    
    def __init__(self, data, text_processor, metadata_processor):
        self.data = data
        self.text_processor = text_processor
        self.metadata_processor = metadata_processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        try:
            # Extract fields
            commit_message = sample.get('text', '')
            metadata = sample.get('metadata', {})
            labels = sample.get('labels', {})
            
            # Process text
            text_encoded = self.text_processor.encode_text_lstm(commit_message)
            
            # Extract enhanced features  
            enhanced_features = self.text_processor.extract_enhanced_features(commit_message)
            
            # Convert enhanced features to tensor (simplified)
            feature_values = []
            for key in ['length', 'word_count', 'char_count', 'digit_count', 'upper_count', 
                       'punctuation_count', 'has_commit_type', 'has_bug_keywords', 
                       'has_feature_keywords', 'has_doc_keywords', 'avg_word_length',
                       'max_word_length', 'unique_word_ratio', 'sentiment_polarity',
                       'sentiment_subjectivity']:
                val = enhanced_features.get(key, 0)
                if isinstance(val, bool):
                    val = float(val)
                elif isinstance(val, str):
                    val = 1.0 if val == 'positive' else (-1.0 if val == 'negative' else 0.0)
                feature_values.append(float(val))
            
            # Pad to 18 features if needed
            while len(feature_values) < 18:
                feature_values.append(0.0)
            enhanced_text_features = torch.tensor(feature_values[:18], dtype=torch.float32)
            
            # Simple metadata features
            files_changed = len(metadata.get('files_mentioned', [])) if isinstance(metadata.get('files_mentioned'), list) else 1
            metadata_features = torch.tensor([
                float(files_changed),
                0.0,  # insertions 
                0.0,  # deletions
                0.5,  # hour_of_day normalized
                0.3,  # day_of_week normalized
                0.0,  # is_merge
                1.0,  # commit_size_medium
                0.0,  # commit_size_small
                0.0,  # commit_size_large
                hash(metadata.get('author', 'unknown')) % 1000 / 1000.0  # author hash
            ], dtype=torch.float32)
            
            # Labels - convert to numeric
            def label_to_numeric(label_str):
                if label_str in ['low', 'simple']:
                    return 0
                elif label_str in ['medium', 'moderate']:
                    return 1
                elif label_str in ['high', 'complex']:
                    return 2
                else:
                    return 0
            
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
            return {
                'text_encoded': torch.zeros(128, dtype=torch.long),
                'enhanced_text_features': torch.zeros(18, dtype=torch.float32),
                'metadata_features': torch.zeros(10, dtype=torch.float32),
                'labels': torch.zeros(4, dtype=torch.long)
            }

def collate_fn(batch):
    """Collate function"""
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

def quick_training_test():
    """Quick training test on small dataset"""
    
    logger.info("🚀 Starting Quick Training Test...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load small subset of data
    data_path = os.path.join(current_dir, 'training_data', 'improved_100k_multimodal_training.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # Take small subset
    train_data = full_data['train_data'][:1000]  # Just 1000 samples
    val_data = full_data['val_data'][:200]       # Just 200 samples
    
    logger.info(f"Using subset: {len(train_data)} train, {len(val_data)} val")
    
    # Initialize processors
    from multimodal_fusion.data_preprocessing.minimal_enhanced_text_processor import MinimalEnhancedTextProcessor
    from multimodal_fusion.data_preprocessing.metadata_processor import MetadataProcessor
    
    text_processor = MinimalEnhancedTextProcessor(
        method="lstm", vocab_size=5000, max_length=128,
        enable_sentiment=True, enable_advanced_cleaning=True
    )
    
    # Fit on subset
    texts = [sample.get('text', '') for sample in train_data + val_data]
    text_processor.fit(texts)
    
    metadata_processor = MetadataProcessor()
    
    # Create datasets
    train_dataset = QuickTestDataset(train_data, text_processor, metadata_processor)
    val_dataset = QuickTestDataset(val_data, text_processor, metadata_processor)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    from multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
    
    model_config = {
        'text_encoder': {
            'vocab_size': text_processor.get_vocab_size(),
            'embedding_dim': 128,
            'hidden_dim': 64,
            'num_layers': 2,
            'method': 'lstm'
        },
        'metadata_encoder': {
            'categorical_dims': {'author': 1000},
            'numerical_features': ['files_changed', 'insertions', 'deletions', 'hour_of_day', 'day_of_week'],
            'embedding_dim': 64,
            'hidden_dim': 28  # 10 metadata + 18 enhanced features
        },
        'fusion': {
            'method': 'cross_attention',
            'fusion_dim': 256
        },
        'task_heads': {
            'risk_prediction': {'num_classes': 3},
            'complexity_prediction': {'num_classes': 3},
            'hotspot_prediction': {'num_classes': 3},
            'urgency_prediction': {'num_classes': 3}
        }
    }
    
    model = MultiModalFusionNetwork(config=model_config).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    # Quick training - 5 epochs
    logger.info("Starting quick training (5 epochs)...")
    
    for epoch in range(5):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            text_encoded = batch['text_encoded'].to(device)
            enhanced_text_features = batch['enhanced_text_features'].to(device)
            metadata_features = batch['metadata_features'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Combine features
            combined_features = torch.cat([metadata_features, enhanced_text_features], dim=1)
            metadata_input = {
                'numerical_features': combined_features,
                'author': torch.zeros(text_encoded.size(0), dtype=torch.long).to(device)
            }
            
            outputs = model(text_encoded, metadata_input)
            
            # Loss calculation
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
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_correct += correct_predictions
            train_total += total_predictions
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                text_encoded = batch['text_encoded'].to(device)
                enhanced_text_features = batch['enhanced_text_features'].to(device)
                metadata_features = batch['metadata_features'].to(device)
                labels = batch['labels'].to(device)
                
                combined_features = torch.cat([metadata_features, enhanced_text_features], dim=1)
                metadata_input = {
                    'numerical_features': combined_features,
                    'author': torch.zeros(text_encoded.size(0), dtype=torch.long).to(device)
                }
                
                outputs = model(text_encoded, metadata_input)
                
                total_loss = 0
                correct_predictions = 0
                total_predictions = 0
                
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
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = train_correct / max(train_total, 1)
        val_accuracy = val_correct / max(val_total, 1)
        
        logger.info(f"Epoch {epoch+1}/5 - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    logger.info("✅ Quick training test completed successfully!")
    
    # Final evaluation
    logger.info("📊 Final Model Evaluation:")
    logger.info(f"   • Final Training Accuracy: {train_accuracy:.4f}")
    logger.info(f"   • Final Validation Accuracy: {val_accuracy:.4f}")
    logger.info(f"   • Final Training Loss: {avg_train_loss:.4f}")
    logger.info(f"   • Final Validation Loss: {avg_val_loss:.4f}")
    
    return {
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'success': True
    }

if __name__ == "__main__":
    try:
        results = quick_training_test()
        print("\n" + "="*60)
        print("🎯 QUICK TRAINING TEST RESULTS")
        print("="*60)
        print(f"✅ Training Success: {results['success']}")
        print(f"📊 Training Accuracy: {results['train_accuracy']:.4f}")
        print(f"📊 Validation Accuracy: {results['val_accuracy']:.4f}")
        print(f"📉 Training Loss: {results['train_loss']:.4f}")
        print(f"📉 Validation Loss: {results['val_loss']:.4f}")
        print(f"🏗️ Model Parameters: {results['model_parameters']:,}")
        print("="*60)
        print("🎉 Quick test completed successfully!")
    except Exception as e:
        logger.error(f"Quick test failed: {e}")
        print(f"❌ Quick test failed: {e}")
        raise
