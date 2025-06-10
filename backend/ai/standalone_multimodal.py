"""
Standalone Multi-Modal Fusion Network Implementation
Working version without external transformers dependency
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# =================== DATA GENERATION ===================
class SimpleGitHubDataGenerator:
    """Simplified GitHub data generator"""
    
    def __init__(self):
        self.commit_patterns = [
            "fix: resolve {} issue",
            "feat: add {} functionality", 
            "docs: update {} documentation",
            "refactor: optimize {} code",
            "test: add {} tests",
            "chore: update {} dependencies"
        ]
        
        self.technical_terms = [
            "authentication", "database", "API", "frontend", "backend",
            "security", "validation", "logging", "caching", "performance",
            "error handling", "user interface", "configuration", "deployment"
        ]
        
    def generate_commit_message(self) -> str:
        pattern = np.random.choice(self.commit_patterns)
        term = np.random.choice(self.technical_terms)
        return pattern.format(term)
    
    def generate_metadata(self) -> Dict[str, Any]:
        return {
            'lines_added': np.random.randint(1, 500),
            'lines_deleted': np.random.randint(0, 200),
            'files_changed': np.random.randint(1, 20),
            'commits_last_month': np.random.randint(0, 100),
            'author_experience': np.random.randint(1, 10),
            'is_weekend': np.random.choice([0, 1]),
            'hour_of_day': np.random.randint(0, 24),
            'file_types': np.random.choice(['py', 'js', 'java', 'cpp'], size=3).tolist()
        }
    
    def generate_labels(self, commit_msg: str, metadata: Dict) -> Dict[str, int]:
        # Simple rule-based label generation
        labels = {}
        
        # Risk assessment based on patterns
        risk_indicators = ['fix', 'bug', 'error', 'critical', 'urgent']
        labels['commit_risk'] = 1 if any(word in commit_msg.lower() for word in risk_indicators) else 0
        
        # Complexity based on file changes
        labels['complexity'] = 2 if metadata['files_changed'] > 10 else (1 if metadata['files_changed'] > 5 else 0)
        
        # Hotspot detection based on high activity
        labels['hotspot'] = 1 if metadata['commits_last_month'] > 50 else 0
        
        # Urgent review based on multiple factors
        urgent_score = (labels['commit_risk'] + 
                       (1 if metadata['lines_added'] > 200 else 0) + 
                       (1 if metadata['is_weekend'] else 0))
        labels['urgent_review'] = 1 if urgent_score >= 2 else 0
        
        return labels
    
    def generate_batch(self, num_samples: int) -> List[Dict]:
        samples = []
        for _ in range(num_samples):
            commit_msg = self.generate_commit_message()
            metadata = self.generate_metadata()
            labels = self.generate_labels(commit_msg, metadata)
            
            samples.append({
                'commit_message': commit_msg,
                'metadata': metadata,
                'labels': labels
            })
        
        return samples

# =================== TEXT PROCESSING ===================
class SimpleTextProcessor:
    """Simplified text processor without external dependencies"""
    
    def __init__(self, vocab_size: int = 5000, max_length: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.word_counts = Counter()
        self.vocab_built = False
        
    def clean_text(self, text: str) -> str:
        """Simple text cleaning"""
        import re
        # Remove special characters, keep only letters, numbers, spaces
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def build_vocabulary(self, texts: List[str]):
        """Build vocabulary from texts"""
        print("Building vocabulary...")
        
        for text in texts:
            cleaned = self.clean_text(text)
            words = cleaned.split()
            self.word_counts.update(words)
        
        # Keep most common words
        most_common = self.word_counts.most_common(self.vocab_size - 4)
        
        for i, (word, count) in enumerate(most_common):
            idx = i + 4
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        self.vocab_built = True
        print(f"Vocabulary built with {len(self.word_to_idx)} words")
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to tensor"""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        cleaned = self.clean_text(text)
        words = cleaned.split()
        
        # Convert to indices
        indices = [self.word_to_idx.get(word, 1) for word in words]  # 1 is UNK
        indices = [2] + indices + [3]  # Add START and END tokens
        
        # Pad or truncate
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices.extend([0] * (self.max_length - len(indices)))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def get_vocab_size(self) -> int:
        return len(self.word_to_idx)

# =================== METADATA PROCESSING ===================
class SimpleMetadataProcessor:
    """Simplified metadata processor"""
    
    def __init__(self):
        self.numerical_features = [
            'lines_added', 'lines_deleted', 'files_changed', 
            'commits_last_month', 'author_experience', 'hour_of_day'
        ]
        self.categorical_features = ['is_weekend']
        self.fitted = False
        self.scalers = {}
        
    def fit(self, metadata_list: List[Dict]):
        """Fit the processor on metadata"""
        print("Fitting metadata processor...")
        
        # Extract numerical features for normalization
        for feature in self.numerical_features:
            values = [m.get(feature, 0) for m in metadata_list]
            mean_val = np.mean(values)
            std_val = np.std(values) if np.std(values) > 0 else 1.0
            self.scalers[feature] = {'mean': mean_val, 'std': std_val}
        
        self.fitted = True
        print("Metadata processor fitted")
    
    def process_metadata(self, metadata: Dict) -> torch.Tensor:
        """Process metadata to tensor"""
        if not self.fitted:
            raise ValueError("Processor not fitted. Call fit first.")
        
        features = []
        
        # Numerical features (normalized)
        for feature in self.numerical_features:
            value = metadata.get(feature, 0)
            scaler = self.scalers[feature]
            normalized = (value - scaler['mean']) / scaler['std']
            features.append(normalized)
        
        # Categorical features
        for feature in self.categorical_features:
            value = metadata.get(feature, 0)
            features.append(float(value))
        
        return torch.tensor(features, dtype=torch.float32)

# =================== MODEL ARCHITECTURE ===================
class SimpleMultiModalFusion(nn.Module):
    """Simplified Multi-Modal Fusion Network"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, 
                 text_hidden_dim: int = 256, metadata_dim: int = 7,
                 fusion_dim: int = 256, num_classes_dict: Dict[str, int] = None):
        super().__init__()
        
        if num_classes_dict is None:
            num_classes_dict = {
                'commit_risk': 2, 'complexity': 3, 
                'hotspot': 2, 'urgent_review': 2
            }
        
        self.num_classes_dict = num_classes_dict
        
        # Text branch
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_lstm = nn.LSTM(embed_dim, text_hidden_dim, batch_first=True, bidirectional=True)
        self.text_proj = nn.Linear(text_hidden_dim * 2, fusion_dim)
        
        # Metadata branch
        self.metadata_layers = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, fusion_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(fusion_dim // 2, num_classes)
            for task, num_classes in num_classes_dict.items()
        })
    
    def forward(self, text_input: torch.Tensor, metadata_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Text processing
        text_embedded = self.embedding(text_input)
        text_lstm_out, _ = self.text_lstm(text_embedded)
        # Use mean pooling over sequence
        text_features = torch.mean(text_lstm_out, dim=1)
        text_features = self.text_proj(text_features)
        
        # Metadata processing
        metadata_features = self.metadata_layers(metadata_input)
        
        # Fusion
        fused_features = torch.cat([text_features, metadata_features], dim=1)
        fused = self.fusion(fused_features)
        
        # Task predictions
        outputs = {}
        for task, head in self.task_heads.items():
            outputs[task] = head(fused)
        
        return outputs

# =================== TRAINING ===================
class SimpleTrainer:
    """Simplified trainer for multi-task learning"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, data_loader, epoch: int):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(data_loader):
            self.optimizer.zero_grad()
            
            text_input = batch['text'].to(self.device)
            metadata_input = batch['metadata'].to(self.device)
            labels = {task: batch['labels'][task].to(self.device) 
                     for task in self.model.num_classes_dict.keys()}
            
            # Forward pass
            outputs = self.model(text_input, metadata_input)
            
            # Compute loss for each task
            loss = 0
            for task in outputs.keys():
                task_loss = self.criterion(outputs[task], labels[task])
                loss += task_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / len(data_loader)
    
    def evaluate(self, data_loader):
        self.model.eval()
        results = {task: {'predictions': [], 'targets': []} 
                  for task in self.model.num_classes_dict.keys()}
        
        with torch.no_grad():
            for batch in data_loader:
                text_input = batch['text'].to(self.device)
                metadata_input = batch['metadata'].to(self.device)
                labels = {task: batch['labels'][task].to(self.device) 
                         for task in self.model.num_classes_dict.keys()}
                
                outputs = self.model(text_input, metadata_input)
                
                for task in outputs.keys():
                    predictions = torch.argmax(outputs[task], dim=1)
                    results[task]['predictions'].extend(predictions.cpu().numpy())
                    results[task]['targets'].extend(labels[task].cpu().numpy())
        
        # Calculate metrics
        metrics = {}
        for task in results.keys():
            predictions = results[task]['predictions']
            targets = results[task]['targets']
            accuracy = accuracy_score(targets, predictions)
            metrics[task] = {'accuracy': accuracy}
        
        return metrics

# =================== DATASET ===================
class CommitDataset(torch.utils.data.Dataset):
    """Dataset for commit data"""
    
    def __init__(self, data: List[Dict], text_processor: SimpleTextProcessor, 
                 metadata_processor: SimpleMetadataProcessor):
        self.data = data
        self.text_processor = text_processor
        self.metadata_processor = metadata_processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        text_tensor = self.text_processor.encode_text(sample['commit_message'])
        metadata_tensor = self.metadata_processor.process_metadata(sample['metadata'])
        
        labels = {}
        for task, label in sample['labels'].items():
            labels[task] = torch.tensor(label, dtype=torch.long)
        
        return {
            'text': text_tensor,
            'metadata': metadata_tensor,
            'labels': labels
        }

# =================== MAIN PIPELINE ===================
def main():
    """Main training pipeline"""
    print("🚀 Starting Multi-Modal Fusion Network Training")
    
    # Configuration
    config = {
        'num_samples': 1000,
        'batch_size': 32,
        'num_epochs': 10,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    # Step 1: Generate data
    print("\n📊 Generating synthetic data...")
    generator = SimpleGitHubDataGenerator()
    data = generator.generate_batch(config['num_samples'])
    
    # Split data
    n_train = int(config['train_split'] * len(data))
    n_val = int(config['val_split'] * len(data))
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Step 2: Process data
    print("\n🔤 Setting up text processor...")
    text_processor = SimpleTextProcessor()
    train_texts = [sample['commit_message'] for sample in train_data]
    text_processor.build_vocabulary(train_texts)
    
    print("\n📋 Setting up metadata processor...")
    metadata_processor = SimpleMetadataProcessor()
    train_metadata = [sample['metadata'] for sample in train_data]
    metadata_processor.fit(train_metadata)
    
    # Step 3: Create datasets
    print("\n📦 Creating datasets...")
    train_dataset = CommitDataset(train_data, text_processor, metadata_processor)
    val_dataset = CommitDataset(val_data, text_processor, metadata_processor)
    test_dataset = CommitDataset(test_data, text_processor, metadata_processor)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Step 4: Create model
    print("\n🧠 Creating model...")
    model = SimpleMultiModalFusion(
        vocab_size=text_processor.get_vocab_size(),
        embed_dim=128,
        text_hidden_dim=256,
        metadata_dim=7,  # Number of metadata features
        fusion_dim=256
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 5: Train model
    print("\n🏃 Training model...")
    trainer = SimpleTrainer(model, config['device'])
    
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch + 1)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = trainer.evaluate(val_loader)
        print("Validation metrics:")
        for task, metrics in val_metrics.items():
            print(f"  {task}: {metrics}")
    
    # Step 6: Final evaluation
    print("\n📊 Final evaluation on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    print("\nFinal Test Results:")
    print("=" * 50)
    for task, metrics in test_metrics.items():
        print(f"{task:15}: Accuracy = {metrics['accuracy']:.4f}")
    
    # Average accuracy
    avg_accuracy = np.mean([metrics['accuracy'] for metrics in test_metrics.values()])
    print(f"{'Average':15}: Accuracy = {avg_accuracy:.4f}")
    
    print("\n🎉 Training completed successfully!")
    
    return model, test_metrics

if __name__ == "__main__":
    model, results = main()
