#!/usr/bin/env python3
"""
Train HAN Model với GitHub Commits Dataset
Script đơn giản để train mô hình HAN với dữ liệu từ GitHub commits
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import re

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

class SimpleHANModel(nn.Module):
    """
    Simplified Hierarchical Attention Network for commit classification
    """
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=None):
        super(SimpleHANModel, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Word-level LSTM
        self.word_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Word-level attention
        self.word_attention = nn.Linear(hidden_dim * 2, 1)
        
        # Sentence-level LSTM
        self.sentence_lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        
        # Sentence-level attention
        self.sentence_attention = nn.Linear(hidden_dim * 2, 1)
        
        # Multi-task classification heads
        self.classifiers = nn.ModuleDict()
        if num_classes:
            for task, num_class in num_classes.items():
                self.classifiers[task] = nn.Linear(hidden_dim * 2, num_class)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, max_sentences, max_words = input_ids.size()
        
        # Reshape for word-level processing
        input_ids = input_ids.view(-1, max_words)  # (batch_size * max_sentences, max_words)
        
        # Word embeddings
        embedded = self.embedding(input_ids)  # (batch_size * max_sentences, max_words, embed_dim)
        
        # Word-level LSTM
        word_output, _ = self.word_lstm(embedded)  # (batch_size * max_sentences, max_words, hidden_dim * 2)
        
        # Word-level attention
        word_attention_weights = torch.softmax(self.word_attention(word_output), dim=1)
        sentence_vectors = torch.sum(word_attention_weights * word_output, dim=1)  # (batch_size * max_sentences, hidden_dim * 2)
        
        # Reshape back to sentence level
        sentence_vectors = sentence_vectors.view(batch_size, max_sentences, -1)  # (batch_size, max_sentences, hidden_dim * 2)
        
        # Sentence-level LSTM
        sentence_output, _ = self.sentence_lstm(sentence_vectors)  # (batch_size, max_sentences, hidden_dim * 2)
        
        # Sentence-level attention
        sentence_attention_weights = torch.softmax(self.sentence_attention(sentence_output), dim=1)
        document_vector = torch.sum(sentence_attention_weights * sentence_output, dim=1)  # (batch_size, hidden_dim * 2)
        
        document_vector = self.dropout(document_vector)
        
        # Multi-task outputs
        outputs = {}
        for task, classifier in self.classifiers.items():
            outputs[task] = classifier(document_vector)
        
        return outputs

class CommitDataset(Dataset):
    """Dataset class for commit messages"""
    
    def __init__(self, texts, labels, tokenizer, max_sentences=10, max_words=50):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_sentences = max_sentences
        self.max_words = max_words
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        # Tokenize text to sentences and words
        input_ids = self.tokenizer.encode_text(text, self.max_sentences, self.max_words)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': labels  # This will be a dictionary
        }

class SimpleTokenizer:
    """Simple tokenizer for commit messages"""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = Counter()
        
    def fit(self, texts):
        """Build vocabulary from texts"""
        print("🔤 Building vocabulary...")
        
        for text in texts:
            # Split into sentences
            sentences = self.split_sentences(text)
            for sentence in sentences:
                words = self.tokenize_words(sentence)
                self.word_counts.update(words)
        
        # Keep most frequent words
        most_common = self.word_counts.most_common(self.vocab_size - 2)
        
        for i, (word, count) in enumerate(most_common):
            idx = i + 2  # Start from 2 (after PAD and UNK)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        print(f"✅ Vocabulary built with {len(self.word_to_idx)} words")
        
    def split_sentences(self, text):
        """Split text into sentences"""
        # Simple sentence splitting for commit messages
        sentences = re.split(r'[.!?;]|\\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences if sentences else [text]
    
    def tokenize_words(self, sentence):
        """Tokenize sentence into words"""
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', sentence.lower())
        return words
    
    def encode_text(self, text, max_sentences, max_words):
        """Encode text to token ids"""
        sentences = self.split_sentences(text)
        
        # Pad or truncate sentences
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        while len(sentences) < max_sentences:
            sentences.append("")
        
        encoded_sentences = []
        for sentence in sentences:
            words = self.tokenize_words(sentence)
            
            # Convert words to indices
            word_ids = []
            for word in words:
                word_ids.append(self.word_to_idx.get(word, 1))  # 1 is UNK
            
            # Pad or truncate words
            if len(word_ids) > max_words:
                word_ids = word_ids[:max_words]
            while len(word_ids) < max_words:
                word_ids.append(0)  # 0 is PAD
            
            encoded_sentences.append(word_ids)
        
        return encoded_sentences

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = [item['labels'] for item in batch]  # Keep as list of dicts
    
    return {
        'input_ids': input_ids,
        'labels': labels
    }

def load_github_dataset(data_file):
    """Load GitHub commits dataset"""
    print(f"📖 Loading dataset: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'data' not in data:
        raise ValueError("Invalid dataset format: missing 'data' field")
    
    samples = data['data']
    print(f"📊 Loaded {len(samples)} samples")
    
    # Extract texts and labels
    texts = []
    labels = []
    
    for sample in samples:
        texts.append(sample['text'])
        labels.append(sample['labels'])
    
    return texts, labels, data.get('metadata', {})

def prepare_label_encoders(labels):
    """Prepare label encoders for multi-task classification"""
    print("🏷️  Preparing label encoders...")
    
    # Get all unique labels for each task
    label_sets = {}
    for label_dict in labels:
        for task, label in label_dict.items():
            if task not in label_sets:
                label_sets[task] = set()
            label_sets[task].add(label)
    
    # Create mappings
    label_encoders = {}
    num_classes = {}
    
    for task, label_set in label_sets.items():
        sorted_labels = sorted(list(label_set))
        label_encoders[task] = {label: idx for idx, label in enumerate(sorted_labels)}
        num_classes[task] = len(sorted_labels)
        
        print(f"  {task}: {len(sorted_labels)} classes -> {sorted_labels}")
    
    return label_encoders, num_classes

def encode_labels(labels, label_encoders):
    """Encode labels using label encoders"""
    encoded_labels = []
    
    for label_dict in labels:
        encoded_dict = {}
        for task, label in label_dict.items():
            if task in label_encoders:
                encoded_dict[task] = label_encoders[task][label]
        encoded_labels.append(encoded_dict)
    
    return encoded_labels

def train_epoch(model, dataloader, optimizers, criteria, device, scaler=None, use_amp=False):
    """Train for one epoch with GPU optimizations and mixed precision"""
    model.train()
    total_losses = {task: 0.0 for task in criteria.keys()}
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device with non_blocking for better GPU utilization
        input_ids = batch['input_ids'].to(device)
        batch_labels = batch['labels']
        
        # Clear gradients
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids)
                
                # Calculate losses for each task
                batch_losses = {}
                for task, criterion in criteria.items():
                    task_labels = []
                    for label_dict in batch_labels:
                        if task in label_dict:
                            task_labels.append(label_dict[task])
                    
                    if task_labels:
                        task_labels_tensor = torch.tensor(task_labels, device=device)
                        task_loss = criterion(outputs[task], task_labels_tensor)
                        batch_losses[task] = task_loss
                        total_losses[task] += task_loss.item()
                
                if batch_losses:
                    combined_loss = sum(batch_losses.values()) / len(batch_losses)
                    total_loss += combined_loss.item()
                    num_batches += 1
            
            # Backward pass with mixed precision
            if batch_losses:
                scaler.scale(combined_loss).backward()
                
                # Gradient clipping
                scaler.unscale_(list(optimizers.values())[0])  # Unscale for clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                for optimizer in optimizers.values():
                    scaler.step(optimizer)
                scaler.update()
        else:
            # Regular forward pass
            outputs = model(input_ids)
              # Calculate losses for each task
            batch_losses = {}
            for task, criterion in criteria.items():
                task_labels = []
                for label_dict in batch_labels:
                    if task in label_dict:
                        task_labels.append(label_dict[task])
                
                if task_labels:
                    task_labels_tensor = torch.tensor(task_labels, device=device)
                    task_loss = criterion(outputs[task], task_labels_tensor)
                    batch_losses[task] = task_loss
                    total_losses[task] += task_loss.item()
            
            # Combined loss
            if batch_losses:
                combined_loss = sum(batch_losses.values()) / len(batch_losses)
                total_loss += combined_loss.item()
                num_batches += 1
                
                # Backward pass
                combined_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                for optimizer in optimizers.values():
                    optimizer.step()
        
        # Memory cleanup every 50 batches on GPU
        if device.type == 'cuda' and batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    # Calculate average losses
    if num_batches > 0:
        avg_losses = {task: loss / num_batches for task, loss in total_losses.items()}
        avg_total_loss = total_loss / num_batches
    else:
        avg_losses = {task: 0.0 for task in total_losses.keys()}
        avg_total_loss = 0.0
    
    return avg_losses, avg_total_loss

def evaluate_model(model, dataloader, criteria, device):
    """Evaluate model with GPU optimizations"""
    model.eval()
    total_losses = {task: 0.0 for task in criteria.keys()}
    predictions = {task: [] for task in criteria.keys()}
    true_labels = {task: [] for task in criteria.keys()}
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device with non_blocking
            input_ids = batch['input_ids'].to(device)
            batch_labels = batch['labels']
            
            # Forward pass
            outputs = model(input_ids)
            
            # Calculate losses and collect predictions
            for task, criterion in criteria.items():                # Extract task labels from batch
                task_labels = []
                for label_dict in batch_labels:
                    if task in label_dict:
                        task_labels.append(label_dict[task])
                
                if task_labels:  # Only if we have labels for this task
                    task_labels_tensor = torch.tensor(task_labels, device=device)
                    task_loss = criterion(outputs[task], task_labels_tensor)
                    total_losses[task] += task_loss.item()
                    
                    # Predictions
                    _, predicted = torch.max(outputs[task], 1)
                    predictions[task].extend(predicted.cpu().numpy())
                    true_labels[task].extend(task_labels_tensor.cpu().numpy())
            
            num_batches += 1
            
            # Memory cleanup every 50 batches on GPU
            if device.type == 'cuda' and batch_idx % 50 == 0:
                torch.cuda.empty_cache()
    
    # Calculate metrics
    metrics = {}
    for task in criteria.keys():
        if predictions[task] and num_batches > 0:
            accuracy = accuracy_score(true_labels[task], predictions[task])
            metrics[task] = {
                'loss': total_losses[task] / num_batches,
                'accuracy': accuracy
            }
    
    return metrics

def main():
    """Main training function"""
    print("🚀 HAN GITHUB COMMITS TRAINER")
    print("="*60)
    
    # GPU Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Clear GPU cache
        torch.cuda.empty_cache()
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Paths
    data_file = Path(__file__).parent / "training_data" / "github_commits_training_data.json"
    model_dir = Path(__file__).parent / "models" / "han_github_model"
    log_dir = Path(__file__).parent / "training_logs"
    
    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    if not data_file.exists():
        print(f"❌ Dataset not found: {data_file}")
        print("   Please run: python download_github_commits.py")
        return
    
    texts, labels, metadata = load_github_dataset(data_file)
    
    # Prepare labels
    label_encoders, num_classes = prepare_label_encoders(labels)
    encoded_labels = encode_labels(labels, label_encoders)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42
    )
    
    print(f"📊 Train samples: {len(train_texts)}")
    print(f"📊 Val samples: {len(val_texts)}")
    
    # Build tokenizer
    tokenizer = SimpleTokenizer(vocab_size=5000)
    tokenizer.fit(train_texts)
      # Create datasets
    train_dataset = CommitDataset(train_texts, train_labels, tokenizer)
    val_dataset = CommitDataset(val_texts, val_labels, tokenizer)
    
    # GPU optimized batch size
    if device.type == 'cuda':
        # Larger batch size for GPU
        batch_size = 32
        num_workers = 4  # More workers for GPU
        pin_memory = True
    else:
        # Smaller batch size for CPU
        batch_size = 16
        num_workers = 2
        pin_memory = False
    
    print(f"🔧 Batch size: {batch_size}")
    print(f"👥 Num workers: {num_workers}")
    
    # Create dataloaders with GPU optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
      # Create model with GPU optimizations
    model = SimpleHANModel(
        vocab_size=len(tokenizer.word_to_idx),
        embed_dim=100,
        hidden_dim=128,
        num_classes=num_classes
    ).to(device)
    
    # Enable mixed precision for GPU if available
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        use_amp = True
        print("🚀 Mixed precision training enabled")
    else:
        scaler = None
        use_amp = False
    
    print(f"🤖 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # GPU memory optimization
    if device.type == 'cuda':
        print(f"📊 GPU Memory before training: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # Setup training with optimized learning rates for GPU
    criteria = {}
    optimizers = {}
    schedulers = {}
    
    # Higher learning rate for GPU training
    base_lr = 0.002 if device.type == 'cuda' else 0.001
    
    for task in num_classes.keys():
        criteria[task] = nn.CrossEntropyLoss()
        optimizers[task] = optim.AdamW(  # AdamW is often better than Adam
            list(model.classifiers[task].parameters()) + 
            list(model.embedding.parameters()) +
            list(model.word_lstm.parameters()) +
            list(model.sentence_lstm.parameters()) +
            list(model.word_attention.parameters()) +
            list(model.sentence_attention.parameters()),
            lr=base_lr,
            weight_decay=1e-4  # L2 regularization
        )        # Add learning rate scheduler
        schedulers[task] = optim.lr_scheduler.ReduceLROnPlateau(
            optimizers[task], 
            mode='min', 
            factor=0.5, 
            patience=3
        )
      # Training loop with GPU monitoring
    num_epochs = 20
    best_val_accuracy = 0.0
    
    log_file = log_dir / f"han_github_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    
    # Training start time
    import time
    training_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        # GPU memory monitoring
        if device.type == 'cuda':
            print(f"📊 GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        # Train with enhanced function
        train_losses, train_total_loss = train_epoch(
            model, train_loader, optimizers, criteria, device, scaler, use_amp
        )
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, criteria, device)
        
        # Update learning rate schedulers
        avg_val_loss = 0.0
        if val_metrics:
            for task, metrics in val_metrics.items():
                if task in schedulers:
                    schedulers[task].step(metrics['loss'])
                avg_val_loss += metrics['loss']
            avg_val_loss /= len(val_metrics)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log results
        print(f"  ⏱️  Epoch time: {epoch_time:.1f}s")
        print(f"  📈 Train Loss: {train_total_loss:.4f}")
        for task, loss in train_losses.items():
            print(f"    {task}: {loss:.4f}")
        
        print(f"  📊 Val Metrics:")
        val_accuracy_sum = 0.0
        for task, metrics in val_metrics.items():
            print(f"    {task}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}")
            val_accuracy_sum += metrics['accuracy']
        
        avg_val_accuracy = val_accuracy_sum / len(val_metrics) if val_metrics else 0.0
        print(f"  🎯 Avg Val Accuracy: {avg_val_accuracy:.4f}")
        
        # GPU memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"  🧹 GPU Memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
          # Save log with enhanced information
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\nEpoch {epoch+1}/{num_epochs}\n")
            f.write(f"Epoch Time: {epoch_time:.1f}s\n")
            f.write(f"Train Loss: {train_total_loss:.4f}\n")
            for task, loss in train_losses.items():
                f.write(f"  {task} Train Loss: {loss:.4f}\n")
            f.write(f"Val Metrics: {val_metrics}\n")
            f.write(f"Avg Val Accuracy: {avg_val_accuracy:.4f}\n")
            if device.type == 'cuda':
                f.write(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB\n")
            f.write("-" * 50 + "\n")
        
        # Save best model with enhanced information
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            
            # Save model with comprehensive information
            model_save_dict = {
                'model_state_dict': model.state_dict(),
                'tokenizer': tokenizer,
                'label_encoders': label_encoders,
                'num_classes': num_classes,
                'metadata': metadata,
                'epoch': epoch + 1,
                'val_accuracy': avg_val_accuracy,
                'train_loss': train_total_loss,
                'device': str(device),
                'batch_size': batch_size,
                'learning_rate': base_lr,
                'model_params': sum(p.numel() for p in model.parameters()),
                'training_config': {
                    'use_amp': use_amp,
                    'vocab_size': len(tokenizer.word_to_idx),
                    'embed_dim': 100,
                    'hidden_dim': 128,                    'max_sentences': 10,
                    'max_words': 50
                }
            }
            
            torch.save(model_save_dict, model_dir / 'best_model.pth')
            
            print(f"  💾 Saved best model (accuracy: {avg_val_accuracy:.4f})")
    
    # Training completion summary
    total_training_time = time.time() - training_start_time
    print(f"\n🎉 Training completed!")
    print(f"⏱️  Total training time: {total_training_time/60:.1f} minutes")
    print(f"📊 Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"💾 Model saved to: {model_dir}")
    print(f"📝 Logs saved to: {log_file}")
    
    if device.type == 'cuda':
        print(f"🎮 GPU training completed successfully")
        print(f"📊 Final GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
