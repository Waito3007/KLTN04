#!/usr/bin/env python3
"""
Multi-Modal Fusion Network - Real Data Training Script
======================================================
Train the model using real GitHub commit data from Kaggle dataset.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Import our components
from multimodal_fusion.data_preprocessing.text_processor import TextProcessor
from multimodal_fusion.data_preprocessing.metadata_processor import MetadataProcessor
from multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
from multimodal_fusion.training.multitask_trainer import MultiTaskTrainer

# Import data generation for label creation
from multimodal_fusion.data.synthetic_generator import GitHubDataGenerator

def load_kaggle_data(file_path: str, max_samples: int = 10000) -> List[Dict]:
    """Load and parse the processed GitHub dataset."""
    print(f"📁 Loading processed data from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if this is the sample_preview format
    if 'data' in data:
        samples = data['data'][:max_samples]
        print(f"✅ Loaded {len(samples)} samples from processed dataset")
        return samples
    elif 'samples' in data:
        samples = data['samples'][:max_samples]
        print(f"✅ Loaded {len(samples)} samples from processed dataset")
        return samples
    else:
        # Handle other formats
        samples = data[:max_samples]
        print(f"✅ Loaded {len(samples)} samples from dataset")
        return samples

def convert_to_training_format(processed_samples: List[Dict]) -> List[Dict]:
    """Convert processed format to our training format."""
    print("🔄 Converting data format...")
    
    training_samples = []
    
    for i, sample in enumerate(processed_samples):
        if i % 1000 == 0:
            print(f"  Processed {i}/{len(processed_samples)} samples")
        
        # Check if this is already in the right format
        if 'commit_message' in sample and 'labels' in sample:
            training_samples.append(sample)
            continue
            
        # Extract commit message
        commit_message = sample.get('message', sample.get('commit_message', '')).strip()
        if not commit_message or len(commit_message) < 5:
            continue
        
        # Get existing labels or create them
        labels = sample.get('labels', {})
        if not labels:
            # Create labels from classification data if available
            labels = {
                'risk_prediction': 0 if sample.get('sentiment', 'neutral') in ['positive', 'neutral'] else 1,
                'complexity_prediction': {'simple': 0, 'medium': 1, 'complex': 2}.get(
                    sample.get('complexity', 'simple'), 1),
                'hotspot_prediction': {'low': 0, 'medium': 1, 'high': 2, 'critical': 3, 'emergency': 4}.get(
                    sample.get('priority', 'medium'), 1),
                'urgency_prediction': 0 if sample.get('sentiment', 'neutral') != 'urgent' else 1
            }
        
        # Get metadata
        metadata = sample.get('metadata', {})
        if not metadata:            # Create synthetic metadata
            from multimodal_fusion.data.synthetic_generator import GitHubDataGenerator
            generator = GitHubDataGenerator()
            synthetic_sample = generator.generate_single_commit()
            metadata = {
                'author': synthetic_sample['author'],
                'repository': synthetic_sample['repository'],
                'timestamp': synthetic_sample['timestamp'],
                'files_changed': synthetic_sample['files_changed'],
                'additions': synthetic_sample['additions'],
                'deletions': synthetic_sample['deletions'],
                'file_types': synthetic_sample['file_types']
            }
        
        # Create training sample
        training_sample = {
            'commit_message': commit_message,
            'author': metadata.get('author', 'unknown'),
            'repository': metadata.get('repository', 'unknown'),
            'timestamp': metadata.get('timestamp', '2025-01-01'),
            'files_changed': metadata.get('files_changed', 1),
            'additions': metadata.get('additions', 1),
            'deletions': metadata.get('deletions', 0),
            'file_types': metadata.get('file_types', ['py']),
            'labels': labels
        }
        
        training_samples.append(training_sample)
    
    print(f"✅ Converted {len(training_samples)} samples to training format")
    return training_samples

def prepare_training_data(samples: List[Dict]) -> Tuple[List[str], List[Dict], List[Dict]]:
    """Prepare text, metadata, and labels for training."""
    print("🔧 Preparing training data...")
    
    texts = []
    metadata_list = []
    labels_list = []
    
    for sample in samples:
        texts.append(sample['commit_message'])
        
        # Metadata
        metadata = {
            'author': sample['author'],
            'repository': sample['repository'],
            'timestamp': sample['timestamp'],
            'files_changed': sample['files_changed'],
            'additions': sample['additions'],
            'deletions': sample['deletions'],
            'file_types': sample['file_types']
        }
        metadata_list.append(metadata)
        
        # Labels
        labels_list.append(sample['labels'])
    
    print(f"✅ Prepared {len(texts)} training samples")
    return texts, metadata_list, labels_list

def main():
    """Main training function."""
    print("🚀 Multi-Modal Fusion Network - Real Data Training")
    print("=" * 70)
      # Configuration
    data_file = "training_data/sample_preview.json"
    max_samples = 5000  # Start with smaller dataset for initial training
    batch_size = 32
    epochs = 5
    
    # Check if data file exists
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        print("Please ensure the Kaggle dataset is available.")
        return
    
    # Load and convert data
    try:
        kaggle_samples = load_kaggle_data(data_file, max_samples)
        training_samples = convert_to_training_format(kaggle_samples)
        texts, metadata_list, labels_list = prepare_training_data(training_samples)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    if len(texts) < 100:
        print("❌ Not enough training samples. Need at least 100 samples.")
        return
    
    # Split data into train/validation
    split_idx = int(0.8 * len(texts))
    train_texts = texts[:split_idx]
    train_metadata = metadata_list[:split_idx]
    train_labels = labels_list[:split_idx]
    
    val_texts = texts[split_idx:]
    val_metadata = metadata_list[split_idx:]
    val_labels = labels_list[split_idx:]
    
    print(f"📊 Training samples: {len(train_texts)}")
    print(f"📊 Validation samples: {len(val_texts)}")
    
    # Initialize processors
    print("\n🔧 Initializing data processors...")
    text_processor = TextProcessor(max_length=128, embedding_dim=128)
    metadata_processor = MetadataProcessor()
    
    # Fit processors on training data
    print("🔧 Fitting text processor...")
    text_processor.fit(train_texts)
    
    print("🔧 Fitting metadata processor...")
    metadata_processor.fit(train_metadata)
    
    # Get dimensions for model
    sample_text_features = text_processor.transform([train_texts[0]])[0]
    sample_metadata_features = metadata_processor.transform([train_metadata[0]])
    
    metadata_dims = {
        'numerical_dim': sample_metadata_features[0].shape[0],
        'author_vocab_size': len(metadata_processor.author_encoder.classes_),
        'season_vocab_size': len(metadata_processor.season_encoder.classes_),
        'file_types_dim': sample_metadata_features[3].shape[0]
    }
    
    print(f"📐 Text features dimension: {sample_text_features.shape}")
    print(f"📐 Metadata dimensions: {metadata_dims}")
    
    # Initialize model
    print("\n🧠 Initializing Multi-Modal Fusion Network...")
    model = MultiModalFusionNetwork(
        text_dim=text_processor.embedding_dim,
        **metadata_dims,
        hidden_dim=256,
        dropout_rate=0.3
    )
    
    # Task configurations
    task_configs = {
        'risk_prediction': {'num_classes': 2, 'weight': 1.0},
        'complexity_prediction': {'num_classes': 3, 'weight': 1.0},
        'hotspot_prediction': {'num_classes': 5, 'weight': 1.0},
        'urgency_prediction': {'num_classes': 2, 'weight': 1.0}
    }
      # Initialize trainer
    print("🏋️ Initializing trainer...")
    trainer = MultiTaskTrainer(
        model=model,
        task_configs=task_configs,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"💻 Training device: {trainer.device}")
    
    # Training loop
    print(f"\n🏋️ Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Training
        model.train()
        train_losses = []
        
        # Process in batches
        for i in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[i:i + batch_size]
            batch_metadata = train_metadata[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]
            
            # Process batch
            text_features = []
            metadata_features = []
            targets = {task: [] for task in task_configs.keys()}
            
            for j, (text, metadata, labels) in enumerate(zip(batch_texts, batch_metadata, batch_labels)):
                # Text features
                text_feat = text_processor.transform([text])[0]
                text_features.append(text_feat)
                
                # Metadata features
                meta_feat = metadata_processor.transform([metadata])
                metadata_features.append(meta_feat)
                
                # Labels
                for task, label in labels.items():
                    targets[task].append(label)
            
            # Convert to tensors
            text_input = torch.stack(text_features)
            metadata_input = [
                torch.stack([meta[i] for meta in metadata_features])
                for i in range(4)
            ]
            
            # Convert targets to tensors
            target_tensors = {}
            for task, task_labels in targets.items():
                target_tensors[task] = torch.tensor(task_labels, dtype=torch.long)
            
            # Training step
            try:
                loss = trainer.train_step(text_input, metadata_input, target_tensors)
                train_losses.append(loss)
                
                if (i // batch_size) % 10 == 0:
                    print(f"  Batch {i//batch_size + 1}: Loss = {loss:.4f}")
                    
            except Exception as e:
                print(f"❌ Error in training step: {e}")
                continue
        
        avg_train_loss = np.mean(train_losses) if train_losses else 0
        print(f"📊 Average training loss: {avg_train_loss:.4f}")
        
        # Validation (simple forward pass)
        if val_texts:
            model.eval()
            with torch.no_grad():
                val_losses = []
                
                # Sample validation batch
                val_batch_size = min(32, len(val_texts))
                val_sample_texts = val_texts[:val_batch_size]
                val_sample_metadata = val_metadata[:val_batch_size]
                val_sample_labels = val_labels[:val_batch_size]
                
                # Process validation batch
                val_text_features = []
                val_metadata_features = []
                val_targets = {task: [] for task in task_configs.keys()}
                
                for text, metadata, labels in zip(val_sample_texts, val_sample_metadata, val_sample_labels):
                    text_feat = text_processor.transform([text])[0]
                    val_text_features.append(text_feat)
                    
                    meta_feat = metadata_processor.transform([metadata])
                    val_metadata_features.append(meta_feat)
                    
                    for task, label in labels.items():
                        val_targets[task].append(label)
                
                # Convert to tensors
                val_text_input = torch.stack(val_text_features)
                val_metadata_input = [
                    torch.stack([meta[i] for meta in val_metadata_features])
                    for i in range(4)
                ]
                
                val_target_tensors = {}
                for task, task_labels in val_targets.items():
                    val_target_tensors[task] = torch.tensor(task_labels, dtype=torch.long)
                
                # Forward pass
                try:
                    outputs = model(val_text_input, val_metadata_input)
                    
                    # Calculate loss (simplified)
                    total_val_loss = 0
                    for task, output in outputs.items():
                        criterion = torch.nn.CrossEntropyLoss()
                        task_loss = criterion(output, val_target_tensors[task])
                        total_val_loss += task_loss.item()
                    
                    print(f"📊 Validation loss: {total_val_loss:.4f}")
                    
                except Exception as e:
                    print(f"⚠️ Validation error: {e}")
    
    # Save model
    print("\n💾 Saving trained model...")
    model_save_path = "trained_models/multimodal_fusion_real_data.pth"
    Path("trained_models").mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'text_processor': text_processor,
        'metadata_processor': metadata_processor,
        'task_configs': task_configs,
        'metadata_dims': metadata_dims
    }, model_save_path)
    
    print(f"✅ Model saved to {model_save_path}")
    
    print("\n🎉 Training completed successfully!")
    print("\nNext steps:")
    print("1. Evaluate model performance with detailed metrics")
    print("2. Fine-tune hyperparameters")
    print("3. Deploy to production environment")
    print("4. Monitor performance on real-world data")

if __name__ == "__main__":
    main()
