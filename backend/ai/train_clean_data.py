#!/usr/bin/env python3
"""
Multi-Modal Fusion Network - Clean Data Training Script
======================================================
Train the model using cleaned GitHub commit data.
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

def load_cleaned_data(file_path: str, max_samples: int = 10000) -> List[Dict]:
    """Load cleaned GitHub commit data."""
    print(f"📁 Loading cleaned data from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Kiểm tra format dữ liệu
    if 'samples' in data:
        samples = data['samples'][:max_samples]
        print(f"✅ Loaded {len(samples)} samples from cleaned dataset")
    elif 'data' in data:
        # Convert format từ sample_preview.json
        raw_samples = data['data'][:max_samples]
        samples = []
        
        for sample in raw_samples:
            # Convert format
            converted_sample = {
                'commit_message': sample.get('text', ''),
                'author': f"user_{hash(sample.get('text', '')) % 1000}",
                'repository': f"repo_{hash(sample.get('text', '')) % 100}",
                'timestamp': '2025-01-01',
                'files_changed': np.random.randint(1, 10),
                'additions': np.random.randint(1, 100),
                'deletions': np.random.randint(0, 50),
                'file_types': ['py', 'js', 'java'][np.random.randint(0, 3):np.random.randint(1, 4)],
                'labels': convert_labels_to_multimodal_format(sample.get('labels', {}))
            }
            samples.append(converted_sample)
        
        print(f"✅ Converted {len(samples)} samples from sample_preview format")
    else:
        print("❌ Không nhận diện được format dữ liệu")
        return []
    
    return samples

def convert_labels_to_multimodal_format(labels: Dict) -> Dict:
    """Convert các labels sang format multi-modal"""
    
    # Map từ original labels sang multi-modal format
    commit_type = labels.get('commit_type', 'other')
    purpose = labels.get('purpose', 'Other')
    sentiment = labels.get('sentiment', 'neutral')
    tech_tag = labels.get('tech_tag', 'general')
    
    # Risk prediction (0: low, 1: high)
    risk_prediction = 0
    if commit_type in ['feat', 'refactor', 'security'] or sentiment == 'urgent':
        risk_prediction = 1
    
    # Complexity prediction (0: low, 1: medium, 2: high)
    complexity_prediction = 0
    if purpose in ['Feature Implementation', 'Refactoring']:
        complexity_prediction = 2
    elif purpose in ['Bug Fix', 'Test Update']:
        complexity_prediction = 1
    
    # Hotspot prediction (0-4: security, api, database, ui, general)
    hotspot_map = {
        'security': 0,
        'api': 1, 
        'database': 2,
        'ui': 3,
        'general': 4
    }
    hotspot_prediction = hotspot_map.get(tech_tag, 4)
    
    # Urgency prediction (0: normal, 1: urgent)
    urgency_prediction = 1 if sentiment == 'urgent' else 0
    
    return {
        'risk_prediction': risk_prediction,
        'complexity_prediction': complexity_prediction,
        'hotspot_prediction': hotspot_prediction,
        'urgency_prediction': urgency_prediction
    }

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
    print("🚀 Multi-Modal Fusion Network - Clean Data Training")
    print("=" * 70)
    
    # Configuration
    max_samples = 3000  # Giảm xuống để test nhanh
    batch_size = 16     # Giảm batch size
    epochs = 3          # Giảm epochs để test
    
    # Tìm file dữ liệu
    data_files = [
        "training_data/sample_preview.json",
        "training_data/cleaned_github_commits_*.json"
    ]
    
    data_file = None
    for pattern in data_files:
        if '*' in pattern:
            # Tìm file matching pattern
            files = list(Path().glob(pattern))
            if files:
                data_file = str(files[0])
                break
        else:
            if Path(pattern).exists():
                data_file = pattern
                break
    
    if not data_file:
        print("❌ Không tìm thấy file dữ liệu nào!")
        print("💡 Chạy script clean_github_data.py trước để tạo dữ liệu")
        return
    
    print(f"📁 Sử dụng file: {data_file}")
    
    # Load and prepare data
    try:
        samples = load_cleaned_data(data_file, max_samples)
        if len(samples) < 50:
            print("❌ Không đủ dữ liệu training. Cần ít nhất 50 samples.")
            return
            
        texts, metadata_list, labels_list = prepare_training_data(samples)
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
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
    text_processor = TextProcessor(max_length=128)
    metadata_processor = MetadataProcessor()
    
    # Fit processors on training data
    print("🔧 Fitting text processor...")
    text_processor.fit(train_texts)
    
    print("🔧 Fitting metadata processor...")
    metadata_processor.fit(train_metadata)
      # Get dimensions for model
    sample_text_batch = text_processor.process_batch([train_texts[0]])
    sample_text_features = sample_text_batch['embeddings'][0]
    sample_metadata_features = metadata_processor.process_batch([train_metadata[0]])
    
    metadata_dims = {
        'numerical_dim': sample_metadata_features['numerical_features'].shape[1],
        'author_vocab_size': len(metadata_processor.author_encoder.classes_),
        'season_vocab_size': 4,  # Fixed: spring, summer, fall, winter
        'file_types_dim': sample_metadata_features['file_types_encoded'].shape[1]
    }
    
    print(f"📐 Text features dimension: {sample_text_features.shape}")
    print(f"📐 Metadata dimensions: {metadata_dims}")
    
    # Initialize model
    print("\n🧠 Initializing Multi-Modal Fusion Network...")
    model = MultiModalFusionNetwork(
        text_dim=text_processor.embed_dim,
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
            text_batch_result = text_processor.process_batch(batch_texts)
            metadata_batch_result = metadata_processor.process_batch(batch_metadata)
            
            targets = {task: [] for task in task_configs.keys()}
            
            for labels in batch_labels:
                # Labels
                for task, label in labels.items():
                    targets[task].append(label)            # Convert to tensors
            text_input = text_batch_result['embeddings'].to(trainer.device)
            metadata_input = {
                'numerical_features': metadata_batch_result['numerical_features'].to(trainer.device),
                'author_encoded': metadata_batch_result['author_encoded'].to(trainer.device), 
                'season_encoded': metadata_batch_result['season_encoded'].to(trainer.device),
                'file_types_encoded': metadata_batch_result['file_types_encoded'].to(trainer.device)
            }
            
            # Convert targets to tensors
            target_tensors = {}
            for task, task_labels in targets.items():
                target_tensors[task] = torch.tensor(task_labels, dtype=torch.long).to(trainer.device)
            
            # Training step
            try:
                loss = trainer.train_step(text_input, metadata_input, target_tensors)
                train_losses.append(loss)
                
                if (i // batch_size) % 5 == 0:
                    print(f"  Batch {i//batch_size + 1}: Loss = {loss:.4f}")
                    
            except Exception as e:
                print(f"❌ Error in training step: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_train_loss = np.mean(train_losses) if train_losses else 0
        print(f"📊 Average training loss: {avg_train_loss:.4f}")
          # Simple validation
        if val_texts and len(val_texts) > 0:
            print("🔍 Running validation...")
            model.eval()
            with torch.no_grad():
                try:
                    # Process một sample validation
                    val_text_batch = text_processor.process_batch([val_texts[0]])
                    val_meta_batch = metadata_processor.process_batch([val_metadata[0]])                    # Convert to model input
                    val_text_input = val_text_batch['embeddings'].to(trainer.device)
                    val_metadata_input = {
                        'numerical_features': val_meta_batch['numerical_features'].to(trainer.device),
                        'author_encoded': val_meta_batch['author_encoded'].to(trainer.device),
                        'season_encoded': val_meta_batch['season_encoded'].to(trainer.device), 
                        'file_types_encoded': val_meta_batch['file_types_encoded'].to(trainer.device)
                    }
                    
                    # Forward pass
                    outputs = model(val_text_input, val_metadata_input)
                    print(f"✅ Validation forward pass successful")
                    
                except Exception as e:
                    print(f"⚠️ Validation error: {e}")
    
    # Save model
    print("\n💾 Saving trained model...")
    model_save_path = "trained_models/multimodal_fusion_clean_data.pth"
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
    print(f"📊 Final training loss: {avg_train_loss:.4f}")
    print(f"📁 Model saved: {model_save_path}")
    
    print(f"\n✨ Next steps:")
    print(f"1. Evaluate model với detailed metrics")
    print(f"2. Fine-tune hyperparameters")
    print(f"3. Train với more data")
    print(f"4. Deploy to production")

if __name__ == "__main__":
    main()
