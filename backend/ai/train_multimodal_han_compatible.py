#!/usr/bin/env python3
"""
Multi-Modal Fusion Network - HAN Compatible Training Script
==========================================================
Train the model using HAN-compatible labels for direct comparison.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random

# Import our components
from multimodal_fusion.data_preprocessing.text_processor import TextProcessor
from multimodal_fusion.data_preprocessing.metadata_processor import MetadataProcessor
from multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
from multimodal_fusion.training.multitask_trainer import MultiTaskTrainer

# HAN Model Tasks and Labels - EXACT MAPPING
HAN_TASKS = {
    'commit_type': {
        'classes': ['feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore', 'perf', 'build', 'ci'],
        'num_classes': 10
    },
    'purpose': {
        'classes': ['Feature Implementation', 'Bug Fix', 'Documentation Update', 'Code Style', 
                   'Refactoring', 'Test Update', 'Maintenance', 'Performance', 'Build'],
        'num_classes': 9
    },
    'sentiment': {
        'classes': ['positive', 'negative', 'neutral', 'urgent'],
        'num_classes': 4
    },
    'tech_tag': {
        'classes': ['general', 'frontend', 'backend', 'database', 'api', 'security', 'ui', 
                   'config', 'deployment', 'testing', 'documentation', 'performance', 'mobile', 'ml'],
        'num_classes': 14
    }
}

def load_cleaned_data(file_path: str, max_samples: int = 3000) -> List[Dict]:
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
            # Convert format với HAN-compatible labels
            converted_sample = {
                'commit_message': sample.get('text', ''),
                'author': f"user_{hash(sample.get('text', '')) % 1000}",
                'repository': f"repo_{hash(sample.get('text', '')) % 100}",
                'timestamp': '2025-01-01',
                'files_changed': np.random.randint(1, 10),
                'additions': np.random.randint(1, 100),
                'deletions': np.random.randint(0, 50),
                'file_types': ['py', 'js', 'java'][np.random.randint(0, 3):np.random.randint(1, 4)],
                'labels': convert_labels_to_han_format(sample.get('labels', {}))
            }
            samples.append(converted_sample)
        
        print(f"✅ Converted {len(samples)} samples with HAN-compatible labels")
    else:
        print("❌ Không nhận diện được format dữ liệu")
        return []
    
    return samples

def convert_labels_to_han_format(original_labels: Dict) -> Dict:
    """Convert labels to exact HAN format for compatibility."""
    
    # Nếu đã có labels từ original HAN
    if 'commit_type' in original_labels:
        return {
            'commit_type': original_labels.get('commit_type', 'feat'),
            'purpose': original_labels.get('purpose', 'Feature Implementation'),
            'sentiment': original_labels.get('sentiment', 'neutral'),
            'tech_tag': original_labels.get('tech_tag', 'general')
        }
    
    # Nếu cần generate từ text hoặc synthetic
    commit_message = original_labels.get('text', '')
    
    # Smart mapping based on commit message
    han_labels = generate_han_labels_from_text(commit_message)
    
    return han_labels

def generate_han_labels_from_text(text: str) -> Dict:
    """Generate HAN-compatible labels based on commit message text."""
    text_lower = text.lower()
    
    # commit_type detection
    commit_type = 'feat'  # default
    if any(word in text_lower for word in ['fix', 'bug', 'error', 'issue']):
        commit_type = 'fix'
    elif any(word in text_lower for word in ['doc', 'readme', 'comment']):
        commit_type = 'docs'
    elif any(word in text_lower for word in ['refactor', 'restructure', 'clean']):
        commit_type = 'refactor'
    elif any(word in text_lower for word in ['test', 'spec', 'unit test']):
        commit_type = 'test'
    elif any(word in text_lower for word in ['style', 'format', 'lint']):
        commit_type = 'style'
    elif any(word in text_lower for word in ['performance', 'optimize', 'speed']):
        commit_type = 'perf'
    elif any(word in text_lower for word in ['build', 'compile', 'webpack']):
        commit_type = 'build'
    elif any(word in text_lower for word in ['ci', 'pipeline', 'workflow']):
        commit_type = 'ci'
    elif any(word in text_lower for word in ['chore', 'maintenance', 'update']):
        commit_type = 'chore'
    
    # purpose detection
    purpose = 'Feature Implementation'  # default
    if commit_type == 'fix':
        purpose = 'Bug Fix'
    elif commit_type == 'docs':
        purpose = 'Documentation Update'
    elif commit_type == 'style':
        purpose = 'Code Style'
    elif commit_type == 'refactor':
        purpose = 'Refactoring'
    elif commit_type == 'test':
        purpose = 'Test Update'
    elif commit_type == 'perf':
        purpose = 'Performance'
    elif commit_type == 'build':
        purpose = 'Build'
    elif commit_type in ['chore', 'ci']:
        purpose = 'Maintenance'
    
    # sentiment detection
    sentiment = 'neutral'  # default
    if any(word in text_lower for word in ['urgent', 'critical', 'emergency', 'hotfix']):
        sentiment = 'urgent'
    elif any(word in text_lower for word in ['improve', 'enhance', 'add', 'new', 'feature']):
        sentiment = 'positive'
    elif any(word in text_lower for word in ['remove', 'delete', 'deprecated', 'broken']):
        sentiment = 'negative'
    
    # tech_tag detection
    tech_tag = 'general'  # default
    if any(word in text_lower for word in ['react', 'vue', 'angular', 'frontend', 'ui', 'css', 'html']):
        tech_tag = 'frontend'
    elif any(word in text_lower for word in ['server', 'backend', 'api', 'service']):
        tech_tag = 'backend'
    elif any(word in text_lower for word in ['database', 'sql', 'mongo', 'redis', 'db']):
        tech_tag = 'database'
    elif any(word in text_lower for word in ['api', 'endpoint', 'rest', 'graphql']):
        tech_tag = 'api'
    elif any(word in text_lower for word in ['security', 'auth', 'permission', 'vulnerability']):
        tech_tag = 'security'
    elif any(word in text_lower for word in ['ui', 'interface', 'component', 'design']):
        tech_tag = 'ui'
    elif any(word in text_lower for word in ['config', 'configuration', 'setting']):
        tech_tag = 'config'
    elif any(word in text_lower for word in ['deploy', 'docker', 'kubernetes', 'prod']):
        tech_tag = 'deployment'
    elif any(word in text_lower for word in ['test', 'testing', 'unit', 'integration']):
        tech_tag = 'testing'
    elif any(word in text_lower for word in ['doc', 'documentation', 'readme']):
        tech_tag = 'documentation'
    elif any(word in text_lower for word in ['performance', 'optimize', 'cache', 'speed']):
        tech_tag = 'performance'
    elif any(word in text_lower for word in ['mobile', 'android', 'ios', 'react-native']):
        tech_tag = 'mobile'
    elif any(word in text_lower for word in ['ml', 'ai', 'model', 'neural', 'learning']):
        tech_tag = 'ml'
    
    return {
        'commit_type': commit_type,
        'purpose': purpose,
        'sentiment': sentiment,
        'tech_tag': tech_tag
    }

def prepare_training_data(samples: List[Dict]) -> Tuple[List[str], List[Dict], List[Dict]]:
    """Prepare text, metadata, and labels for training."""
    print("🔧 Preparing training data with HAN-compatible format...")
    
    texts = []
    metadata_list = []
    labels_list = []
    
    for sample in samples:
        texts.append(sample['commit_message'])
        
        # Metadata processing
        metadata = {
            'author': sample.get('author', 'unknown'),
            'repository': sample.get('repository', 'unknown'),
            'timestamp': sample.get('timestamp', '2025-01-01'),
            'files_changed': sample.get('files_changed', 1),
            'additions': sample.get('additions', 10),
            'deletions': sample.get('deletions', 5),
            'file_types': sample.get('file_types', ['py'])
        }
        metadata_list.append(metadata)
        
        # HAN-compatible labels
        labels_list.append(sample['labels'])
    
    print(f"✅ Prepared {len(texts)} samples")
    return texts, metadata_list, labels_list

def create_han_compatible_model():
    """Create MultiModalFusionNetwork with HAN-compatible outputs."""
    print("🏗️ Creating HAN-compatible Multi-Modal Fusion Network...")
    
    # Configuration với HAN tasks
    config = {
        # Text processing
        'text_encoder': {
            'vocab_size': 5000,
            'embedding_dim': 128,
            'hidden_dim': 64,
            'num_layers': 2
        },
        
        # Metadata processing  
        'metadata_encoder': {
            'categorical_dims': {
                'author': 1000,
                'repository': 100,
                'file_types': 20
            },
            'numerical_features': ['files_changed', 'additions', 'deletions'],
            'embedding_dim': 32,
            'hidden_dim': 64
        },
        
        # Fusion layer
        'fusion': {
            'fusion_dim': 128,
            'dropout': 0.3
        },
        
        # HAN-compatible task heads
        'task_heads': HAN_TASKS
    }
    
    # Create model
    model = MultiModalFusionNetwork(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model created:")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")
    print(f"   • Tasks: {list(HAN_TASKS.keys())}")
    
    return model

def train_model(model, texts, metadata_list, labels_list):
    """Train the HAN-compatible model."""
    print("🚀 Starting HAN-compatible training...")
    
    # Text processor
    text_processor = TextProcessor(vocab_size=5000, max_length=100)
    text_processor.fit(texts)
    
    # Metadata processor
    metadata_processor = MetadataProcessor()
    metadata_processor.fit(metadata_list)
    
    # Prepare datasets
    processed_texts = [text_processor.process(text) for text in texts]
    processed_metadata = [metadata_processor.process(metadata) for metadata in metadata_list]
    
    # Convert labels to indices
    label_encoders = {}
    encoded_labels = []
    
    for task, task_info in HAN_TASKS.items():
        label_encoders[task] = {label: idx for idx, label in enumerate(task_info['classes'])}
    
    for labels in labels_list:
        encoded_label = {}
        for task, label in labels.items():
            if task in label_encoders and label in label_encoders[task]:
                encoded_label[task] = label_encoders[task][label]
            else:
                # Default to first class if label not found
                encoded_label[task] = 0
        encoded_labels.append(encoded_label)
    
    # Training configuration
    training_config = {
        'batch_size': 16,
        'num_epochs': 3,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"🔧 Training configuration:")
    print(f"   • Batch size: {training_config['batch_size']}")
    print(f"   • Epochs: {training_config['num_epochs']}")
    print(f"   • Learning rate: {training_config['learning_rate']}")
    print(f"   • Device: {training_config['device']}")
    
    # Create trainer
    trainer = MultiTaskTrainer(model, training_config)
    
    # Train model
    trainer.train(processed_texts, processed_metadata, encoded_labels)
    
    print("✅ Training completed!")
    
    return model, text_processor, metadata_processor, label_encoders

def main():
    """Main training function."""
    print("=" * 60)
    print("🤖 Multi-Modal Fusion Network - HAN Compatible Training")
    print("=" * 60)
    
    # Check if we have cleaned data or use sample data
    data_files = [
        "d:/Project/KLTN04/backend/ai/training_data/cleaned_github_commits.json",
        "d:/Project/KLTN04/backend/ai/training_data/sample_preview.json"
    ]
    
    data_file = None
    for file_path in data_files:
        if Path(file_path).exists():
            data_file = file_path
            break
    
    if not data_file:
        print("❌ No training data found!")
        print("Available options:")
        print("1. Run clean_github_data.py to download and clean GitHub data")
        print("2. Use sample_preview.json if available")
        return
    
    print(f"📊 Using data file: {data_file}")
    
    try:
        # Load data
        samples = load_cleaned_data(data_file, max_samples=3000)
        
        if not samples:
            print("❌ No samples loaded!")
            return
        
        # Prepare training data
        texts, metadata_list, labels_list = prepare_training_data(samples)
        
        # Create model
        model = create_han_compatible_model()
        
        # Train model
        trained_model, text_processor, metadata_processor, label_encoders = train_model(
            model, texts, metadata_list, labels_list
        )
        
        # Save model and processors
        save_dir = Path("d:/Project/KLTN04/backend/ai/models/multimodal_han_compatible")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(trained_model.state_dict(), save_dir / "model.pth")
        
        # Save processors and config
        import pickle
        with open(save_dir / "text_processor.pkl", 'wb') as f:
            pickle.dump(text_processor, f)
        
        with open(save_dir / "metadata_processor.pkl", 'wb') as f:
            pickle.dump(metadata_processor, f)
        
        with open(save_dir / "label_encoders.json", 'w') as f:
            json.dump(label_encoders, f, indent=2)
        
        # Save HAN tasks config
        with open(save_dir / "han_tasks.json", 'w') as f:
            json.dump(HAN_TASKS, f, indent=2)
        
        print(f"💾 Model and processors saved to {save_dir}")
        
        # Validation test
        print("\n🧪 Running validation test...")
        test_sample(trained_model, text_processor, metadata_processor, texts[0], metadata_list[0])
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

def test_sample(model, text_processor, metadata_processor, text, metadata):
    """Test model with a sample."""
    model.eval()
    
    with torch.no_grad():
        # Process inputs
        processed_text = text_processor.process(text)
        processed_metadata = metadata_processor.process(metadata)
        
        # Convert to tensors
        text_tensor = torch.tensor([processed_text], dtype=torch.long)
        
        # Convert metadata to tensors
        metadata_tensors = {}
        for key, value in processed_metadata.items():
            if isinstance(value, (int, float)):
                metadata_tensors[key] = torch.tensor([[value]], dtype=torch.float32)
            else:
                metadata_tensors[key] = torch.tensor([value], dtype=torch.long)
        
        # Forward pass
        outputs = model(text_tensor, metadata_tensors)
        
        print(f"📝 Test commit: '{text[:50]}...'")
        print("🎯 Predictions:")
        
        for task, logits in outputs.items():
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
            class_name = HAN_TASKS[task]['classes'][predicted_class]
            
            print(f"   • {task}: {class_name} (confidence: {confidence:.3f})")

if __name__ == "__main__":
    main()
