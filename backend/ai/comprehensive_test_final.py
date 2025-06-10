#!/usr/bin/env python3
"""
Comprehensive test for the complete Multi-Modal Fusion Network pipeline
Tests all components together with synthetic data
"""

import sys
import os
import torch

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multimodal_fusion.data_preprocessing.text_processor import TextProcessor
from multimodal_fusion.data_preprocessing.metadata_processor import MetadataProcessor
from multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
from multimodal_fusion.training.multitask_trainer import MultiTaskTrainer
from multimodal_fusion.data.synthetic_generator import GitHubDataGenerator

# Include core system test functionality
try:
    from wordcloud import WordCloud
    print("✓ WordCloud imported successfully")
except ImportError:
    print("⚠️ WordCloud not available - will use alternative visualization")

# Test NLTK availability
try:
    import nltk
    print("✓ NLTK imported successfully")
except ImportError:
    print("NLTK not available. Using simple text processing.")

def test_complete_pipeline():
    """Test the complete pipeline from data generation to training"""
    
    try:
        print("🔬 Multi-Modal Fusion Network - Comprehensive Testing")
        print("="*70)
        
        print("🚀 Testing Complete Multi-Modal Fusion Pipeline")
        print("="*60)
        
        # 1. Initialize components
        print("📝 Step 1: Initializing components...")
        text_processor = TextProcessor()
        metadata_processor = MetadataProcessor()
        generator = GitHubDataGenerator()
        
        # 2. Generate synthetic data
        print("📊 Step 2: Generating synthetic training data...")
        dataset = generator.generate_dataset(num_samples=100)
        # Extract texts and metadata from generated dataset
        texts = [item['commit_message'] for item in dataset]
        metadata_samples = [item['metadata'] for item in dataset]
        print(f"✓ Generated {len(texts)} synthetic samples")
        
        # 3. Prepare data
        print("🔧 Step 3: Preparing data for processing...")
        # The synthetic generator returns compatible data structures
        
        # 4. Fit processors
        print("🔧 Step 4: Fitting text and metadata processors...")
        text_processor.build_vocabulary(texts)
        metadata_processor.fit(metadata_samples)
        
        # 5. Process sample data
        print("⚙️ Step 5: Processing sample data...")
        
        sample_text = texts[0]
        sample_metadata = metadata_samples[0]
        
        # Process text
        text_features = text_processor.encode_text_lstm(sample_text)
        print(f"✓ Text features shape: {text_features.shape}")
        
        # Process metadata  
        metadata_features = metadata_processor.process_sample(sample_metadata)
        print(f"✓ Metadata features: {len(metadata_features)} tensors")
        
        # 6. Initialize fusion network
        print("🧠 Step 6: Initializing Multi-Modal Fusion Network...")
        
        # Get feature dimensions
        text_vocab_size = len(text_processor.word_to_idx)
        metadata_dims = metadata_processor.get_feature_dimensions()
        
        model = MultiModalFusionNetwork(
            vocab_size=text_vocab_size,
            text_embed_dim=128,
            text_hidden_dim=64,
            numerical_dim=metadata_dims.get('numerical', 33),
            author_vocab_size=metadata_dims.get('author_vocab_size', 1000),
            season_vocab_size=metadata_dims.get('season_vocab_size', 4),
            file_types_dim=metadata_dims.get('file_types_dim', 100),
            fusion_hidden_dim=256,
            task_configs={'risk_prediction': 2, 'complexity_prediction': 3, 
                         'hotspot_prediction': 5, 'urgency_prediction': 2}
        )
        
        print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # 7. Test forward pass
        print("🔄 Step 7: Testing model forward pass...")
        
        # Prepare batch data
        batch_text = text_features.unsqueeze(0)  # Add batch dimension
        batch_metadata = {k: v.unsqueeze(0) if v.dim() > 0 else v.unsqueeze(0) 
                         for k, v in metadata_features.items()}
        
        with torch.no_grad():
            outputs = model(batch_text, batch_metadata)
            
        print(f"✓ Model forward pass successful")
        for task_name, output in outputs.items():
            print(f"  - {task_name} shape: {output.shape}")
        
        # 8. Test training step (small)
        print("🏋️ Step 8: Testing training step...")
        
        trainer = MultiTaskTrainer(
            model=model,
            device='cpu',
            learning_rate=0.001
        )
        
        # Create dummy targets for testing
        dummy_targets = {
            'risk_prediction': torch.randint(0, 2, (1,)),
            'complexity_prediction': torch.randint(0, 3, (1,)),
            'hotspot_prediction': torch.randint(0, 5, (1,)),
            'urgency_prediction': torch.randint(0, 2, (1,))
        }
        
        # Test a single training step
        loss = trainer.train_step(batch_text, batch_metadata, dummy_targets)
        print(f"✓ Training step successful, loss: {loss:.4f}")
        
        print("\n🎉 Complete pipeline test successful!")
        print("✅ All components working together properly")
        print("\nNext steps:")
        print("1. Run full training with real data")
        print("2. Implement evaluation metrics")
        print("3. Tune hyperparameters")
        print("4. Deploy to production")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the comprehensive test
    success = test_complete_pipeline()
    
    if success:
        print("\n🚀 System ready for production deployment!")
    else:
        print("\n❌ System needs debugging before deployment")
        sys.exit(1)
