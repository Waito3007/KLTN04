"""
Windows-Compatible Test Scripts
Simplified test scripts that avoid Unicode characters for Windows compatibility
"""

import os
import sys
import json
import logging
from pathlib import Path

# Setup basic logging without Unicode
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test all required imports without Unicode output"""
    try:
        import torch
        logger.info("PyTorch imported successfully")
        
        import transformers
        logger.info("Transformers imported successfully")
        
        import nltk
        logger.info("NLTK imported successfully")
        
        import sklearn
        logger.info("Scikit-learn imported successfully")
        
        import pandas
        logger.info("Pandas imported successfully")
        
        import numpy
        logger.info("NumPy imported successfully")
        
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA available: {cuda_available}")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import failed: {e}")
        return False

def test_text_processor():
    """Test enhanced text processor without Unicode output"""
    try:
        from multimodal_fusion.data_preprocessing.minimal_enhanced_text_processor import MinimalEnhancedTextProcessor
        
        logger.info("Testing Enhanced Text Processor...")
        
        # Initialize processor
        processor = MinimalEnhancedTextProcessor(
            method="lstm",
            vocab_size=1000,
            max_length=128,
            enable_sentiment=True,
            enable_advanced_cleaning=True
        )
        
        # Test fitting
        test_texts = ["fix: update authentication", "feat: add dashboard", "docs: update README"]
        processor.fit(test_texts)
        logger.info("Text processor fitting - SUCCESS")
        
        # Test encoding
        encoded = processor.encode_text_lstm(test_texts[0])
        logger.info(f"Text encoding shape: {encoded.shape}")
        
        # Test enhanced features
        features = processor.extract_enhanced_features(test_texts[0])
        logger.info(f"Enhanced features count: {len(features)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Text processor test failed: {e}")
        return False

def test_model_structure():
    """Test model structure without Unicode output"""
    try:
        from multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
        
        logger.info("Testing Model Structure...")
        
        # Test model config
        config = {
            'text_encoder': {
                'vocab_size': 1000,
                'embedding_dim': 64,
                'hidden_dim': 32,
                'num_layers': 1,
                'method': 'lstm'
            },
            'metadata_encoder': {
                'categorical_dims': {'author': 100},
                'numerical_features': ['files_changed'],
                'embedding_dim': 32,
                'hidden_dim': 16
            },
            'fusion': {
                'method': 'cross_attention',
                'fusion_dim': 64
            },
            'task_heads': {
                'risk_prediction': {'num_classes': 3},
                'complexity_prediction': {'num_classes': 3},
                'hotspot_prediction': {'num_classes': 3},
                'urgency_prediction': {'num_classes': 3}
            }
        }
        
        # Initialize model
        model = MultiModalFusionNetwork(config=config)
        logger.info("Model initialization - SUCCESS")
        
        # Test forward pass
        import torch
        batch_size = 2
        text_input = torch.randint(0, 1000, (batch_size, 10))
        metadata_input = {
            'numerical_features': torch.randn(batch_size, 16),
            'author': torch.randint(0, 100, (batch_size,))
        }
        
        outputs = model(text_input, metadata_input)
        logger.info(f"Forward pass - SUCCESS, outputs: {list(outputs.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model structure test failed: {e}")
        return False

def test_quick_training():
    """Test quick training without Unicode output"""
    try:
        logger.info("Testing Quick Training...")
        
        # Check dataset
        current_dir = Path(__file__).parent
        data_path = current_dir / 'training_data' / 'improved_100k_multimodal_training.json'
        
        if not data_path.exists():
            logger.error("Dataset not found")
            return False
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if 'train_data' not in data or 'val_data' not in data:
            logger.error("Invalid dataset format")
            return False
            
        train_count = len(data['train_data'])
        val_count = len(data['val_data'])
        logger.info(f"Dataset loaded: {train_count} train, {val_count} val")
        
        # Test processors
        from multimodal_fusion.data_preprocessing.minimal_enhanced_text_processor import MinimalEnhancedTextProcessor
        
        text_processor = MinimalEnhancedTextProcessor(
            method="lstm", vocab_size=1000, max_length=128
        )
        
        # Use small subset
        subset = data['train_data'][:10]
        texts = [sample.get('text', '') for sample in subset]
        text_processor.fit(texts)
        logger.info("Quick training setup - SUCCESS")
        
        return True
        
    except Exception as e:
        logger.error(f"Quick training test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and return results"""
    logger.info("Starting Windows-Compatible Tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Text Processor Test", test_text_processor),
        ("Model Structure Test", test_model_structure),
        ("Quick Training Test", test_quick_training)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"Running: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            status = "PASS" if result else "FAIL"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"{test_name}: FAIL - {e}")
        
        logger.info("-" * 30)
    
    # Summary
    logger.info("=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("All tests PASSED - System ready for training!")
        return True
    else:
        logger.info("Some tests FAILED - Check logs for details")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
