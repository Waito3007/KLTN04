"""
Comprehensive Multimodal Model Evaluation
Evaluates the model structure, data flow, and potential issues
"""

import os
import sys
import torch
import json
import logging
from datetime import datetime

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model_structure():
    """Evaluate the multimodal model structure and components"""
    
    logger.info("🔍 Starting Comprehensive Multimodal Model Evaluation...")
    
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'model_structure': {},
        'data_flow': {},
        'component_tests': {},
        'integration_tests': {},
        'issues_found': [],
        'recommendations': []
    }
    
    try:
        # 1. Test Data Loading
        logger.info("📊 Testing Data Loading...")
        
        data_path = os.path.join(current_dir, 'training_data', 'improved_100k_multimodal_training.json')
        if not os.path.exists(data_path):
            evaluation_results['issues_found'].append("Training data file not found")
            return evaluation_results
            
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if 'train_data' in data and 'val_data' in data:
            train_data = data['train_data']
            val_data = data['val_data']
            
            evaluation_results['data_flow']['train_samples'] = len(train_data)
            evaluation_results['data_flow']['val_samples'] = len(val_data)
            evaluation_results['data_flow']['total_samples'] = len(train_data) + len(val_data)
            
            # Check sample structure
            if len(train_data) > 0:
                sample = train_data[0]
                evaluation_results['data_flow']['sample_structure'] = {
                    'keys': list(sample.keys()),
                    'text_field': 'text' in sample,
                    'metadata_field': 'metadata' in sample,
                    'labels_field': 'labels' in sample
                }
                
                if 'metadata' in sample:
                    metadata = sample['metadata']
                    evaluation_results['data_flow']['metadata_structure'] = list(metadata.keys())
                    
                if 'labels' in sample:
                    labels = sample['labels']
                    evaluation_results['data_flow']['label_structure'] = list(labels.keys())
            
            logger.info(f"✅ Data loading successful: {len(train_data)} train, {len(val_data)} val")
        else:
            evaluation_results['issues_found'].append("Unexpected data format - missing train_data/val_data")
            
    except Exception as e:
        evaluation_results['issues_found'].append(f"Data loading error: {str(e)}")
        logger.error(f"❌ Data loading failed: {e}")
    
    try:
        # 2. Test Text Processor
        logger.info("📝 Testing Enhanced Text Processor...")
        
        from multimodal_fusion.data_preprocessing.minimal_enhanced_text_processor import MinimalEnhancedTextProcessor
        
        text_processor = MinimalEnhancedTextProcessor(
            method="lstm",
            vocab_size=1000,  # Smaller for testing
            max_length=128,
            enable_sentiment=True,
            enable_advanced_cleaning=True
        )
        
        # Test fitting
        test_texts = ["fix: update user authentication", "feat: add new dashboard", "docs: update README"]
        text_processor.fit(test_texts)
        
        # Test encoding
        encoded = text_processor.encode_text_lstm(test_texts[0])
        evaluation_results['component_tests']['text_processor'] = {
            'vocab_size': text_processor.get_vocab_size(),
            'encoding_shape': list(encoded.shape),
            'encoding_dtype': str(encoded.dtype)
        }
        
        # Test enhanced features
        features = text_processor.extract_enhanced_features(test_texts[0])
        evaluation_results['component_tests']['enhanced_features'] = {
            'feature_count': len(features),
            'feature_keys': list(features.keys())
        }
        
        logger.info(f"✅ Text processor test successful: vocab={text_processor.get_vocab_size()}")
        
    except Exception as e:
        evaluation_results['issues_found'].append(f"Text processor error: {str(e)}")
        logger.error(f"❌ Text processor test failed: {e}")
    
    try:
        # 3. Test Metadata Processor
        logger.info("🗃️ Testing Metadata Processor...")
        
        from multimodal_fusion.data_preprocessing.metadata_processor import MetadataProcessor
        
        metadata_processor = MetadataProcessor()
        
        # Test fitting
        test_metadata = [
            {
                'author': 'user1',
                'files_changed': 3,
                'insertions': 50,
                'deletions': 10,
                'hour_of_day': 14,
                'day_of_week': 3,
                'is_merge': False,
                'commit_size': 'medium'
            }
        ]
        
        metadata_processor.fit(test_metadata)
        
        evaluation_results['component_tests']['metadata_processor'] = {
            'fitted': True,
            'test_sample_structure': list(test_metadata[0].keys())
        }
        
        logger.info("✅ Metadata processor test successful")
        
    except Exception as e:
        evaluation_results['issues_found'].append(f"Metadata processor error: {str(e)}")
        logger.error(f"❌ Metadata processor test failed: {e}")
    
    try:
        # 4. Test Model Architecture
        logger.info("🏗️ Testing Model Architecture...")
        
        from multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
        
        # Test with config-based initialization
        model_config = {
            'text_encoder': {
                'vocab_size': 1000,
                'embedding_dim': 64,
                'hidden_dim': 32,
                'num_layers': 1,
                'method': 'lstm'
            },
            'metadata_encoder': {
                'categorical_dims': {
                    'author': 100
                },
                'numerical_features': ['files_changed', 'insertions', 'deletions'],
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
        
        model = MultiModalFusionNetwork(config=model_config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        evaluation_results['model_structure'] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': model_config
        }
        
        logger.info(f"✅ Model architecture test successful: {total_params:,} parameters")
        
        # 5. Test Forward Pass
        logger.info("⚡ Testing Model Forward Pass...")
        
        batch_size = 4
        seq_length = 10
        
        # Create test inputs
        text_input = torch.randint(0, 1000, (batch_size, seq_length))
        metadata_input = {
            'numerical_features': torch.randn(batch_size, 28),  # 10 base + 18 enhanced
            'author': torch.randint(0, 100, (batch_size,))
        }
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(text_input, metadata_input)
            
        evaluation_results['integration_tests']['forward_pass'] = {
            'successful': True,
            'output_tasks': list(outputs.keys()),
            'output_shapes': {task: list(output.shape) for task, output in outputs.items()}
        }
        
        logger.info(f"✅ Forward pass successful: {list(outputs.keys())}")
        
        # 6. Test Loss Calculation
        logger.info("🎯 Testing Loss Calculation...")
        
        criterion = torch.nn.CrossEntropyLoss()
        labels = torch.randint(0, 3, (batch_size, 4))  # 4 tasks, 3 classes each
        
        total_loss = 0
        task_names = ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']
        
        for task_idx, task_name in enumerate(task_names):
            if task_name in outputs:
                task_output = outputs[task_name]
                task_labels = labels[:, task_idx]
                task_loss = criterion(task_output, task_labels)
                total_loss += task_loss
        
        evaluation_results['integration_tests']['loss_calculation'] = {
            'successful': True,
            'total_loss': float(total_loss.item()),
            'tasks_tested': task_names
        }
        
        logger.info(f"✅ Loss calculation successful: {total_loss.item():.4f}")
        
    except Exception as e:
        evaluation_results['issues_found'].append(f"Model architecture error: {str(e)}")
        logger.error(f"❌ Model test failed: {e}")
    
    # 7. Performance Analysis
    logger.info("📈 Analyzing Performance Characteristics...")
    
    if len(evaluation_results['issues_found']) == 0:
        evaluation_results['recommendations'].extend([
            "✅ All core components are working correctly",
            "🚀 Model is ready for full training",
            "💡 Consider monitoring training metrics closely",
            "🔧 Validate on small dataset first before full training"
        ])
    else:
        evaluation_results['recommendations'].extend([
            "⚠️ Fix identified issues before proceeding",
            "🔍 Review component integration",
            "🛠️ Debug individual components if needed"
        ])
    
    # 8. Resource Analysis
    if 'model_structure' in evaluation_results and evaluation_results['model_structure']:
        params = evaluation_results['model_structure']['total_parameters']
        if params > 10_000_000:
            evaluation_results['recommendations'].append("💡 Large model - consider reducing parameters or using gradient checkpointing")
        elif params < 100_000:
            evaluation_results['recommendations'].append("💡 Small model - might need more capacity for complex tasks")
        else:
            evaluation_results['recommendations'].append("✅ Model size is reasonable for the task")
    
    return evaluation_results

def main():
    """Main evaluation function"""
    
    results = evaluate_model_structure()
    
    # Save results
    output_path = os.path.join(current_dir, 'multimodal_evaluation_report.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 MULTIMODAL MODEL EVALUATION SUMMARY")
    print("="*80)
    
    if results.get('data_flow'):
        print(f"📊 Data: {results['data_flow'].get('total_samples', 0):,} samples")
    
    if results.get('model_structure'):
        print(f"🏗️ Model: {results['model_structure'].get('total_parameters', 0):,} parameters")
    
    if results.get('component_tests'):
        print(f"🧪 Components: {len([k for k, v in results['component_tests'].items() if v])} tested")
    
    if results.get('integration_tests'):
        print(f"⚡ Integration: {len([k for k, v in results['integration_tests'].items() if v.get('successful')])} tests passed")
    
    print(f"\n❌ Issues Found: {len(results['issues_found'])}")
    for issue in results['issues_found']:
        print(f"   • {issue}")
    
    print(f"\n💡 Recommendations:")
    for rec in results['recommendations']:
        print(f"   • {rec}")
    
    print(f"\n📄 Full report saved to: {output_path}")
    print("="*80)
    
    return len(results['issues_found']) == 0

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 All tests passed! Model is ready for training.")
    else:
        print("⚠️ Issues found. Please review and fix before training.")
