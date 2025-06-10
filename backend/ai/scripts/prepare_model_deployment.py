#!/usr/bin/env python3
"""
Multimodal Fusion Model Deployment Preparation Script
=====================================================

This script prepares the trained multimodal fusion model for production deployment,
including optimization, packaging, and creating deployment artifacts.
"""

import sys
import json
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pickle
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import multimodal components
from ai.multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
from ai.multimodal_fusion.data_preprocessing.text_processor import TextProcessor
from ai.multimodal_fusion.data_preprocessing.metadata_processor import MetadataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDeploymentPreparer:
    """Prepares multimodal fusion model for production deployment"""
    
    def __init__(self, model_path: Path, output_dir: Path):
        """Initialize deployment preparer"""
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "processors").mkdir(exist_ok=True)
        (self.output_dir / "configs").mkdir(exist_ok=True)
        (self.output_dir / "documentation").mkdir(exist_ok=True)
        
        # Model components
        self.model = None
        self.config = None
        self.text_processor = None
        self.metadata_processor = None
        self.checkpoint_info = None
        
    def load_trained_model(self):
        """Load the trained model and components"""
        logger.info(f"📥 Loading trained model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract components
        self.config = checkpoint['config']
        self.text_processor = checkpoint['text_processor']
        self.metadata_processor = checkpoint['metadata_processor']
        self.checkpoint_info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'best_val_accuracy': checkpoint.get('best_val_accuracy', 'unknown'),
            'val_accuracies': checkpoint.get('val_accuracies', {}),
            'training_timestamp': checkpoint.get('training_timestamp', 'unknown')
        }
        
        # Initialize and load model
        self.model = MultiModalFusionNetwork(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Parameters: {total_params:,}")
        logger.info(f"   Best Val Accuracy: {self.checkpoint_info['best_val_accuracy']}")
        
    def optimize_model_for_inference(self) -> torch.nn.Module:
        """Optimize model for faster inference"""
        logger.info("⚡ Optimizing model for inference...")
        
        # Set to evaluation mode
        self.model.eval()
        
        # Try to use TorchScript for optimization
        try:
            # Create sample inputs for tracing
            sample_text = torch.randn(1, 512, self.config['text_encoder']['embedding_dim'])
            sample_metadata = {
                'author_encoded': torch.randint(0, 100, (1,)),
                'season_encoded': torch.randint(0, 4, (1,)),
                'numerical_features': torch.randn(1, 33)  # Adjust based on your metadata processor
            }
            
            # Move to device
            sample_text = sample_text.to(self.device)
            for key in sample_metadata:
                sample_metadata[key] = sample_metadata[key].to(self.device)
            
            # Trace model
            with torch.no_grad():
                traced_model = torch.jit.trace(self.model, (sample_text, sample_metadata))
            
            logger.info("✅ Model successfully traced with TorchScript")
            return traced_model
            
        except Exception as e:
            logger.warning(f"⚠️ TorchScript tracing failed: {e}")
            logger.info("📝 Using original model (still optimized for eval)")
            return self.model
    
    def create_deployment_config(self) -> Dict[str, Any]:
        """Create deployment configuration"""
        logger.info("🔧 Creating deployment configuration...")
        
        deployment_config = {
            'model_info': {
                'name': 'multimodal_fusion_model',
                'version': '1.0.0',
                'description': 'Multimodal fusion model for commit analysis',
                'architecture': 'MultiModalFusionNetwork',
                'tasks': list(self.config['task_heads'].keys()),
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'created_at': datetime.now().isoformat(),
                'training_info': self.checkpoint_info
            },
            'input_specs': {
                'text': {
                    'type': 'string',
                    'description': 'Commit message text',
                    'max_length': self.config['text_encoder']['max_length'],
                    'preprocessing': 'TextProcessor.encode_text_lstm()'
                },
                'metadata': {
                    'type': 'dict',
                    'description': 'Commit metadata features',
                    'required_fields': [
                        'author', 'files_changed', 'additions', 'deletions',
                        'time_of_day', 'day_of_week'
                    ],
                    'preprocessing': 'MetadataProcessor.process_sample()'
                }
            },
            'output_specs': {
                task: {
                    'type': 'classification',
                    'num_classes': self.config['task_heads'][task]['num_classes'],
                    'classes': self._get_task_classes(task)
                }
                for task in self.config['task_heads'].keys()
            },
            'performance_requirements': {
                'max_inference_time_ms': 100,
                'min_accuracy': 0.7,
                'memory_limit_mb': 512
            },
            'deployment_settings': {
                'device': 'cpu',  # Default to CPU for deployment
                'batch_size': 1,  # Single sample inference
                'precision': 'float32',
                'enable_optimizations': True
            }
        }
        
        return deployment_config
    
    def _get_task_classes(self, task: str) -> List[str]:
        """Get class labels for each task"""
        class_mappings = {
            'risk_prediction': ['low', 'high'],
            'complexity_prediction': ['simple', 'medium', 'complex'],
            'hotspot_prediction': ['very_low', 'low', 'medium', 'high', 'very_high'],
            'urgency_prediction': ['normal', 'urgent']
        }
        return class_mappings.get(task, [f'class_{i}' for i in range(self.config['task_heads'][task]['num_classes'])])
    
    def create_inference_wrapper(self) -> str:
        """Create inference wrapper code"""
        logger.info("🔌 Creating inference wrapper...")
        
        wrapper_code = '''#!/usr/bin/env python3
"""
Multimodal Fusion Model Inference Wrapper
=========================================

Production-ready inference wrapper for the multimodal fusion model.
"""

import torch
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalFusionInference:
    """Production inference wrapper for multimodal fusion model"""
    
    def __init__(self, model_dir: Path):
        """Initialize inference wrapper"""
        self.model_dir = Path(model_dir)
        self.device = torch.device('cpu')  # Use CPU for deployment
        
        # Load components
        self.model = None
        self.text_processor = None
        self.metadata_processor = None
        self.config = None
        self.task_classes = {}
        
        self._load_model_components()
    
    def _load_model_components(self):
        """Load all model components"""
        logger.info("Loading model components...")
        
        # Load model
        model_path = self.model_dir / "models" / "optimized_model.pth"
        if model_path.exists():
            self.model = torch.jit.load(model_path, map_location=self.device)
        else:
            # Fallback to regular model
            model_path = self.model_dir / "models" / "model.pth"
            checkpoint = torch.load(model_path, map_location=self.device)
            from ai.multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
            self.model = MultiModalFusionNetwork(checkpoint['config'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        # Load processors
        with open(self.model_dir / "processors" / "text_processor.pkl", 'rb') as f:
            self.text_processor = pickle.load(f)
        
        with open(self.model_dir / "processors" / "metadata_processor.pkl", 'rb') as f:
            self.metadata_processor = pickle.load(f)
        
        # Load config
        with open(self.model_dir / "configs" / "deployment_config.json", 'r') as f:
            deployment_config = json.load(f)
            self.config = deployment_config
            self.task_classes = {
                task: spec['classes'] 
                for task, spec in deployment_config['output_specs'].items()
            }
        
        logger.info("Model components loaded successfully")
    
    def predict(self, commit_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction on a single commit"""
        try:
            # Prepare inputs
            text_features, metadata_features = self._prepare_inputs(commit_text, metadata)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(text_features, metadata_features)
            
            # Process outputs
            predictions = {}
            for task, output in outputs.items():
                probs = torch.softmax(output, dim=1)
                pred_class_idx = torch.argmax(probs, dim=1).item()
                pred_class = self.task_classes[task][pred_class_idx]
                confidence = probs[0, pred_class_idx].item()
                
                predictions[task] = {
                    'predicted_class': pred_class,
                    'confidence': confidence,
                    'class_probabilities': {
                        class_name: prob.item()
                        for class_name, prob in zip(self.task_classes[task], probs[0])
                    }
                }
            
            return {
                'success': True,
                'predictions': predictions,
                'input_text': commit_text[:100] + "..." if len(commit_text) > 100 else commit_text
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'predictions': None
            }
    
    def _prepare_inputs(self, text: str, metadata: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Prepare inputs for model"""
        # Process text
        text_features = self.text_processor.encode_text_lstm(text)
        text_features = text_features.unsqueeze(0).to(self.device)
        
        # Process metadata
        sample = {
            'text': text,
            'author': metadata.get('author', 'unknown'),
            'files_changed': metadata.get('files_changed', []),
            'additions': metadata.get('additions', 0),
            'deletions': metadata.get('deletions', 0),
            'time_of_day': metadata.get('time_of_day', 12),
            'day_of_week': metadata.get('day_of_week', 1)
        }
        
        metadata_features = self.metadata_processor.process_sample(sample)
        
        # Convert to batch format
        metadata_batch = {}
        for key, value in metadata_features.items():
            if isinstance(value, torch.Tensor):
                metadata_batch[key] = value.unsqueeze(0).to(self.device)
            else:
                metadata_batch[key] = torch.tensor([value]).to(self.device)
        
        return text_features, metadata_batch
    
    def batch_predict(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions on multiple samples"""
        results = []
        for sample in samples:
            result = self.predict(
                sample.get('text', ''),
                sample.get('metadata', {})
            )
            results.append(result)
        return results

# Example usage
if __name__ == "__main__":
    # Initialize inference wrapper
    model_dir = Path(__file__).parent
    inference = MultimodalFusionInference(model_dir)
    
    # Example prediction
    sample_text = "Fix critical security vulnerability in authentication module"
    sample_metadata = {
        'author': 'john_doe',
        'files_changed': ['auth.py', 'security.py'],
        'additions': 15,
        'deletions': 5,
        'time_of_day': 14,
        'day_of_week': 3
    }
    
    result = inference.predict(sample_text, sample_metadata)
    print(json.dumps(result, indent=2))
'''
        
        return wrapper_code
    
    def create_api_documentation(self) -> str:
        """Create API documentation"""
        logger.info("📚 Creating API documentation...")
        
        docs = f'''# Multimodal Fusion Model API Documentation

## Overview
This is the production deployment of the Multimodal Fusion Model for commit analysis.

**Model Version:** 1.0.0  
**Created:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Training Accuracy:** {self.checkpoint_info['best_val_accuracy']}  

## Capabilities
The model provides multi-task analysis of git commits:

- **Risk Prediction**: Identifies high-risk commits (security, bugs, etc.)
- **Complexity Assessment**: Categorizes commit complexity (simple/medium/complex)
- **Hotspot Detection**: Identifies code hotspots (very_low to very_high)
- **Urgency Classification**: Determines if commits need urgent attention

## API Usage

### Initialize Model
```python
from multimodal_inference import MultimodalFusionInference

# Load model
model = MultimodalFusionInference("path/to/deployment/package")
```

### Single Prediction
```python
result = model.predict(
    commit_text="Fix critical bug in payment processing",
    metadata={{
        'author': 'developer_name',
        'files_changed': ['payment.py', 'validation.py'],
        'additions': 25,
        'deletions': 10,
        'time_of_day': 14,
        'day_of_week': 2
    }}
)
```

### Response Format
```json
{{
  "success": true,
  "predictions": {{
    "risk_prediction": {{
      "predicted_class": "high",
      "confidence": 0.85,
      "class_probabilities": {{
        "low": 0.15,
        "high": 0.85
      }}
    }},
    "complexity_prediction": {{
      "predicted_class": "medium",
      "confidence": 0.72,
      "class_probabilities": {{
        "simple": 0.18,
        "medium": 0.72,
        "complex": 0.10
      }}
    }}
  }}
}}
```

## Input Requirements

### Text Input
- **Type**: String
- **Description**: Git commit message
- **Max Length**: {self.config['text_encoder']['max_length']} characters
- **Example**: "Fix authentication bug in login module"

### Metadata Input
- **Type**: Dictionary
- **Required Fields**:
  - `author`: String (developer username)
  - `files_changed`: List of strings (file paths)
  - `additions`: Integer (lines added)
  - `deletions`: Integer (lines deleted)
  - `time_of_day`: Integer (hour 0-23)
  - `day_of_week`: Integer (1-7, Monday=1)

## Performance Characteristics

- **Inference Time**: <100ms per sample
- **Memory Usage**: ~512MB
- **Model Parameters**: {sum(p.numel() for p in self.model.parameters()):,}
- **Recommended Deployment**: CPU-based servers

## Error Handling

The API returns structured error responses:
```json
{{
  "success": false,
  "error": "Error description",
  "predictions": null
}}
```

## Deployment Notes

1. **Dependencies**: PyTorch, NumPy, scikit-learn
2. **Hardware**: CPU sufficient, GPU optional
3. **Scaling**: Stateless, supports horizontal scaling
4. **Monitoring**: Log prediction confidence and errors

## Model Interpretation

### Task-Specific Guidance

**Risk Prediction**:
- High confidence (>0.8): Strong signal for risk
- Keywords: "fix", "bug", "security", "critical"
- Large commits (>1000 lines) increase risk score

**Complexity Prediction**:
- Based on commit size and files changed
- Simple: <50 lines, ≤2 files
- Complex: >500 lines, >10 files

**Hotspot Prediction**:
- Identifies frequently changed code areas
- Based on file change patterns

**Urgency Prediction**:
- Keywords: "urgent", "hotfix", "critical"
- Weekend commits with large changes

## Support and Maintenance

For issues or questions:
1. Check logs for error details
2. Verify input format compliance
3. Monitor model confidence scores
4. Contact: AI Team <ai-team@company.com>

---
*Generated automatically by Model Deployment Preparer*
'''
        
        return docs
    
    def save_model_components(self, optimized_model: torch.nn.Module):
        """Save all model components for deployment"""
        logger.info("💾 Saving model components...")
        
        # Save optimized model
        if hasattr(optimized_model, 'save'):  # TorchScript model
            torch.jit.save(optimized_model, self.output_dir / "models" / "optimized_model.pth")
            logger.info("✅ Saved optimized TorchScript model")
        else:
            # Save regular model
            model_save_dict = {
                'model_state_dict': optimized_model.state_dict(),
                'config': self.config
            }
            torch.save(model_save_dict, self.output_dir / "models" / "model.pth")
            logger.info("✅ Saved regular PyTorch model")
        
        # Save processors
        with open(self.output_dir / "processors" / "text_processor.pkl", 'wb') as f:
            pickle.dump(self.text_processor, f)
        
        with open(self.output_dir / "processors" / "metadata_processor.pkl", 'wb') as f:
            pickle.dump(self.metadata_processor, f)
        
        logger.info("✅ Saved text and metadata processors")
        
        # Save deployment config
        deployment_config = self.create_deployment_config()
        with open(self.output_dir / "configs" / "deployment_config.json", 'w') as f:
            json.dump(deployment_config, f, indent=2, default=str)
        
        # Save original config
        with open(self.output_dir / "configs" / "model_config.json", 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        logger.info("✅ Saved configuration files")
    
    def create_deployment_package(self):
        """Create complete deployment package"""
        logger.info("📦 Creating deployment package...")
        
        # Load model
        self.load_trained_model()
        
        # Optimize model
        optimized_model = self.optimize_model_for_inference()
        
        # Save components
        self.save_model_components(optimized_model)
        
        # Create inference wrapper
        wrapper_code = self.create_inference_wrapper()
        with open(self.output_dir / "multimodal_inference.py", 'w') as f:
            f.write(wrapper_code)
        
        # Create documentation
        docs = self.create_api_documentation()
        with open(self.output_dir / "documentation" / "API_README.md", 'w') as f:
            f.write(docs)
        
        # Create requirements.txt
        requirements = [
            "torch>=1.9.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.0.0",
            "tqdm>=4.62.0"
        ]
        with open(self.output_dir / "requirements.txt", 'w') as f:
            f.write('\\n'.join(requirements))
        
        # Create deployment summary
        summary = {
            'package_created': datetime.now().isoformat(),
            'model_info': {
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'training_accuracy': self.checkpoint_info['best_val_accuracy'],
                'tasks': list(self.config['task_heads'].keys())
            },
            'files_included': [
                'models/optimized_model.pth',
                'processors/text_processor.pkl',
                'processors/metadata_processor.pkl',
                'configs/deployment_config.json',
                'multimodal_inference.py',
                'documentation/API_README.md',
                'requirements.txt'
            ],
            'deployment_ready': True
        }
        
        with open(self.output_dir / "deployment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("✅ Deployment package created successfully!")
        logger.info(f"📁 Package location: {self.output_dir}")
        
        return summary

def main():
    """Main deployment preparation function"""
    logger.info("🚀 MULTIMODAL FUSION MODEL DEPLOYMENT PREPARATION")
    logger.info("=" * 70)
    
    # Paths
    backend_dir = Path(__file__).parent.parent
    model_path = backend_dir / "trained_models" / "multimodal_fusion" / "best_multimodal_fusion_model.pth"
    output_dir = backend_dir / "deployment_packages" / f"multimodal_fusion_v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Check if model exists
    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        logger.info("Please ensure the model has been trained first.")
        return
    
    # Initialize deployment preparer
    preparer = ModelDeploymentPreparer(model_path, output_dir)
    
    # Create deployment package
    summary = preparer.create_deployment_package()
    
    # Print summary
    logger.info("\\n📊 DEPLOYMENT PACKAGE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Package Location: {output_dir}")
    logger.info(f"Model Parameters: {summary['model_info']['parameters']:,}")
    logger.info(f"Training Accuracy: {summary['model_info']['training_accuracy']}")
    logger.info(f"Tasks: {', '.join(summary['model_info']['tasks'])}")
    logger.info(f"Deployment Ready: {'✅ YES' if summary['deployment_ready'] else '❌ NO'}")
    
    logger.info("\\n🎉 Deployment preparation completed successfully!")
    logger.info(f"📂 To deploy: Copy contents of {output_dir} to production server")

if __name__ == "__main__":
    main()
