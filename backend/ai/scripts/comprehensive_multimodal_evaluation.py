#!/usr/bin/env python3
"""
Comprehensive Multimodal Fusion Model Evaluation Script
========================================================

This script provides thorough evaluation of the trained multimodal fusion model,
including performance analysis, validation, and recommendations.
"""

import sys
import json
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

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

class ComprehensiveModelEvaluator:
    """Comprehensive evaluator for multimodal fusion model"""
    
    def __init__(self, model_path: Path, device: str = 'auto'):
        """Initialize evaluator with trained model"""
        self.device = self._setup_device(device)
        self.model_path = model_path
        self.model = None
        self.config = None
        self.text_processor = None
        self.metadata_processor = None
        self.evaluation_results = {}
        
        self._load_model()
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Using device: {device}")
        if device == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        return torch.device(device)
    
    def _load_model(self):
        """Load trained model and components"""
        logger.info(f"Loading model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract components
        self.config = checkpoint['config']
        self.text_processor = checkpoint['text_processor']
        self.metadata_processor = checkpoint['metadata_processor']
        
        # Initialize model
        self.model = MultiModalFusionNetwork(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded with {total_params:,} parameters")
        
        if 'best_val_accuracy' in checkpoint:
            logger.info(f"Best validation accuracy: {checkpoint['best_val_accuracy']:.4f}")
    
    def load_test_data(self, data_file: Path, sample_size: int = None) -> List[Dict]:
        """Load test data for evaluation"""
        logger.info(f"Loading test data from {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different data formats
        if 'data' in data:
            samples = data['data']
        elif isinstance(data, list):
            samples = data
        else:
            samples = [data]
        
        # Sample subset if requested
        if sample_size and len(samples) > sample_size:
            import random
            samples = random.sample(samples, sample_size)
            logger.info(f"Using {sample_size} random samples from {len(data)} total")
        
        logger.info(f"Loaded {len(samples)} test samples")
        return samples
    
    def evaluate_model_performance(self, test_samples: List[Dict]) -> Dict[str, Any]:
        """Comprehensive model performance evaluation"""
        logger.info("🔍 Starting comprehensive model evaluation...")
        
        # Prepare test data
        predictions = {task: [] for task in self.config['task_heads'].keys()}
        ground_truth = {task: [] for task in self.config['task_heads'].keys()}
        
        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(test_samples):
                if i % 100 == 0:
                    logger.info(f"Processing sample {i}/{len(test_samples)}")
                
                try:
                    # Process sample
                    text_features, metadata_features, labels = self._process_sample(sample)
                    
                    # Model prediction
                    outputs = self.model(text_features, metadata_features)
                    
                    # Extract predictions and labels
                    for task in predictions.keys():
                        pred = torch.argmax(outputs[task], dim=1).cpu().item()
                        predictions[task].append(pred)
                        ground_truth[task].append(labels[task])
                        
                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    continue
        
        # Calculate metrics for each task
        task_metrics = {}
        for task in predictions.keys():
            if len(predictions[task]) > 0:
                task_metrics[task] = self._calculate_task_metrics(
                    ground_truth[task], predictions[task], task
                )
        
        # Overall performance summary
        overall_accuracy = np.mean([metrics['accuracy'] for metrics in task_metrics.values()])
        
        evaluation_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_samples': len(test_samples),
            'processed_samples': len(predictions['risk_prediction']),
            'overall_accuracy': overall_accuracy,
            'task_metrics': task_metrics,
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'model_config': self.config
            }
        }
        
        logger.info(f"✅ Evaluation completed. Overall accuracy: {overall_accuracy:.4f}")
        return evaluation_results
    
    def _process_sample(self, sample: Dict) -> Tuple[torch.Tensor, Dict, Dict]:
        """Process a single sample for model evaluation"""
        # Extract text
        commit_text = sample.get('text', '') or sample.get('message', '')
        
        # Process text features
        text_features = self.text_processor.encode_text_lstm(commit_text)
        text_features = text_features.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Process metadata features
        metadata_features = self.metadata_processor.process_sample(sample)
        # Convert to proper format for model
        metadata_batch = {}
        for key, value in metadata_features.items():
            if isinstance(value, torch.Tensor):
                metadata_batch[key] = value.unsqueeze(0).to(self.device)
            else:
                metadata_batch[key] = torch.tensor([value]).to(self.device)
        
        # Generate labels (same logic as in training)
        labels = self._generate_sample_labels(sample, commit_text)
        
        return text_features, metadata_batch, labels
    
    def _generate_sample_labels(self, sample: Dict, text: str) -> Dict[str, int]:
        """Generate labels for evaluation sample"""
        metadata = self._extract_metadata(sample)
        
        labels = {}
        
        # Risk prediction
        risk_score = self._calculate_risk_score(text, metadata)
        labels['risk_prediction'] = 1 if risk_score > 0.5 else 0
        
        # Complexity prediction
        labels['complexity_prediction'] = self._calculate_complexity(text, metadata)
        
        # Hotspot prediction
        labels['hotspot_prediction'] = self._calculate_hotspot_score(text, metadata)
        
        # Urgency prediction
        labels['urgency_prediction'] = self._calculate_urgency(text, metadata)
        
        return labels
    
    def _extract_metadata(self, sample: Dict) -> Dict:
        """Extract metadata from sample"""
        return {
            'author': sample.get('author', 'unknown'),
            'files_changed': len(sample.get('files_changed', [])),
            'additions': sample.get('additions', 0),
            'deletions': sample.get('deletions', 0),
            'time_of_day': sample.get('time_of_day', 12),
            'day_of_week': sample.get('day_of_week', 1),
            'commit_size': sample.get('additions', 0) + sample.get('deletions', 0),
            'is_merge': 'merge' in sample.get('text', '').lower()
        }
    
    def _calculate_risk_score(self, text: str, metadata: Dict) -> float:
        """Calculate risk score"""
        risk_keywords = ['fix', 'bug', 'error', 'crash', 'security', 'vulnerability', 'critical']
        risk_score = 0.0
        
        text_lower = text.lower()
        for keyword in risk_keywords:
            if keyword in text_lower:
                risk_score += 0.2
        
        if metadata['commit_size'] > 1000:
            risk_score += 0.2
        if metadata['files_changed'] > 10:
            risk_score += 0.1
            
        return min(risk_score, 1.0)
    
    def _calculate_complexity(self, text: str, metadata: Dict) -> int:
        """Calculate complexity level"""
        commit_size = metadata['commit_size']
        files_changed = metadata['files_changed']
        
        if commit_size < 50 and files_changed <= 2:
            return 0  # simple
        elif commit_size < 500 and files_changed <= 10:
            return 1  # medium
        else:
            return 2  # complex
    
    def _calculate_hotspot_score(self, text: str, metadata: Dict) -> int:
        """Calculate hotspot score"""
        files_changed = metadata['files_changed']
        
        if files_changed <= 1:
            return 0
        elif files_changed <= 3:
            return 1
        elif files_changed <= 7:
            return 2
        elif files_changed <= 15:
            return 3
        else:
            return 4
    
    def _calculate_urgency(self, text: str, metadata: Dict) -> int:
        """Calculate urgency"""
        urgent_keywords = ['urgent', 'critical', 'hotfix', 'emergency', 'asap', 'immediately']
        text_lower = text.lower()
        
        for keyword in urgent_keywords:
            if keyword in text_lower:
                return 1
        
        if metadata['day_of_week'] in [6, 7] and metadata['commit_size'] > 500:
            return 1
            
        return 0
    
    def _calculate_task_metrics(self, y_true: List, y_pred: List, task_name: str) -> Dict:
        """Calculate comprehensive metrics for a task"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Class distribution
        unique, counts = np.unique(y_true, return_counts=True)
        class_distribution = dict(zip(unique.tolist(), counts.tolist()))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'class_distribution': class_distribution,
            'num_samples': len(y_true)
        }
    
    def analyze_model_architecture(self) -> Dict[str, Any]:
        """Analyze model architecture and components"""
        logger.info("🏗️ Analyzing model architecture...")
        
        analysis = {
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'architecture_breakdown': {},
            'config_analysis': self.config
        }
        
        # Analyze each component
        for name, module in self.model.named_children():
            param_count = sum(p.numel() for p in module.parameters())
            analysis['architecture_breakdown'][name] = {
                'parameters': param_count,
                'percentage': param_count / analysis['total_parameters'] * 100
            }
        
        return analysis
    
    def generate_recommendations(self, evaluation_results: Dict) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        overall_acc = evaluation_results['overall_accuracy']
        
        if overall_acc < 0.6:
            recommendations.append("🔴 LOW PERFORMANCE: Model accuracy is below 60%. Consider retraining with more data or different architecture.")
        elif overall_acc < 0.8:
            recommendations.append("🟡 MODERATE PERFORMANCE: Model shows potential but needs improvement. Consider fine-tuning or data augmentation.")
        else:
            recommendations.append("🟢 GOOD PERFORMANCE: Model performs well. Ready for deployment with monitoring.")
        
        # Task-specific recommendations
        for task, metrics in evaluation_results['task_metrics'].items():
            if metrics['accuracy'] < 0.5:
                recommendations.append(f"❌ {task}: Poor performance ({metrics['accuracy']:.3f}). Review label generation logic.")
            elif metrics['accuracy'] < 0.7:
                recommendations.append(f"⚠️ {task}: Needs improvement ({metrics['accuracy']:.3f}). Consider task-specific tuning.")
        
        # Data recommendations
        processed = evaluation_results['processed_samples']
        total = evaluation_results['test_samples']
        if processed < total * 0.9:
            recommendations.append(f"⚠️ DATA ISSUES: Only {processed}/{total} samples processed successfully. Check data preprocessing.")
        
        return recommendations
    
    def save_evaluation_report(self, evaluation_results: Dict, output_path: Path):
        """Save comprehensive evaluation report"""
        logger.info(f"💾 Saving evaluation report to {output_path}")
        
        # Add architecture analysis
        evaluation_results['architecture_analysis'] = self.analyze_model_architecture()
        
        # Add recommendations
        evaluation_results['recommendations'] = self.generate_recommendations(evaluation_results)
        
        # Save to JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ Evaluation report saved")

def main():
    """Main evaluation function"""
    logger.info("🚀 COMPREHENSIVE MULTIMODAL FUSION MODEL EVALUATION")
    logger.info("=" * 70)
    
    # Paths
    backend_dir = Path(__file__).parent.parent
    model_path = backend_dir / "trained_models" / "multimodal_fusion" / "best_multimodal_fusion_model.pth"
    data_file = backend_dir / "training_data" / "sample_preview.json"
    output_dir = backend_dir / "evaluation_results"
    
    # Check if model exists
    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        logger.info("Please ensure the model has been trained first.")
        return
    
    # Initialize evaluator
    evaluator = ComprehensiveModelEvaluator(model_path)
    
    # Load test data
    test_samples = evaluator.load_test_data(data_file, sample_size=500)  # Use subset for evaluation
    
    # Run comprehensive evaluation
    evaluation_results = evaluator.evaluate_model_performance(test_samples)
    
    # Generate and save report
    output_path = output_dir / f"comprehensive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    evaluator.save_evaluation_report(evaluation_results, output_path)
    
    # Print summary
    logger.info("\n📊 EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Overall Accuracy: {evaluation_results['overall_accuracy']:.4f}")
    logger.info(f"Processed Samples: {evaluation_results['processed_samples']}/{evaluation_results['test_samples']}")
    
    for task, metrics in evaluation_results['task_metrics'].items():
        logger.info(f"{task}: {metrics['accuracy']:.4f}")
    
    logger.info(f"\n📋 RECOMMENDATIONS:")
    for rec in evaluation_results['recommendations']:
        logger.info(f"  {rec}")
    
    logger.info(f"\n📄 Full report saved to: {output_path}")

if __name__ == "__main__":
    main()
