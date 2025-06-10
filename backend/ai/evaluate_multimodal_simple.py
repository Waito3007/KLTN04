#!/usr/bin/env python3
"""
Multi-Modal Fusion Network - Simple Evaluation
=============================================
Simplified evaluation script for Multi-Modal Fusion Network
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

# Import our components
from multimodal_fusion.data_preprocessing.text_processor import TextProcessor
from multimodal_fusion.data_preprocessing.metadata_processor import MetadataProcessor
from multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
from train_clean_data import load_cleaned_data, prepare_training_data

class SimpleMultiModalEvaluator:
    """Simplified evaluator for Multi-Modal Fusion Network"""
    
    def __init__(self, model_path: str):
        """Load trained model and processors"""
        print("🔧 Loading trained Multi-Modal Fusion Network...")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        self.text_processor = checkpoint['text_processor']
        self.metadata_processor = checkpoint['metadata_processor']
        self.task_configs = checkpoint['task_configs']
        self.metadata_dims = checkpoint['metadata_dims']
          # Initialize model architecture
        self.model = MultiModalFusionNetwork(
            text_dim=self.text_processor.embed_dim,
            **self.metadata_dims,
            hidden_dim=256,
            dropout_rate=0.3,
            task_configs=self.task_configs  # Add task configurations
        )
          # Load trained weights (allow missing task heads for models trained without them)
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        if missing_keys:
            print(f"⚠️  Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
            print(f"   First few missing keys: {missing_keys[:3]}...")
            print("   This usually means task heads need to be trained from scratch")
        
        if unexpected_keys:
            print(f"⚠️  Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"✅ Model loaded successfully on {self.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def evaluate_on_dataset(self, texts: List[str], metadata_list: List[Dict], 
                           labels_list: List[Dict]) -> Dict[str, Any]:
        """Evaluate model on a dataset"""
        print(f"\n🔍 Evaluating on {len(texts)} samples...")
        
        all_predictions = {task: [] for task in self.task_configs.keys()}
        all_true_labels = {task: [] for task in self.task_configs.keys()}
        all_probabilities = {task: [] for task in self.task_configs.keys()}
        
        # Process in batches
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadata = metadata_list[i:i + batch_size]
                batch_labels = labels_list[i:i + batch_size]
                
                # Process inputs
                text_batch_result = self.text_processor.process_batch(batch_texts)
                metadata_batch_result = self.metadata_processor.process_batch(batch_metadata)
                
                # Convert to model input
                text_input = text_batch_result['embeddings'].to(self.device)
                metadata_input = {
                    'numerical_features': metadata_batch_result['numerical_features'].to(self.device),
                    'author_encoded': metadata_batch_result['author_encoded'].to(self.device),
                    'season_encoded': metadata_batch_result['season_encoded'].to(self.device),
                    'file_types_encoded': metadata_batch_result['file_types_encoded'].to(self.device)
                }
                
                # Forward pass
                outputs = self.model(text_input, metadata_input)
                  # Collect predictions and true labels
                for j, labels in enumerate(batch_labels):
                    for task in self.task_configs.keys():
                        if task in outputs and task in labels:
                            # Get prediction
                            logits = outputs[task][j:j+1]  # Single sample
                            probabilities = torch.softmax(logits, dim=1)
                            prediction = torch.argmax(probabilities, dim=1).cpu().item()
                            
                            all_predictions[task].append(prediction)
                            all_true_labels[task].append(labels[task])
                            all_probabilities[task].append(probabilities.cpu().numpy()[0])
                        else:
                            # Debug missing tasks
                            if task not in outputs:
                                print(f"⚠️  Task '{task}' not found in model outputs. Available: {list(outputs.keys())}")
                            if task not in labels:
                                print(f"⚠️  Task '{task}' not found in labels. Available: {list(labels.keys())}")
                
                # Debug output after first batch
                if i == 0:
                    print(f"🔍 Model output tasks: {list(outputs.keys())}")
                    print(f"🔍 Label tasks: {list(batch_labels[0].keys()) if batch_labels else 'No labels'}")
                    print(f"🔍 Expected tasks: {list(self.task_configs.keys())}")
                    for task in self.task_configs.keys():
                        print(f"🔍 Task '{task}' predictions so far: {len(all_predictions[task])}")
        
        return self._calculate_metrics(all_predictions, all_true_labels, all_probabilities)

    def _calculate_metrics(self, predictions: Dict, true_labels: Dict, 
                          probabilities: Dict) -> Dict[str, Any]:
        """Calculate comprehensive metrics for all tasks"""
        results = {}
        
        for task in predictions.keys():
            if len(predictions[task]) == 0:
                continue
                
            y_true = np.array(true_labels[task])
            y_pred = np.array(predictions[task])
            y_proba = np.array(probabilities[task])
              # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            # Per-class metrics
            per_class_report = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            results[task] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': len(y_true),  # Use actual sample count instead of support sum
                'per_class_metrics': per_class_report,
                'confusion_matrix': cm.tolist(),
                'class_distribution': np.bincount(y_true).tolist(),
                'prediction_distribution': np.bincount(y_pred).tolist()
            }
            
            print(f"\n📊 {task.replace('_', ' ').title()} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Samples: {len(y_true)}")
        
        return results

    def analyze_model_architecture(self) -> Dict[str, Any]:
        """Analyze model architecture and parameters"""
        print("\n🏗️ Analyzing Model Architecture...")
        
        # Count parameters by component
        text_params = sum(p.numel() for p in self.model.text_branch.parameters())
        metadata_params = sum(p.numel() for p in self.model.metadata_branch.parameters())
        fusion_params = sum(p.numel() for p in self.model.fusion.parameters()) if self.model.fusion else 0
        
        task_head_params = {}
        total_task_params = 0
        for task_name in self.task_configs.keys():
            if hasattr(self.model, 'task_heads') and task_name in self.model.task_heads:
                head = self.model.task_heads[task_name]
                params = sum(p.numel() for p in head.parameters())
                task_head_params[task_name] = params
                total_task_params += params
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        architecture_info = {
            'total_parameters': total_params,
            'text_branch_parameters': text_params,
            'metadata_branch_parameters': metadata_params,
            'fusion_parameters': fusion_params,
            'task_heads_parameters': task_head_params,
            'total_task_parameters': total_task_params,
            'text_dimension': self.text_processor.embed_dim,
            'metadata_dimensions': self.metadata_dims,
            'task_configurations': self.task_configs
        }
        
        print(f"📐 Total Parameters: {total_params:,}")
        print(f"📝 Text Branch: {text_params:,} ({text_params/total_params*100:.1f}%)")
        print(f"📊 Metadata Branch: {metadata_params:,} ({metadata_params/total_params*100:.1f}%)")
        print(f"🔗 Fusion Layer: {fusion_params:,} ({fusion_params/total_params*100:.1f}%)")
        print(f"🎯 Task Heads: {total_task_params:,} ({total_task_params/total_params*100:.1f}%)")
        
        return architecture_info

    def generate_evaluation_report(self, test_data_path: str) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        print("\n🎯 Generating Multi-Modal Fusion Network Evaluation Report")
        print("=" * 70)
        
        # Load test data
        test_samples = load_cleaned_data(test_data_path, max_samples=500)
        test_texts, test_metadata, test_labels = prepare_training_data(test_samples)
        
        # 1. Model Architecture Analysis
        architecture_info = self.analyze_model_architecture()
        
        # 2. Performance Evaluation
        performance_results = self.evaluate_on_dataset(test_texts, test_metadata, test_labels)
        
        # Calculate summary metrics
        task_f1_scores = [metrics['f1_score'] for metrics in performance_results.values()]
        avg_f1 = np.mean(task_f1_scores)
        
        # Compile report
        report = {
            'model_type': 'Multi-Modal Fusion Network',
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'test_dataset_size': len(test_samples),
            'architecture_analysis': architecture_info,
            'performance_metrics': performance_results,
            'summary_metrics': {
                'average_f1_score': avg_f1,
                'task_f1_scores': {task: metrics['f1_score'] for task, metrics in performance_results.items()},
                'best_performing_task': max(performance_results.keys(), key=lambda k: performance_results[k]['f1_score']),
                'worst_performing_task': min(performance_results.keys(), key=lambda k: performance_results[k]['f1_score'])
            }
        }
        
        return report

def main():
    """Main evaluation function"""
    print("🎯 Multi-Modal Fusion Network - Simple Evaluation")
    print("=" * 60)
    
    # Configuration
    model_path = "trained_models/multimodal_fusion_clean_data.pth"
    test_data_path = "training_data/sample_preview.json"
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("💡 Please train the model first using train_clean_data.py")
        return
    
    # Initialize evaluator
    evaluator = SimpleMultiModalEvaluator(model_path)
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(test_data_path)
    
    # Save report
    report_path = "evaluation_results/multimodal_fusion_simple_report.json"
    Path("evaluation_results").mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Evaluation report saved to: {report_path}")
    
    # Print summary
    print("\n📋 EVALUATION SUMMARY")
    print("=" * 40)
    summary = report['summary_metrics']
    
    print(f"🎯 Average F1-Score: {summary['average_f1_score']:.4f}")
    print(f"🏆 Best Task: {summary['best_performing_task']} (F1: {summary['task_f1_scores'][summary['best_performing_task']]:.4f})")
    print(f"🎯 Focus Area: {summary['worst_performing_task']} (F1: {summary['task_f1_scores'][summary['worst_performing_task']]:.4f})")
    
    print(f"\n🏗️ Model Architecture:")
    print(f"📐 Total Parameters: {report['architecture_analysis']['total_parameters']:,}")
    print(f"📝 Text Branch: {report['architecture_analysis']['text_branch_parameters']:,}")
    print(f"📊 Metadata Branch: {report['architecture_analysis']['metadata_branch_parameters']:,}")
    print(f"🔗 Fusion: {report['architecture_analysis']['fusion_parameters']:,}")
    
    print(f"\n✨ Next Steps:")
    print(f"1. Review detailed metrics in {report_path}")
    print(f"2. Compare with baseline models")
    print(f"3. Implement improvements for weakest task")
    print(f"4. Deploy for production testing")

if __name__ == "__main__":
    main()
