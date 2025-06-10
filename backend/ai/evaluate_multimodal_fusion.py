#!/usr/bin/env python3
"""
Multi-Modal Fusion Network - Comprehensive Evaluation
====================================================
Đánh giá toàn diện mô hình Multi-Modal Fusion Network
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

# Import our components
from multimodal_fusion.data_preprocessing.text_processor import TextProcessor
from multimodal_fusion.data_preprocessing.metadata_processor import MetadataProcessor
from multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork
from train_clean_data import load_cleaned_data, prepare_training_data

class MultiModalEvaluator:
    """Comprehensive evaluator for Multi-Modal Fusion Network"""
    
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
            dropout_rate=0.3
        )
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
                'support': int(np.sum(support)),
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
    
    def analyze_feature_importance(self, texts: List[str], metadata_list: List[Dict], 
                                 sample_size: int = 50) -> Dict[str, Any]:
        """Analyze feature importance using attention weights and gradients"""
        print(f"\n🔍 Analyzing Feature Importance (sample size: {sample_size})...")
        
        # Sample subset for analysis
        sample_indices = np.random.choice(len(texts), min(sample_size, len(texts)), replace=False)
        sample_texts = [texts[i] for i in sample_indices]
        sample_metadata = [metadata_list[i] for i in sample_indices]
        
        feature_analysis = {
            'text_attention_weights': [],
            'metadata_feature_gradients': {},
            'fusion_contributions': []
        }
        
        # Enable gradient computation
        self.model.train()
        
        with torch.enable_grad():
            for i in range(0, len(sample_texts), 8):  # Small batches
                batch_texts = sample_texts[i:i+8]
                batch_metadata = sample_metadata[i:i+8]
                
                # Process inputs
                text_batch_result = self.text_processor.process_batch(batch_texts)
                metadata_batch_result = self.metadata_processor.process_batch(batch_metadata)
                
                # Convert to model input with gradients
                text_input = text_batch_result['embeddings'].to(self.device)
                text_input.requires_grad_(True)
                
                metadata_input = {
                    'numerical_features': metadata_batch_result['numerical_features'].to(self.device),
                    'author_encoded': metadata_batch_result['author_encoded'].to(self.device),
                    'season_encoded': metadata_batch_result['season_encoded'].to(self.device),
                    'file_types_encoded': metadata_batch_result['file_types_encoded'].to(self.device)
                }
                
                for key in metadata_input:
                    metadata_input[key].requires_grad_(True)
                
                # Forward pass
                outputs = self.model(text_input, metadata_input)
                
                # Compute gradients for first task (as example)
                first_task = list(outputs.keys())[0]
                loss = outputs[first_task].sum()
                loss.backward()
                
                # Collect gradient magnitudes
                if text_input.grad is not None:
                    text_importance = torch.norm(text_input.grad, dim=-1).mean().item()
                    feature_analysis['text_attention_weights'].append(text_importance)
                
                for key in metadata_input:
                    if metadata_input[key].grad is not None:
                        grad_magnitude = torch.norm(metadata_input[key].grad).item()
                        if key not in feature_analysis['metadata_feature_gradients']:
                            feature_analysis['metadata_feature_gradients'][key] = []
                        feature_analysis['metadata_feature_gradients'][key].append(grad_magnitude)
        
        # Calculate average importance
        avg_text_importance = np.mean(feature_analysis['text_attention_weights'])
        avg_metadata_importance = {}
        for key in feature_analysis['metadata_feature_gradients']:
            avg_metadata_importance[key] = np.mean(feature_analysis['metadata_feature_gradients'][key])
        
        print(f"📝 Average Text Importance: {avg_text_importance:.6f}")
        print(f"📊 Metadata Feature Importance:")
        for key, importance in avg_metadata_importance.items():
            print(f"  {key}: {importance:.6f}")
        
        self.model.eval()  # Return to eval mode
        
        return {
            'average_text_importance': avg_text_importance,
            'average_metadata_importance': avg_metadata_importance,
            'raw_analysis': feature_analysis
        }
    
    def generate_comprehensive_report(self, test_data_path: str) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        print("\n🎯 Generating Comprehensive Multi-Modal Fusion Network Report")
        print("=" * 80)
        
        # Load test data
        test_samples = load_cleaned_data(test_data_path, max_samples=500)
        test_texts, test_metadata, test_labels = prepare_training_data(test_samples)
        
        # 1. Model Architecture Analysis
        architecture_info = self.analyze_model_architecture()
        
        # 2. Performance Evaluation
        performance_results = self.evaluate_on_dataset(test_texts, test_metadata, test_labels)
        
        # 3. Feature Importance Analysis
        feature_importance = self.analyze_feature_importance(test_texts, test_metadata)
        
        # 4. Multi-Modal Fusion Analysis
        fusion_analysis = self._analyze_fusion_effectiveness(test_texts, test_metadata, test_labels)
        
        # Compile comprehensive report
        report = {
            'model_type': 'Multi-Modal Fusion Network',
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'test_dataset_size': len(test_samples),
            'architecture_analysis': architecture_info,
            'performance_metrics': performance_results,
            'feature_importance': feature_importance,
            'fusion_analysis': fusion_analysis,
            'summary_insights': self._generate_summary_insights(
                architecture_info, performance_results, feature_importance, fusion_analysis
            )
        }
        
        return report
    
    def _analyze_fusion_effectiveness(self, texts: List[str], metadata_list: List[Dict], 
                                    labels_list: List[Dict]) -> Dict[str, Any]:
        """Analyze effectiveness of multi-modal fusion"""
        print("\n🔗 Analyzing Multi-Modal Fusion Effectiveness...")
        
        # Compare performance: text-only vs metadata-only vs combined
        fusion_analysis = {
            'text_only_performance': {},
            'metadata_only_performance': {},
            'combined_performance': {},
            'fusion_gain': {}
        }
        
        # This would require separate text-only and metadata-only models
        # For now, we'll analyze the contribution weights in the fusion layer
        
        sample_size = min(100, len(texts))
        sample_indices = np.random.choice(len(texts), sample_size, replace=False)
        
        text_contributions = []
        metadata_contributions = []
        
        with torch.no_grad():
            for i in sample_indices:
                text_batch = self.text_processor.process_batch([texts[i]])
                metadata_batch = self.metadata_processor.process_batch([metadata_list[i]])
                
                text_input = text_batch['embeddings'].to(self.device)
                metadata_input = {
                    'numerical_features': metadata_batch['numerical_features'].to(self.device),
                    'author_encoded': metadata_batch['author_encoded'].to(self.device),
                    'season_encoded': metadata_batch['season_encoded'].to(self.device),
                    'file_types_encoded': metadata_batch['file_types_encoded'].to(self.device)
                }
                
                # Get intermediate representations
                text_features = self.model.text_branch(text_input, None)  # No attention mask
                metadata_features = self.model.metadata_branch(metadata_input)
                
                # Analyze contribution magnitudes
                text_magnitude = torch.norm(text_features).item()
                metadata_magnitude = torch.norm(metadata_features).item()
                
                text_contributions.append(text_magnitude)
                metadata_contributions.append(metadata_magnitude)
        
        avg_text_contribution = np.mean(text_contributions)
        avg_metadata_contribution = np.mean(metadata_contributions)
        total_contribution = avg_text_contribution + avg_metadata_contribution
        
        fusion_analysis = {
            'average_text_contribution': avg_text_contribution,
            'average_metadata_contribution': avg_metadata_contribution,
            'text_contribution_ratio': avg_text_contribution / total_contribution,
            'metadata_contribution_ratio': avg_metadata_contribution / total_contribution,
            'contribution_balance': abs(0.5 - (avg_text_contribution / total_contribution)),
            'fusion_effectiveness_score': min(avg_text_contribution, avg_metadata_contribution) / max(avg_text_contribution, avg_metadata_contribution)
        }
        
        print(f"📝 Text Contribution: {avg_text_contribution:.4f} ({fusion_analysis['text_contribution_ratio']*100:.1f}%)")
        print(f"📊 Metadata Contribution: {avg_metadata_contribution:.4f} ({fusion_analysis['metadata_contribution_ratio']*100:.1f}%)")
        print(f"⚖️ Balance Score: {1 - fusion_analysis['contribution_balance']:.4f}")
        print(f"🔗 Fusion Effectiveness: {fusion_analysis['fusion_effectiveness_score']:.4f}")
        
        return fusion_analysis
    
    def _generate_summary_insights(self, architecture: Dict, performance: Dict, 
                                 feature_importance: Dict, fusion: Dict) -> Dict[str, Any]:
        """Generate high-level insights and recommendations"""
        
        # Calculate overall performance
        task_f1_scores = [metrics['f1_score'] for metrics in performance.values()]
        avg_f1 = np.mean(task_f1_scores)
        
        # Analyze model complexity
        total_params = architecture['total_parameters']
        params_per_task = total_params / len(self.task_configs)
        
        # Fusion balance analysis
        fusion_balance = 1 - fusion['contribution_balance']
        fusion_effectiveness = fusion['fusion_effectiveness_score']
        
        insights = {
            'overall_performance': {
                'average_f1_score': avg_f1,
                'performance_level': 'Excellent' if avg_f1 > 0.8 else 'Good' if avg_f1 > 0.6 else 'Needs Improvement',
                'best_performing_task': max(performance.keys(), key=lambda k: performance[k]['f1_score']),
                'worst_performing_task': min(performance.keys(), key=lambda k: performance[k]['f1_score'])
            },
            'model_efficiency': {
                'total_parameters': total_params,
                'parameters_per_task': params_per_task,
                'efficiency_score': avg_f1 / (total_params / 1000000),  # F1 per million params
                'complexity_level': 'High' if total_params > 5000000 else 'Medium' if total_params > 1000000 else 'Low'
            },
            'fusion_quality': {
                'balance_score': fusion_balance,
                'effectiveness_score': fusion_effectiveness,
                'modality_balance': 'Balanced' if fusion_balance > 0.8 else 'Moderate' if fusion_balance > 0.6 else 'Imbalanced',
                'fusion_recommendation': self._get_fusion_recommendation(fusion_balance, fusion_effectiveness)
            },
            'recommendations': self._generate_recommendations(avg_f1, total_params, fusion_balance, performance)
        }
        
        return insights
    
    def _get_fusion_recommendation(self, balance: float, effectiveness: float) -> str:
        """Generate fusion-specific recommendations"""
        if balance > 0.8 and effectiveness > 0.7:
            return "Excellent fusion - both modalities contribute effectively"
        elif balance < 0.6:
            return "Consider rebalancing modality contributions"
        elif effectiveness < 0.5:
            return "One modality dominates - consider architectural adjustments"
        else:
            return "Good fusion with room for optimization"
    
    def _generate_recommendations(self, avg_f1: float, total_params: int, 
                                balance: float, performance: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if avg_f1 < 0.7:
            recommendations.append("Consider increasing model capacity or improving data quality")
        
        if total_params > 3000000 and avg_f1 < 0.8:
            recommendations.append("Model may be overparameterized - consider pruning or distillation")
        
        if balance < 0.7:
            recommendations.append("Rebalance fusion weights or adjust modality-specific architectures")
        
        # Task-specific recommendations
        f1_scores = {task: metrics['f1_score'] for task, metrics in performance.items()}
        worst_task = min(f1_scores.keys(), key=lambda k: f1_scores[k])
        if f1_scores[worst_task] < 0.6:
            recommendations.append(f"Focus on improving {worst_task} - consider task-specific data augmentation")
        
        return recommendations

def main():
    """Main evaluation function"""
    print("🎯 Multi-Modal Fusion Network - Comprehensive Evaluation")
    print("=" * 70)
    
    # Configuration
    model_path = "trained_models/multimodal_fusion_clean_data.pth"
    test_data_path = "training_data/sample_preview.json"
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("💡 Please train the model first using train_clean_data.py")
        return
    
    # Initialize evaluator
    evaluator = MultiModalEvaluator(model_path)
    
    # Generate comprehensive report
    report = evaluator.generate_comprehensive_report(test_data_path)
    
    # Save report
    report_path = "evaluation_results/multimodal_fusion_evaluation_report.json"
    Path("evaluation_results").mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Evaluation report saved to: {report_path}")
    
    # Print summary
    print("\n📋 EVALUATION SUMMARY")
    print("=" * 50)
    insights = report['summary_insights']
    
    print(f"🎯 Overall Performance: {insights['overall_performance']['performance_level']}")
    print(f"📊 Average F1-Score: {insights['overall_performance']['average_f1_score']:.4f}")
    print(f"🏆 Best Task: {insights['overall_performance']['best_performing_task']}")
    print(f"🎯 Focus Area: {insights['overall_performance']['worst_performing_task']}")
    
    print(f"\n🏗️ Model Efficiency:")
    print(f"📐 Total Parameters: {insights['model_efficiency']['total_parameters']:,}")
    print(f"⚡ Efficiency Score: {insights['model_efficiency']['efficiency_score']:.4f}")
    print(f"🔧 Complexity: {insights['model_efficiency']['complexity_level']}")
    
    print(f"\n🔗 Fusion Quality:")
    print(f"⚖️ Balance Score: {insights['fusion_quality']['balance_score']:.4f}")
    print(f"🎯 Effectiveness: {insights['fusion_quality']['effectiveness_score']:.4f}")
    print(f"📊 Status: {insights['fusion_quality']['modality_balance']}")
    
    print(f"\n💡 Key Recommendations:")
    for i, rec in enumerate(insights['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print(f"\n✨ Next Steps:")
    print(f"1. Review detailed metrics in {report_path}")
    print(f"2. Implement recommended improvements")
    print(f"3. Compare with HAN model performance")
    print(f"4. Deploy for production testing")

if __name__ == "__main__":
    main()
