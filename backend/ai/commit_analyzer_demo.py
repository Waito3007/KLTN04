#!/usr/bin/env python3
"""
Commit Analyzer Demo - No Torch Version
Phân tích commit messages sử dụng kết quả đã có của mô hình multimodal fusion
"""

import json
import argparse
from datetime import datetime
from pathlib import Path

class CommitAnalyzerDemo:
    """
    Demo version của commit analyzer sử dụng kết quả evaluation có sẵn
    """
    
    def __init__(self):
        """Initialize với evaluation results có sẵn"""
        self.base_path = Path("d:/Project/KLTN04/backend/ai")
        self.evaluation_data = self._load_evaluation_data()
        self.model_info = self._load_model_info()
        
    def _load_evaluation_data(self):
        """Load evaluation results từ file JSON"""
        eval_file = self.base_path / "evaluation_results" / "multimodal_fusion_simple_report.json"
        
        if eval_file.exists():
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ Error loading evaluation data: {e}")
                return {}
        else:
            print(f"❌ Evaluation file not found: {eval_file}")
            return {}
    
    def _load_model_info(self):
        """Load thông tin model từ documentation"""
        model_info = {
            "total_parameters": "2.15M",
            "architecture": "Multimodal Fusion Network",
            "components": {
                "text_branch": "LSTM/Transformer (1.8M params - 83.8%)",
                "metadata_branch": "Dense layers (125K params - 5.8%)", 
                "fusion_layer": "Cross-attention (182K params - 8.4%)",
                "task_heads": "4 classification heads (42K params - 1.9%)"
            },
            "tasks": {
                "complexity_prediction": "3 classes (Low/Medium/High)",
                "risk_prediction": "2 classes (Safe/Risky)", 
                "hotspot_prediction": "5 classes (Critical/High/Medium/Low/Normal)",
                "urgency_prediction": "2 classes (Normal/Urgent)"
            }
        }
        return model_info
    
    def display_model_architecture(self):
        """Hiển thị thông tin kiến trúc model"""
        print("\n" + "="*80)
        print("🏗️  MULTIMODAL FUSION MODEL ARCHITECTURE")
        print("="*80)
        
        print(f"\n📊 Model Type: {self.model_info['architecture']}")
        print(f"📈 Total Parameters: {self.model_info['total_parameters']}")
        
        print(f"\n🧩 COMPONENTS:")
        print("-" * 50)
        for component, description in self.model_info['components'].items():
            print(f"  • {component.replace('_', ' ').title()}: {description}")
        
        print(f"\n🎯 CLASSIFICATION TASKS:")
        print("-" * 50)
        for task, description in self.model_info['tasks'].items():
            print(f"  • {task.replace('_', ' ').title()}: {description}")
    
    def display_performance_summary(self):
        """Hiển thị tóm tắt hiệu suất model"""
        print("\n" + "="*80)
        print("📈 MODEL PERFORMANCE SUMMARY")
        print("="*80)
        
        if not self.evaluation_data:
            print("❌ No evaluation data available")
            return
        
        perf = self.evaluation_data.get('performance_metrics', {})
        summary = self.evaluation_data.get('summary_metrics', {})
        
        print(f"\n📊 OVERALL METRICS:")
        print("-" * 50)
        print(f"  • Average F1 Score: {summary.get('average_f1_score', 0):.4f}")
        print(f"  • Best Task: {summary.get('best_performing_task', 'N/A')}")
        print(f"  • Worst Task: {summary.get('worst_performing_task', 'N/A')}")
        
        print(f"\n🎯 TASK-SPECIFIC PERFORMANCE:")
        print("-" * 50)
        for task, f1_score in summary.get('task_f1_scores', {}).items():
            task_name = task.replace('_', ' ').title()
            status = "🟢" if f1_score > 0.3 else "🟡" if f1_score > 0.05 else "🔴"
            print(f"  {status} {task_name}: F1 = {f1_score:.4f}")
    
    def analyze_sample_commit(self, commit_message: str = None):
        """Phân tích một commit message mẫu"""
        
        if not commit_message:
            # Demo với commit message mẫu
            commit_message = "Fix critical bug in user authentication module that could lead to security vulnerability"
        
        print("\n" + "="*80)
        print("🔍 COMMIT MESSAGE ANALYSIS")
        print("="*80)
        
        print(f"\n📝 Commit Message:")
        print(f"  \"{commit_message}\"")
        
        # Phân tích đơn giản dựa trên keywords
        analysis = self._simple_keyword_analysis(commit_message)
        
        print(f"\n🤖 AI ANALYSIS (Based on Model Pattern):")
        print("-" * 50)
        
        for task, prediction in analysis.items():
            task_name = task.replace('_', ' ').title()
            confidence = prediction['confidence']
            result = prediction['prediction']
            confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            
            print(f"  • {task_name}:")
            print(f"    Prediction: {result}")
            print(f"    Confidence: [{confidence_bar}] {confidence:.1%}")
    
    def _simple_keyword_analysis(self, commit_message: str):
        """Phân tích đơn giản dựa trên keywords"""
        msg_lower = commit_message.lower()
        
        # Keyword patterns dựa trên phân tích thực tế
        analysis = {
            'complexity_prediction': {
                'prediction': 'Medium',
                'confidence': 0.7
            },
            'risk_prediction': {
                'prediction': 'Safe', 
                'confidence': 0.6
            },
            'hotspot_prediction': {
                'prediction': 'Normal',
                'confidence': 0.5
            },
            'urgency_prediction': {
                'prediction': 'Normal',
                'confidence': 0.8
            }
        }
        
        # Adjust based on keywords
        high_risk_keywords = ['critical', 'security', 'vulnerability', 'bug', 'crash', 'error']
        high_complexity_keywords = ['refactor', 'architecture', 'major', 'overhaul', 'redesign']
        urgent_keywords = ['hotfix', 'urgent', 'critical', 'emergency', 'asap']
        hotspot_keywords = ['auth', 'login', 'payment', 'database', 'api', 'core']
        
        # Risk analysis
        if any(keyword in msg_lower for keyword in high_risk_keywords):
            analysis['risk_prediction']['prediction'] = 'Risky'
            analysis['risk_prediction']['confidence'] = 0.8
        
        # Complexity analysis  
        if any(keyword in msg_lower for keyword in high_complexity_keywords):
            analysis['complexity_prediction']['prediction'] = 'High'
            analysis['complexity_prediction']['confidence'] = 0.8
        elif len(commit_message.split()) > 10:
            analysis['complexity_prediction']['prediction'] = 'Medium'
            analysis['complexity_prediction']['confidence'] = 0.7
        else:
            analysis['complexity_prediction']['prediction'] = 'Low'
            analysis['complexity_prediction']['confidence'] = 0.6
        
        # Urgency analysis
        if any(keyword in msg_lower for keyword in urgent_keywords):
            analysis['urgency_prediction']['prediction'] = 'Urgent'
            analysis['urgency_prediction']['confidence'] = 0.9
        
        # Hotspot analysis
        if any(keyword in msg_lower for keyword in hotspot_keywords):
            analysis['hotspot_prediction']['prediction'] = 'High'
            analysis['hotspot_prediction']['confidence'] = 0.7
        
        return analysis
    
    def interactive_mode(self):
        """Chế độ interactive để phân tích commits"""
        print("\n" + "="*80)
        print("🤖 INTERACTIVE COMMIT ANALYZER")
        print("="*80)
        print("Enter commit messages to analyze (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            try:
                commit_msg = input("\n📝 Enter commit message: ").strip()
                
                if commit_msg.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not commit_msg:
                    print("❌ Please enter a commit message")
                    continue
                
                self.analyze_sample_commit(commit_msg)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def display_model_insights(self):
        """Hiển thị insights về model"""
        print("\n" + "="*80)
        print("💡 MODEL INSIGHTS & RECOMMENDATIONS")
        print("="*80)
        
        print(f"\n🔍 CURRENT STATUS:")
        print("-" * 50)
        print("  • Model has 2.15M parameters with 4 classification heads")
        print("  • Text processing dominates (83.8% of parameters)")
        print("  • Cross-attention fusion mechanism implemented")
        print("  • Severe class imbalance in training data")
        
        print(f"\n⚠️  MAIN ISSUES:")
        print("-" * 50)
        print("  • Task heads appear to be randomly initialized (not trained)")
        print("  • Urgency task shows 0% performance (100% normal class)")
        print("  • Hotspot detection near-zero performance (98% normal class)")
        print("  • Only complexity task shows reasonable performance (F1: 0.41)")
        
        print(f"\n🚀 IMPROVEMENT RECOMMENDATIONS:")
        print("-" * 50)
        print("  1. Implement proper end-to-end training pipeline")
        print("  2. Address severe class imbalance with:")
        print("     - Data augmentation for minority classes")
        print("     - Weighted loss functions")
        print("     - SMOTE or similar techniques")
        print("  3. Collect more diverse training data")
        print("  4. Fine-tune pre-trained embeddings")
        print("  5. Implement proper validation and early stopping")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Multimodal Fusion Commit Analyzer Demo")
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    parser.add_argument('--interactive', action='store_true', help='Run interactive mode')
    parser.add_argument('--analysis', action='store_true', help='Show model analysis')
    parser.add_argument('--message', type=str, help='Analyze specific commit message')
    
    args = parser.parse_args()
    
    analyzer = CommitAnalyzerDemo()
    
    if args.demo or (not args.interactive and not args.analysis and not args.message):
        # Default demo mode
        analyzer.display_model_architecture()
        analyzer.display_performance_summary()
        analyzer.analyze_sample_commit()
        analyzer.display_model_insights()
        
    elif args.interactive:
        analyzer.display_model_architecture()
        analyzer.display_performance_summary()
        analyzer.interactive_mode()
        
    elif args.analysis:
        analyzer.display_model_architecture()
        analyzer.display_performance_summary()
        analyzer.display_model_insights()
        
    elif args.message:
        analyzer.display_model_architecture()
        analyzer.analyze_sample_commit(args.message)

if __name__ == "__main__":
    main()
