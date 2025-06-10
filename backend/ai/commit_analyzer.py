#!/usr/bin/env python3
"""
MultiModal Fusion Commit Analyzer
=================================
Script để sử dụng mô hình MultiModal Fusion phân tích commit messages
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'multimodal_fusion'))

# Import multimodal fusion components
from multimodal_fusion.data_preprocessing.text_processor import TextProcessor
from multimodal_fusion.data_preprocessing.metadata_processor import MetadataProcessor
from multimodal_fusion.models.multimodal_fusion import MultiModalFusionNetwork

# Import standalone multimodal analyzer
try:
    from multimodal_commit_inference import MultimodalCommitAnalyzer
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    print("⚠️ Multimodal fusion standalone analyzer not available. Using neural network model.")

class CommitAnalyzer:
    """
    Multimodal Fusion Commit Analyzer
    Phân tích commit messages sử dụng mô hình multimodal fusion
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize commit analyzer
        
        Args:
            model_path: Path to trained model checkpoint
        """
        if model_path is None:
            model_path = "trained_models/multimodal_fusion_clean_data.pth"
        
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Task mappings
        self.task_mappings = {
            'complexity': {0: 'Low', 1: 'Medium', 2: 'High'},
            'risk': {0: 'Low Risk', 1: 'High Risk'},
            'hotspot': {0: 'Security', 1: 'API', 2: 'Database', 3: 'UI', 4: 'General'},
            'urgency': {0: 'Normal', 1: 'Urgent'}
        }
        
        self.model = None
        self.text_processor = None
        self.metadata_processor = None
        self.task_configs = None
        
        print(f"🤖 MultiModal Fusion Commit Analyzer initialized")
        print(f"📱 Device: {self.device}")
    
    def load_model(self):
        """Load trained model and processors"""
        try:
            print(f"\n🔧 Loading model from: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                print("📝 Available model files:")
                model_dir = os.path.dirname(self.model_path) or "."
                for file in os.listdir(model_dir):
                    if file.endswith('.pth'):
                        print(f"   - {file}")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract components
            self.text_processor = checkpoint['text_processor']
            self.metadata_processor = checkpoint['metadata_processor']
            self.task_configs = checkpoint['task_configs']
            metadata_dims = checkpoint['metadata_dims']
            
            # Initialize model
            self.model = MultiModalFusionNetwork(
                text_dim=self.text_processor.embed_dim,
                **metadata_dims,
                hidden_dim=256,
                dropout_rate=0.3,
                task_configs=self.task_configs
            ).to(self.device)
            
            # Load weights
            missing_keys, _ = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if missing_keys:
                print(f"⚠️  Warning: {len(missing_keys)} keys missing (likely untrained task heads)")
            
            self.model.eval()
            
            print(f"✅ Model loaded successfully!")
            print(f"📊 Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"🎯 Available tasks: {list(self.task_configs.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def analyze_single_commit(self, 
                             commit_text: str, 
                             author: str = "unknown",
                             files_changed: int = 1,
                             insertions: int = 10,
                             deletions: int = 5,
                             file_extensions: List[str] = None) -> Dict[str, Any]:
        """
        Analyze a single commit
        
        Args:
            commit_text: Commit message text
            author: Author name
            files_changed: Number of files changed
            insertions: Number of insertions
            deletions: Number of deletions
            file_extensions: List of file extensions (e.g., ['.py', '.js'])
        
        Returns:
            Dict containing predictions and confidence scores
        """
        if self.model is None:
            if not self.load_model():
                return {"error": "Failed to load model"}
        
        try:
            # Prepare metadata
            if file_extensions is None:
                file_extensions = ['.py']  # Default
            
            # Create metadata dict
            metadata = {
                'author': author,
                'files_changed': files_changed,
                'insertions': insertions,
                'deletions': deletions,
                'file_extensions': file_extensions,
                'timestamp': datetime.now().isoformat()
            }
            
            # Process text
            text_result = self.text_processor.process_batch([commit_text])
            text_input = text_result['embeddings'].to(self.device)
            
            # Process metadata
            metadata_result = self.metadata_processor.process_batch([metadata])
            metadata_input = {
                k: v.to(self.device) for k, v in metadata_result.items()
            }
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(text_input, metadata_input)
            
            # Process predictions
            predictions = {}
            confidences = {}
            
            for task, logits in outputs.items():
                if task in self.task_mappings:
                    # Get probabilities
                    probs = torch.softmax(logits, dim=1)
                    confidence, pred_idx = torch.max(probs, 1)
                    
                    pred_idx = pred_idx.cpu().item()
                    confidence = confidence.cpu().item()
                    
                    # Map to label
                    predicted_label = self.task_mappings[task].get(pred_idx, f"Unknown_{pred_idx}")
                    
                    predictions[task] = predicted_label
                    confidences[task] = float(confidence)
            
            result = {
                'commit_text': commit_text,
                'metadata': metadata,
                'predictions': predictions,
                'confidences': confidences,
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'total_parameters': sum(p.numel() for p in self.model.parameters()),
                    'device': str(self.device),
                    'tasks': list(self.task_configs.keys())
                }
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def analyze_multiple_commits(self, commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple commits
        
        Args:
            commits: List of commit dicts with keys: text, author, files_changed, etc.
        
        Returns:
            List of analysis results
        """
        results = []
        
        print(f"\n🔍 Analyzing {len(commits)} commits...")
        
        for i, commit in enumerate(commits):
            print(f"📝 Processing commit {i+1}/{len(commits)}: {commit.get('text', '')[:50]}...")
            
            result = self.analyze_single_commit(
                commit_text=commit.get('text', ''),
                author=commit.get('author', 'unknown'),
                files_changed=commit.get('files_changed', 1),
                insertions=commit.get('insertions', 10),
                deletions=commit.get('deletions', 5),
                file_extensions=commit.get('file_extensions', ['.py'])
            )
            
            results.append(result)
        
        return results
    
    def analyze_commit_multimodal(self, commit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced commit analysis using multimodal fusion
        
        Args:
            commit_data: Dict with commit information
        
        Returns:
            Analysis result with multimodal predictions
        """
        if MULTIMODAL_AVAILABLE:
            try:
                # Initialize multimodal analyzer if not already done
                if not hasattr(self, '_multimodal_analyzer'):
                    self._multimodal_analyzer = MultimodalCommitAnalyzer()
                
                # Extract commit text and metadata
                commit_text = commit_data.get('message', '') or commit_data.get('text', '')
                
                metadata = {
                    'author': commit_data.get('author', 'unknown'),
                    'files_changed': commit_data.get('files_changed', []),
                    'additions': commit_data.get('additions', 0) or commit_data.get('insertions', 0),
                    'deletions': commit_data.get('deletions', 0),
                    'timestamp': commit_data.get('timestamp', datetime.now().isoformat())
                }
                
                # Perform multimodal analysis
                result = self._multimodal_analyzer.analyze_commit(commit_text, metadata)
                
                # Convert to dict and add model type
                analysis_result = result.to_dict()
                analysis_result['model_used'] = 'multimodal_fusion'
                analysis_result['fallback_used'] = not result.success
                
                print(f"🤖 Multimodal analysis completed - Success: {result.success}")
                return analysis_result
                
            except Exception as e:
                print(f"⚠️ Multimodal analysis failed: {e}")
                # Fall back to basic analysis
                return self._fallback_analysis(commit_data)
        else:
            print("⚠️ Multimodal analyzer not available, using basic analysis")
            return self._fallback_analysis(commit_data)
    
    def _fallback_analysis(self, commit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis using existing neural network model"""
        commit_text = commit_data.get('message', '') or commit_data.get('text', '')
        
        # Use existing single commit analysis
        result = self.analyze_single_commit(
            commit_text=commit_text,
            author=commit_data.get('author', 'unknown'),
            files_changed=len(commit_data.get('files_changed', [])),
            insertions=commit_data.get('additions', 0) or commit_data.get('insertions', 0),
            deletions=commit_data.get('deletions', 0),
            file_extensions=commit_data.get('file_extensions', ['.py'])
        )
        
        # Add metadata to indicate fallback was used
        result['model_used'] = 'neural_network_fallback'
        result['fallback_used'] = True
        
        return result
    
    def print_analysis_result(self, result: Dict[str, Any]):
        """Pretty print analysis result"""
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"\n📝 Commit Analysis Result")
        print(f"{'='*50}")
        print(f"💬 Text: {result['commit_text']}")
        print(f"👤 Author: {result['metadata']['author']}")
        print(f"📊 Files changed: {result['metadata']['files_changed']}")
        print(f"➕ Insertions: {result['metadata']['insertions']}")
        print(f"➖ Deletions: {result['metadata']['deletions']}")
        
        print(f"\n🎯 Predictions:")
        for task, prediction in result['predictions'].items():
            confidence = result['confidences'][task]
            confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            print(f"  {task.capitalize()}: {prediction} ({confidence:.2%}) {confidence_bar}")
        
        print(f"\n⏰ Analysis time: {result['timestamp']}")
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str = None):
        """Save analysis results to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"commit_analysis_results_{timestamp}.json"
        
        output_path = Path("test_results") / output_file
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path}")
        return output_path

def demo_analysis():
    """Demo function showing how to use the analyzer"""
    print("🚀 MultiModal Fusion Commit Analyzer Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = CommitAnalyzer()
    
    # Example commits
    example_commits = [
        {
            'text': "fix: resolve critical authentication bug in login endpoint",
            'author': "john_doe",
            'files_changed': 3,
            'insertions': 15,
            'deletions': 8,
            'file_extensions': ['.py', '.js']
        },
        {
            'text': "feat: implement user dashboard with real-time analytics",
            'author': "jane_smith",
            'files_changed': 8,
            'insertions': 150,
            'deletions': 20,
            'file_extensions': ['.py', '.html', '.css', '.js']
        },
        {
            'text': "docs: update API documentation for new endpoints",
            'author': "bob_wilson",
            'files_changed': 2,
            'insertions': 45,
            'deletions': 5,
            'file_extensions': ['.md', '.rst']
        },
        {
            'text': "refactor: optimize database queries for better performance",
            'author': "alice_johnson",
            'files_changed': 5,
            'insertions': 25,
            'deletions': 40,
            'file_extensions': ['.py', '.sql']
        }
    ]
    
    # Analyze commits
    results = analyzer.analyze_multiple_commits(example_commits)
    
    # Print results
    for result in results:
        analyzer.print_analysis_result(result)
    
    # Save results
    output_file = analyzer.save_results(results)
    
    print(f"\n✅ Demo completed! Results saved to {output_file}")

def interactive_analysis():
    """Interactive mode for analyzing commits"""
    print("🤖 Interactive Commit Analysis Mode")
    print("Enter commit information (press Ctrl+C to exit)")
    print("=" * 50)
    
    analyzer = CommitAnalyzer()
    
    try:
        while True:
            print("\n📝 Enter commit details:")
            
            commit_text = input("Commit message: ").strip()
            if not commit_text:
                continue
            
            author = input("Author (default: unknown): ").strip() or "unknown"
            
            try:
                files_changed = int(input("Files changed (default: 1): ") or "1")
                insertions = int(input("Insertions (default: 10): ") or "10")
                deletions = int(input("Deletions (default: 5): ") or "5")
            except ValueError:
                files_changed, insertions, deletions = 1, 10, 5
            
            file_ext_input = input("File extensions (e.g., .py,.js): ").strip()
            file_extensions = [ext.strip() for ext in file_ext_input.split(',')] if file_ext_input else ['.py']
            
            # Analyze
            result = analyzer.analyze_single_commit(
                commit_text=commit_text,
                author=author,
                files_changed=files_changed,
                insertions=insertions,
                deletions=deletions,
                file_extensions=file_extensions
            )
            
            # Show result
            analyzer.print_analysis_result(result)
            
            # Ask if want to save
            save_choice = input("\n💾 Save this result? (y/n): ").strip().lower()
            if save_choice == 'y':
                analyzer.save_results([result])
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MultiModal Fusion Commit Analyzer")
    parser.add_argument("--mode", choices=["demo", "interactive"], default="demo",
                       help="Mode to run: demo or interactive")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "demo":
            demo_analysis()
        elif args.mode == "interactive":
            interactive_analysis()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n📋 Troubleshooting:")
        print("1. Make sure model checkpoint exists")
        print("2. Check if all dependencies are installed")
        print("3. Verify CUDA availability if using GPU")
