#!/usr/bin/env python3
"""
Test Commit Analyzer - Phân tích và đánh giá commit với HAN model
Tính toán loại commit, thống kê tác giả, phát hiện overload/underload
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

# Import from training script
from train_han_github import SimpleHANModel, SimpleTokenizer

@dataclass
class CommitAnalysis:
    """Kết quả phân tích một commit"""
    text: str
    author: str
    predicted_labels: Dict[str, str]
    confidence_scores: Dict[str, float]
    timestamp: Optional[str] = None

@dataclass
class AuthorStats:
    """Thống kê của một tác giả"""
    name: str
    total_commits: int
    commit_types: Dict[str, int]
    purposes: Dict[str, int]
    sentiments: Dict[str, int]
    tech_tags: Dict[str, int]
    avg_confidence: float
    activity_level: str  # "low", "normal", "high", "overloaded"

class CommitAnalyzer:
    """Lớp phân tích commit với HAN model"""
    
    def __init__(self, model_path: str):
        """Khởi tạo analyzer với model đã train"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.label_encoders = None
        self.num_classes = None
        self.reverse_label_encoders = None
        
        print(f"🔧 Initializing Commit Analyzer on {self.device}")
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load model và các thành phần cần thiết"""
        print(f"📦 Loading model from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract components
        self.tokenizer = checkpoint['tokenizer']
        self.label_encoders = checkpoint['label_encoders']
        self.num_classes = checkpoint['num_classes']
        
        # Create reverse label encoders for prediction decoding
        self.reverse_label_encoders = {}
        for task, encoder in self.label_encoders.items():
            self.reverse_label_encoders[task] = {v: k for k, v in encoder.items()}
        
        # Initialize model
        vocab_size = len(self.tokenizer.word_to_idx)
        self.model = SimpleHANModel(
            vocab_size=vocab_size,
            embed_dim=100,
            hidden_dim=128,
            num_classes=self.num_classes
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   📊 Vocab size: {vocab_size}")
        print(f"   🏷️  Tasks: {list(self.num_classes.keys())}")
        print(f"   📈 Best accuracy: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
    
    def predict_commit(self, commit_text: str, author: str, timestamp: str = None) -> CommitAnalysis:
        """Dự đoán loại commit và confidence scores"""
        with torch.no_grad():
            # Tokenize input
            input_ids = self.tokenizer.encode_text(commit_text, max_sentences=10, max_words=50)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            
            # Forward pass
            outputs = self.model(input_tensor)
            
            # Decode predictions
            predicted_labels = {}
            confidence_scores = {}
            
            for task, output in outputs.items():
                # Get probabilities
                probs = torch.softmax(output, dim=1)
                confidence, pred_idx = torch.max(probs, 1)
                
                # Convert to labels
                pred_idx = pred_idx.item()
                confidence = confidence.item()
                
                predicted_label = self.reverse_label_encoders[task][pred_idx]
                
                predicted_labels[task] = predicted_label
                confidence_scores[task] = confidence
        
        return CommitAnalysis(
            text=commit_text,
            author=author,
            predicted_labels=predicted_labels,
            confidence_scores=confidence_scores,
            timestamp=timestamp
        )
    
    def analyze_commits_batch(self, commits: List[Dict]) -> List[CommitAnalysis]:
        """Phân tích nhiều commit cùng lúc"""
        print(f"🔍 Analyzing {len(commits)} commits...")
        
        results = []
        for i, commit in enumerate(commits):
            if i % 10 == 0 and i > 0:
                print(f"   Processed {i}/{len(commits)} commits")
            
            result = self.predict_commit(
                commit['text'],
                commit['author'],
                commit.get('timestamp', None)
            )
            results.append(result)
        
        print(f"✅ Completed analysis of {len(results)} commits")
        return results
    
    def generate_author_statistics(self, analyses: List[CommitAnalysis]) -> Dict[str, AuthorStats]:
        """Tạo thống kê chi tiết cho từng tác giả"""
        print("📊 Generating author statistics...")
        
        author_data = defaultdict(lambda: {
            'commits': [],
            'commit_types': Counter(),
            'purposes': Counter(),
            'sentiments': Counter(),
            'tech_tags': Counter(),
            'confidences': []
        })
        
        # Collect data by author
        for analysis in analyses:
            author = analysis.author
            author_data[author]['commits'].append(analysis)
            
            # Count labels
            if 'commit_type' in analysis.predicted_labels:
                author_data[author]['commit_types'][analysis.predicted_labels['commit_type']] += 1
            if 'purpose' in analysis.predicted_labels:
                author_data[author]['purposes'][analysis.predicted_labels['purpose']] += 1
            if 'sentiment' in analysis.predicted_labels:
                author_data[author]['sentiments'][analysis.predicted_labels['sentiment']] += 1
            if 'tech_tag' in analysis.predicted_labels:
                author_data[author]['tech_tags'][analysis.predicted_labels['tech_tag']] += 1
            
            # Collect confidence scores
            avg_confidence = np.mean(list(analysis.confidence_scores.values()))
            author_data[author]['confidences'].append(avg_confidence)
        
        # Calculate statistics and activity levels
        author_stats = {}
        total_commits = len(analyses)
        avg_commits_per_author = total_commits / len(author_data) if author_data else 0
        
        for author, data in author_data.items():
            commit_count = len(data['commits'])
            avg_confidence = np.mean(data['confidences']) if data['confidences'] else 0.0
            
            # Determine activity level
            if commit_count < avg_commits_per_author * 0.3:
                activity_level = "low"
            elif commit_count < avg_commits_per_author * 1.5:
                activity_level = "normal"
            elif commit_count < avg_commits_per_author * 3:
                activity_level = "high"
            else:
                activity_level = "overloaded"
            
            author_stats[author] = AuthorStats(
                name=author,
                total_commits=commit_count,
                commit_types=dict(data['commit_types']),
                purposes=dict(data['purposes']),
                sentiments=dict(data['sentiments']),
                tech_tags=dict(data['tech_tags']),
                avg_confidence=avg_confidence,
                activity_level=activity_level
            )
        
        return author_stats
    
    def print_analysis_report(self, analyses: List[CommitAnalysis], author_stats: Dict[str, AuthorStats]):
        """In báo cáo phân tích chi tiết"""
        print("\n" + "="*80)
        print("📈 COMMIT ANALYSIS REPORT")
        print("="*80)
        
        # Overall statistics
        total_commits = len(analyses)
        unique_authors = len(author_stats)
        
        print(f"\n📊 OVERVIEW:")
        print(f"   Total commits analyzed: {total_commits}")
        print(f"   Unique authors: {unique_authors}")
        print(f"   Average commits per author: {total_commits/unique_authors:.1f}")
        
        # Commit type distribution
        all_types = Counter()
        all_purposes = Counter()
        all_sentiments = Counter()
        all_tech_tags = Counter()
        
        for analysis in analyses:
            if 'commit_type' in analysis.predicted_labels:
                all_types[analysis.predicted_labels['commit_type']] += 1
            if 'purpose' in analysis.predicted_labels:
                all_purposes[analysis.predicted_labels['purpose']] += 1
            if 'sentiment' in analysis.predicted_labels:
                all_sentiments[analysis.predicted_labels['sentiment']] += 1
            if 'tech_tag' in analysis.predicted_labels:
                all_tech_tags[analysis.predicted_labels['tech_tag']] += 1
        
        print(f"\n🏷️  COMMIT TYPE DISTRIBUTION:")
        for commit_type, count in all_types.most_common():
            percentage = (count / total_commits) * 100
            print(f"   {commit_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n🎯 PURPOSE DISTRIBUTION:")
        for purpose, count in all_purposes.most_common():
            percentage = (count / total_commits) * 100
            print(f"   {purpose}: {count} ({percentage:.1f}%)")
        
        print(f"\n😊 SENTIMENT DISTRIBUTION:")
        for sentiment, count in all_sentiments.most_common():
            percentage = (count / total_commits) * 100
            print(f"   {sentiment}: {count} ({percentage:.1f}%)")
        
        # Author activity analysis
        activity_levels = Counter([stats.activity_level for stats in author_stats.values()])
        
        print(f"\n👥 AUTHOR ACTIVITY LEVELS:")
        for level, count in activity_levels.items():
            percentage = (count / unique_authors) * 100
            print(f"   {level.title()}: {count} authors ({percentage:.1f}%)")
        
        # Top contributors
        sorted_authors = sorted(author_stats.values(), key=lambda x: x.total_commits, reverse=True)
        
        print(f"\n🏆 TOP 10 CONTRIBUTORS:")
        for i, author in enumerate(sorted_authors[:10]):
            print(f"   {i+1:2d}. {author.name}: {author.total_commits} commits ({author.activity_level})")
        
        # Identify overloaded and underperforming authors
        overloaded = [a for a in sorted_authors if a.activity_level == "overloaded"]
        low_activity = [a for a in sorted_authors if a.activity_level == "low"]
        
        if overloaded:
            print(f"\n⚠️  OVERLOADED AUTHORS ({len(overloaded)}):")
            for author in overloaded:
                print(f"   🔥 {author.name}: {author.total_commits} commits")
                top_types = sorted(author.commit_types.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"      Most common types: {', '.join([f'{t}({c})' for t, c in top_types])}")
        
        if low_activity:
            print(f"\n📉 LOW ACTIVITY AUTHORS ({len(low_activity)}):")
            for author in low_activity[-10:]:  # Show bottom 10
                print(f"   💤 {author.name}: {author.total_commits} commits")
        
        # Confidence analysis
        all_confidences = [np.mean(list(a.confidence_scores.values())) for a in analyses]
        avg_confidence = np.mean(all_confidences)
        
        print(f"\n🎯 MODEL CONFIDENCE:")
        print(f"   Average confidence: {avg_confidence:.3f}")
        print(f"   Confidence range: {min(all_confidences):.3f} - {max(all_confidences):.3f}")
    
    def save_detailed_report(self, analyses: List[CommitAnalysis], author_stats: Dict[str, AuthorStats], output_path: str):
        """Lưu báo cáo chi tiết ra file JSON"""
        print(f"💾 Saving detailed report to {output_path}")
        
        # Prepare data for JSON serialization
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_commits": len(analyses),
                "unique_authors": len(author_stats),
                "avg_commits_per_author": len(analyses) / len(author_stats) if author_stats else 0
            },
            "commit_analyses": [],
            "author_statistics": {},
            "overall_distributions": {},
            "activity_analysis": {}
        }
        
        # Add commit analyses
        for analysis in analyses:
            report_data["commit_analyses"].append({
                "text": analysis.text,
                "author": analysis.author,
                "predicted_labels": analysis.predicted_labels,
                "confidence_scores": analysis.confidence_scores,
                "timestamp": analysis.timestamp
            })
        
        # Add author statistics
        for author_name, stats in author_stats.items():
            report_data["author_statistics"][author_name] = {
                "total_commits": stats.total_commits,
                "commit_types": stats.commit_types,
                "purposes": stats.purposes,
                "sentiments": stats.sentiments,
                "tech_tags": stats.tech_tags,
                "avg_confidence": stats.avg_confidence,
                "activity_level": stats.activity_level
            }
        
        # Calculate overall distributions
        all_types = Counter()
        all_purposes = Counter()
        all_sentiments = Counter()
        
        for analysis in analyses:
            if 'commit_type' in analysis.predicted_labels:
                all_types[analysis.predicted_labels['commit_type']] += 1
            if 'purpose' in analysis.predicted_labels:
                all_purposes[analysis.predicted_labels['purpose']] += 1
            if 'sentiment' in analysis.predicted_labels:
                all_sentiments[analysis.predicted_labels['sentiment']] += 1
        
        report_data["overall_distributions"] = {
            "commit_types": dict(all_types),
            "purposes": dict(all_purposes),
            "sentiments": dict(all_sentiments)
        }
        
        # Activity analysis
        activity_levels = Counter([stats.activity_level for stats in author_stats.values()])
        report_data["activity_analysis"] = {
            "activity_levels": dict(activity_levels),
            "overloaded_authors": [name for name, stats in author_stats.items() if stats.activity_level == "overloaded"],
            "low_activity_authors": [name for name, stats in author_stats.items() if stats.activity_level == "low"]
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Report saved successfully!")

def create_sample_commits():
    """Tạo dữ liệu commit mẫu để test"""
    sample_commits = [
        {
            "text": "fix: resolve authentication bug in login endpoint",
            "author": "John Doe",
            "timestamp": "2024-01-15T10:30:00Z"
        },
        {
            "text": "feat: add new user dashboard with analytics",
            "author": "Jane Smith",
            "timestamp": "2024-01-15T11:45:00Z"
        },
        {
            "text": "docs: update README with installation instructions",
            "author": "Bob Wilson",
            "timestamp": "2024-01-15T14:20:00Z"
        },
        {
            "text": "refactor: improve database query performance",
            "author": "John Doe",
            "timestamp": "2024-01-16T09:15:00Z"
        },
        {
            "text": "test: add unit tests for payment module",
            "author": "Alice Johnson",
            "timestamp": "2024-01-16T13:30:00Z"
        },
        {
            "text": "fix: memory leak in image processing service",
            "author": "John Doe",
            "timestamp": "2024-01-16T16:45:00Z"
        },
        {
            "text": "feat: implement real-time notifications",
            "author": "Jane Smith",
            "timestamp": "2024-01-17T08:00:00Z"
        },
        {
            "text": "chore: update dependencies to latest versions",
            "author": "Bob Wilson",
            "timestamp": "2024-01-17T10:30:00Z"
        },
        {
            "text": "style: format code according to style guide",
            "author": "Alice Johnson",
            "timestamp": "2024-01-17T15:20:00Z"
        },
        {
            "text": "feat: add dark mode support to UI components",
            "author": "Jane Smith",
            "timestamp": "2024-01-18T09:45:00Z"
        },
        {
            "text": "fix: critical security vulnerability in auth system",
            "author": "John Doe",
            "timestamp": "2024-01-18T11:00:00Z"
        },
        {
            "text": "perf: optimize database queries for better performance",
            "author": "Alice Johnson",
            "timestamp": "2024-01-18T14:15:00Z"
        },
        {
            "text": "docs: add API documentation for new endpoints",
            "author": "Bob Wilson",
            "timestamp": "2024-01-19T08:30:00Z"
        },
        {
            "text": "feat: integrate payment gateway with Stripe",
            "author": "John Doe",
            "timestamp": "2024-01-19T10:45:00Z"
        },
        {
            "text": "test: improve test coverage for core modules",
            "author": "Alice Johnson",
            "timestamp": "2024-01-19T13:20:00Z"
        },
        # Thêm commits để tạo pattern overload cho John Doe
        {
            "text": "fix: resolve timeout issues in API calls",
            "author": "John Doe",
            "timestamp": "2024-01-20T08:00:00Z"
        },
        {
            "text": "feat: add caching mechanism for better performance",
            "author": "John Doe",
            "timestamp": "2024-01-20T10:30:00Z"
        },
        {
            "text": "fix: handle edge cases in data validation",
            "author": "John Doe",
            "timestamp": "2024-01-20T14:15:00Z"
        },
        {
            "text": "refactor: clean up legacy code in user service",
            "author": "John Doe",
            "timestamp": "2024-01-20T16:45:00Z"
        },
        {
            "text": "feat: implement advanced search functionality",
            "author": "John Doe",
            "timestamp": "2024-01-21T09:20:00Z"
        },
        # Thêm ít commits cho Charlie Brown để tạo low activity
        {
            "text": "docs: fix typo in configuration guide",
            "author": "Charlie Brown",
            "timestamp": "2024-01-22T11:00:00Z"
        }
    ]
    
    return sample_commits

def main():
    """Hàm chính để test commit analyzer"""
    print("🚀 COMMIT ANALYZER TEST")
    print("="*60)
    
    # Paths
    model_path = Path(__file__).parent / "models" / "han_github_model" / "best_model.pth"
    output_dir = Path(__file__).parent / "test_results"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    try:
        analyzer = CommitAnalyzer(str(model_path))
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create or load test data
    print(f"\n📝 Preparing test data...")
    
    # Option 1: Use sample data
    commits = create_sample_commits()
    print(f"✅ Using {len(commits)} sample commits")
    
    # Option 2: Load real data (uncomment if available)
    # try:
    #     data_file = Path(__file__).parent / "training_data" / "github_commits_training_data.json"
    #     with open(data_file, 'r', encoding='utf-8') as f:
    #         real_data = json.load(f)
    #     commits = real_data['data'][:100]  # Use first 100 for testing
    #     print(f"✅ Using {len(commits)} real commits from training data")
    # except Exception as e:
    #     commits = create_sample_commits()
    #     print(f"⚠️  Could not load real data, using sample commits: {e}")
    
    # Analyze commits
    print(f"\n🔍 Starting commit analysis...")
    analyses = analyzer.analyze_commits_batch(commits)
    
    # Generate author statistics
    author_stats = analyzer.generate_author_statistics(analyses)
    
    # Print report
    analyzer.print_analysis_report(analyses, author_stats)
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"commit_analysis_report_{timestamp}.json"
    analyzer.save_detailed_report(analyses, author_stats, str(report_file))
    
    # Test individual commit prediction
    print(f"\n🧪 INDIVIDUAL COMMIT TEST:")
    print("-" * 40)
    
    test_commits = [
        "fix: critical security vulnerability in authentication",
        "feat: implement machine learning model for recommendations",
        "docs: update installation guide for new users"
    ]
    
    for i, commit_text in enumerate(test_commits, 1):
        print(f"\n{i}. Testing: '{commit_text}'")
        result = analyzer.predict_commit(commit_text, "Test User")
        
        print(f"   Predictions:")
        for task, label in result.predicted_labels.items():
            confidence = result.confidence_scores[task]
            print(f"     {task}: {label} (confidence: {confidence:.3f})")
    
    print(f"\n✅ Testing completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📄 Detailed report: {report_file}")

if __name__ == "__main__":
    main()
