import sys
import os
import torch
import json
from datetime import datetime
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hierarchical_attention import HierarchicalAttentionNetwork
from data_preprocessing.text_processor import TextProcessor
from data_preprocessing.embedding_loader import EmbeddingLoader

class HANTester:
    def __init__(self, model_path, device=None):
        """Initialize the HAN tester with a trained model"""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load the trained model
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.author_map = self.checkpoint['author_map']
        self.repo_map = self.checkpoint['repo_map']
        self.num_classes_dict = self.checkpoint['num_classes_dict']
        
        # Initialize text processor and embedding loader
        self.processor = TextProcessor()
        self.embed_loader = EmbeddingLoader(embedding_type='codebert')
        self.embed_loader.load()
        
        # Initialize model
        if self.embed_loader.embedding_type == 'codebert':
            embed_dim = self.embed_loader.model.config.hidden_size
        else:
            embed_dim = 768
            
        self.model = HierarchicalAttentionNetwork(
            embed_dim=embed_dim,
            hidden_dim=128,
            num_classes_dict=self.num_classes_dict
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create reverse mappings for interpretation
        self.reverse_author_map = {v: k for k, v in self.author_map.items()}
        self.reverse_repo_map = {v: k for k, v in self.repo_map.items()}
        
        # Define commit type mapping
        self.commit_type_map = {
            0: "feature",
            1: "bugfix",
            2: "documentation",
            3: "refactor",
            4: "test",
            5: "chore",
            6: "style",
            7: "other"
        }

    def process_commit(self, text_commit, author_name):
        """Process a single commit and return predictions"""        # Process text using process_document
        processed_text = self.processor.process_document(text_commit)
          # Get embeddings for the processed document
        with torch.no_grad():
            embeddings = self.embed_loader.get_embeddings_for_doc(processed_text)
            embeddings = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(self.device)
              # Get predictions - model returns (outputs, word_attn, sent_attn)
            outputs, _, _ = self.model(embeddings)
            
            # Convert predictions to labels
            predictions = {}
            if isinstance(outputs, dict):
                for task, output in outputs.items():
                    pred = torch.argmax(output, dim=1).item()
                    if task == 'commit_type':
                        predictions[task] = self.commit_type_map[pred]
                    elif task == 'author':
                        predictions[task] = self.reverse_author_map.get(pred, "Unknown")
                    elif task == 'source_repo':
                        predictions[task] = self.reverse_repo_map.get(pred, "Unknown")
                    else:
                        predictions[task] = pred
                    
        return predictions

    def analyze_author_commits(self, commits_data):
        """Analyze multiple commits and generate statistics per author"""
        author_stats = {}
        
        for commit in commits_data:
            text_commit = commit['text_commit']
            author_name = commit['author_name']
            
            predictions = self.process_commit(text_commit, author_name)
            
            if author_name not in author_stats:
                author_stats[author_name] = {
                    'total_commits': 0,
                    'commit_types': Counter(),
                    'predictions': []
                }
            
            author_stats[author_name]['total_commits'] += 1
            author_stats[author_name]['commit_types'][predictions['commit_type']] += 1
            author_stats[author_name]['predictions'].append({
                'text': text_commit,
                'predicted_type': predictions['commit_type'],
                'suspicious': bool(predictions['suspicious']),
                'sentiment': predictions['sentiment']
            })
            
        return author_stats

    def generate_report(self, author_stats, output_file=None):
        """Generate a detailed report of the analysis"""
        if output_file is None:
            output_file = f"commit_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        report = {
            'analysis_date': datetime.now().isoformat(),
            'total_authors': len(author_stats),
            'authors': {}
        }
        
        for author, stats in author_stats.items():
            report['authors'][author] = {
                'total_commits': stats['total_commits'],
                'commit_type_distribution': dict(stats['commit_types']),
                'commit_samples': stats['predictions'][:5],  # Include first 5 commits as samples
                'summary': {
                    'most_common_type': stats['commit_types'].most_common(1)[0][0],
                    'average_commits': stats['total_commits']
                }
            }
            
        # Save report to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        return report

def main():
    # Example usage
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "han_multitask_final.pth")  # Using the trained model file
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    tester = HANTester(model_path)
    
    # Real commit data from training set
    test_commits = [
        {
            "text_commit": "align docker build and maven build output version\n\nmaven use the hardcoded version `8.7.0` and output zip file with `8.7.0` in file name suffix. This breaks COPY command",
            "author_name": "Tran Tien Duc"
        },
        {
            "text_commit": "Merge pull request #141 from duydo/dependabot/maven/com.google.guava-guava-32.0.0-jre\n\nBump guava from 31.1-jre to 32.0.0-jre",
            "author_name": "Duy Do"
        },
        {
            "text_commit": "Refactor Unsafe, removed unused code",
            "author_name": "Duy Do"
        },
        {
            "text_commit": "Fix critical security vulnerability in authentication module\n\nUrgent patch for SQL injection vulnerability CVE-2023-1234",
            "author_name": "Security Team"
        },
        {
            "text_commit": "Add comprehensive unit tests for user service\n\nIncreased test coverage to 85% with new integration tests",
            "author_name": "QA Engineer"
        },
        {
            "text_commit": "Update documentation for API endpoints\n\nImproved swagger docs and added examples",
            "author_name": "Technical Writer"
        },
        {
            "text_commit": "Refactor user authentication flow\n\nSplit monolithic auth service into smaller modules",
            "author_name": "Tran Tien Duc"
        },
        {
            "text_commit": "[ci skip] Fix typo in README.md\n\nFixed spelling mistakes in documentation",
            "author_name": "Duy Do"
        },
        {
            "text_commit": "feat: implement new payment gateway\n\nAdded Stripe integration with webhook support",
            "author_name": "Payment Team"
        },
        {
            "text_commit": "chore: update dependencies\n\nBump spring-boot from 2.7.0 to 3.0.0",
            "author_name": "Dependabot"
        }
    ]
    
    try:
        # Analyze commits
        print("Analyzing commits...")
        author_stats = tester.analyze_author_commits(test_commits)
        
        # Generate and save report
        print("Generating report...")
        report = tester.generate_report(author_stats)
        print("Analysis complete. Report generated successfully.")
        
        # Print summary
        print("\nAnalysis Summary:")
        for author, stats in report['authors'].items():
            print(f"\nAuthor: {author}")
            print(f"Total commits: {stats['total_commits']}")
            print("Commit type distribution:")
            for commit_type, count in stats['commit_type_distribution'].items():
                print(f"  - {commit_type}: {count}")
            print(f"Most common commit type: {stats['summary']['most_common_type']}")
            print("\nCommit samples:")
            for commit in stats['commit_samples'][:2]:  # Show first 2 samples
                print(f"  Message: {commit['text'][:100]}...")
                print(f"  Type: {commit['predicted_type']}")
                print(f"  Suspicious: {commit['suspicious']}")
                print(f"  Sentiment: {commit['sentiment']}")
                print()
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()