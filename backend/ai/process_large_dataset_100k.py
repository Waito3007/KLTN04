#!/usr/bin/env python3
"""
Script để lấy 100K dữ liệu từ dataset lớn để tiếp tục training
===============================================================

Script này sẽ:
1. Load dataset lớn (4.3M commits) 
2. Lấy mẫu 100K commits ngẫu nhiên
3. Xử lý và format dữ liệu cho multimodal fusion training
4. Tạo balanced dataset với label phân phối tốt
"""

import pandas as pd
import numpy as np
import json
import random
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(__file__))

class LargeDatasetProcessor:
    """Processor cho dataset lớn"""
    
    def __init__(self, data_dir="kaggle_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("training_data")
        self.output_dir.mkdir(exist_ok=True)
        
        # Random seed for reproducible results
        random.seed(42)
        np.random.seed(42)
    
    def load_large_dataset(self, csv_file="full.csv", sample_size=100000):
        """
        Load large dataset và lấy mẫu ngẫu nhiên
        
        Args:
            csv_file: Tên file CSV
            sample_size: Số lượng samples cần lấy
        """
        csv_path = self.data_dir / csv_file
        
        print(f"🔍 Đang phân tích dataset: {csv_path}")
        
        # Đếm tổng số dòng trước
        total_rows = sum(1 for line in open(csv_path, 'r', encoding='utf-8')) - 1  # -1 for header
        print(f"📊 Tổng số commits trong dataset: {total_rows:,}")
        
        if sample_size >= total_rows:
            print(f"⚠️ Sample size ({sample_size:,}) >= total rows ({total_rows:,}), sử dụng toàn bộ dataset")
            sample_size = total_rows
        
        # Load dataset với chunks để tiết kiệm memory
        print(f"📥 Đang load {sample_size:,} samples ngẫu nhiên...")
        
        # Tính skip probability để có được số lượng samples mong muốn
        skip_prob = 1 - (sample_size / total_rows)
        
        chunks = []
        rows_collected = 0
        target_rows = sample_size
        
        for chunk in pd.read_csv(csv_path, chunksize=10000, encoding='utf-8'):
            # Random sampling trong chunk
            if skip_prob > 0:
                mask = np.random.random(len(chunk)) > skip_prob
                chunk = chunk[mask]
            
            chunks.append(chunk)
            rows_collected += len(chunk)
            
            if rows_collected >= target_rows:
                print(f"✅ Đã collect đủ {rows_collected:,} samples")
                break
        
        # Combine all chunks
        df = pd.concat(chunks, ignore_index=True)
        
        # Shuffle and take exact number if needed
        if len(df) > target_rows:
            df = df.sample(n=target_rows, random_state=42).reset_index(drop=True)
        
        print(f"📊 Dataset shape sau sampling: {df.shape}")
        return df
    
    def clean_and_preprocess(self, df):
        """Clean và preprocess dữ liệu"""
        print("🧹 Cleaning và preprocessing dữ liệu...")
        
        initial_size = len(df)
        
        # Remove missing values
        df = df.dropna(subset=['commit', 'message'])
        print(f"📉 Removed {initial_size - len(df)} rows với missing values")
        
        # Clean commit messages
        df['message'] = df['message'].astype(str)
        df['message'] = df['message'].str.strip()
        
        # Remove very short or very long messages
        df = df[df['message'].str.len() >= 10]  # Ít nhất 10 ký tự
        df = df[df['message'].str.len() <= 1000]  # Tối đa 1000 ký tự
        
        print(f"📊 Final dataset size: {len(df):,} commits")
        return df
    
    def generate_multimodal_labels(self, df):
        """Tạo labels cho 4 multimodal tasks"""
        print("🏷️ Generating labels cho multimodal tasks...")
        
        processed_data = []
        
        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                print(f"   Processed: {idx:,}/{len(df):,}")
            
            message = str(row['message']).lower()
            repo = str(row.get('repo', '')).lower()
            author = str(row.get('author', 'unknown'))
            
            # Risk prediction (high/low)
            risk_keywords = ['fix', 'bug', 'error', 'crash', 'security', 'vulnerability', 'critical', 'urgent', 'hotfix']
            risk_score = sum(1 for keyword in risk_keywords if keyword in message)
            risk_prediction = 'high' if risk_score >= 2 else 'low'
            
            # Complexity prediction (simple/medium/complex)
            complexity_indicators = {
                'simple': ['typo', 'rename', 'format', 'style', 'comment', 'doc'],
                'complex': ['refactor', 'migrate', 'redesign', 'architecture', 'framework', 'algorithm']
            }
            
            simple_count = sum(1 for keyword in complexity_indicators['simple'] if keyword in message)
            complex_count = sum(1 for keyword in complexity_indicators['complex'] if keyword in message)
            
            if complex_count > 0:
                complexity_prediction = 'complex'
            elif simple_count > 0:
                complexity_prediction = 'simple'
            else:
                complexity_prediction = 'medium'
            
            # Hotspot prediction (very_low/low/medium/high/very_high)
            file_count = message.count('.') + message.count('/')  # Proxy for file changes
            if any(keyword in message for keyword in ['core', 'main', 'base', 'fundamental']):
                hotspot_prediction = 'high'
            elif any(keyword in message for keyword in ['test', 'spec', 'mock']):
                hotspot_prediction = 'low'
            elif file_count > 3:
                hotspot_prediction = 'medium'
            else:
                hotspot_prediction = 'low'
            
            # Urgency prediction (urgent/normal)
            urgent_keywords = ['critical', 'urgent', 'hotfix', 'emergency', 'asap', 'immediate']
            urgency_prediction = 'urgent' if any(keyword in message for keyword in urgent_keywords) else 'normal'
            
            # Extract metadata
            metadata = {
                'author': author,
                'repo': repo,
                'commit_hash': str(row.get('commit', '')),
                'date': str(row.get('date', '')),
                # Derived features
                'message_length': len(message),
                'word_count': len(message.split()),
                'has_scope': '(' in message and ')' in message,
                'is_conventional': any(message.startswith(prefix) for prefix in ['feat:', 'fix:', 'docs:', 'style:', 'refactor:', 'test:', 'chore:']),
            }
            
            processed_data.append({
                'text': str(row['message']),
                'metadata': metadata,
                'labels': {
                    'risk_prediction': risk_prediction,
                    'complexity_prediction': complexity_prediction,
                    'hotspot_prediction': hotspot_prediction,
                    'urgency_prediction': urgency_prediction
                }
            })
        
        return processed_data
    
    def balance_dataset(self, processed_data, target_size=80000):
        """Balance dataset để có label distribution tốt"""
        print(f"⚖️ Balancing dataset to {target_size:,} samples...")
        
        # Group by all label combinations
        label_combinations = defaultdict(list)
        
        for sample in processed_data:
            labels = sample['labels']
            key = (
                labels['risk_prediction'],
                labels['complexity_prediction'], 
                labels['hotspot_prediction'],
                labels['urgency_prediction']
            )
            label_combinations[key].append(sample)
        
        print(f"📊 Found {len(label_combinations)} unique label combinations")
        
        # Calculate target per combination
        samples_per_combination = max(1, target_size // len(label_combinations))
        
        balanced_samples = []
        for key, samples in label_combinations.items():
            # Take up to samples_per_combination from each group
            if len(samples) >= samples_per_combination:
                selected = random.sample(samples, samples_per_combination)
            else:
                selected = samples
            
            balanced_samples.extend(selected)
        
        # If we need more samples, add randomly
        if len(balanced_samples) < target_size:
            remaining_needed = target_size - len(balanced_samples)
            all_samples = [s for samples in label_combinations.values() for s in samples]
            additional = random.sample(all_samples, min(remaining_needed, len(all_samples)))
            balanced_samples.extend(additional)
        
        # Shuffle final dataset
        random.shuffle(balanced_samples)
        
        # Take exact target size
        if len(balanced_samples) > target_size:
            balanced_samples = balanced_samples[:target_size]
        
        print(f"✅ Final balanced dataset: {len(balanced_samples):,} samples")
        return balanced_samples
    
    def print_statistics(self, processed_data):
        """In thống kê dataset"""
        print("\n📊 DATASET STATISTICS")
        print("=" * 50)
        
        total = len(processed_data)
        print(f"Total samples: {total:,}")
        
        # Label distribution
        for task in ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']:
            print(f"\n{task.upper()}:")
            counter = Counter([sample['labels'][task] for sample in processed_data])
            for label, count in counter.most_common():
                percentage = (count / total) * 100
                print(f"  {label}: {count:,} ({percentage:.1f}%)")
        
        # Metadata statistics
        print(f"\nMETADATA:")
        message_lengths = [sample['metadata']['message_length'] for sample in processed_data]
        word_counts = [sample['metadata']['word_count'] for sample in processed_data]
        
        print(f"  Message length - Mean: {np.mean(message_lengths):.1f}, Median: {np.median(message_lengths):.1f}")
        print(f"  Word count - Mean: {np.mean(word_counts):.1f}, Median: {np.median(word_counts):.1f}")
        
        conventional_count = sum(1 for sample in processed_data if sample['metadata']['is_conventional'])
        print(f"  Conventional commits: {conventional_count:,} ({(conventional_count/total)*100:.1f}%)")
    
    def save_training_data(self, processed_data, filename="large_dataset_100k_multimodal_training.json"):
        """Save processed data for training"""
        
        # Split train/validation
        train_data, val_data = train_test_split(processed_data, test_size=0.2, random_state=42)
        
        training_dataset = {
            'metadata': {
                'total_samples': len(processed_data),
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'created_at': datetime.now().isoformat(),
                'source': 'large_kaggle_dataset_100k_sample',
                'tasks': ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction'],
                'label_distributions': {}
            },
            'train_data': train_data,
            'val_data': val_data
        }
        
        # Add label distributions to metadata
        for task in ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']:
            task_labels = [sample['labels'][task] for sample in processed_data]
            training_dataset['metadata']['label_distributions'][task] = dict(Counter(task_labels))
        
        # Save to file
        output_path = self.output_dir / filename
        
        print(f"💾 Saving training data to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(processed_data):,} samples")
        print(f"   Train: {len(train_data):,}")
        print(f"   Validation: {len(val_data):,}")
        
        return output_path

def main():
    """Main function"""
    print("🚀 LARGE DATASET PROCESSOR FOR MULTIMODAL FUSION")
    print("=" * 70)
    
    processor = LargeDatasetProcessor()
    
    try:
        # 1. Load large dataset (100K samples)
        df = processor.load_large_dataset(sample_size=100000)
        
        # 2. Clean and preprocess
        df = processor.clean_and_preprocess(df)
        
        # 3. Generate multimodal labels
        processed_data = processor.generate_multimodal_labels(df)
        
        # 4. Balance dataset
        balanced_data = processor.balance_dataset(processed_data, target_size=80000)
        
        # 5. Print statistics
        processor.print_statistics(balanced_data)
        
        # 6. Save training data
        output_path = processor.save_training_data(balanced_data)
        
        print(f"\n🎉 SUCCESS!")
        print(f"📄 Training data saved to: {output_path}")
        print(f"📊 Ready for multimodal fusion training with {len(balanced_data):,} samples")
        
        # Create a summary file
        summary = {
            'status': 'SUCCESS',
            'total_samples': len(balanced_data),
            'output_file': str(output_path),
            'created_at': datetime.now().isoformat(),
            'next_steps': [
                'Run multimodal fusion training script',
                'Use the generated training data',
                'Monitor training progress and accuracy'
            ]
        }
        
        summary_path = processor.output_dir / "large_dataset_processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Processing summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
