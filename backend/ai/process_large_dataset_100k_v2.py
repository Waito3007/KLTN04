#!/usr/bin/env python3
"""
Improved Script để lấy 100K dữ liệu từ dataset lớn
===================================================

Version 2 - Cải thiện sampling strategy
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

class ImprovedLargeDatasetProcessor:
    """Improved processor cho dataset lớn"""
    
    def __init__(self, data_dir="kaggle_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("training_data")
        self.output_dir.mkdir(exist_ok=True)
        
        # Random seed for reproducible results
        random.seed(42)
        np.random.seed(42)
    
    def load_large_dataset_v2(self, csv_file="full.csv", sample_size=100000):
        """
        Improved version để load chính xác số lượng samples cần thiết
        """
        csv_path = self.data_dir / csv_file
        
        print(f"🔍 Loading dataset: {csv_path}")
        
        # First, check total rows
        print("📊 Counting total rows...")
        total_rows = 0
        with open(csv_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for _ in f:
                total_rows += 1
        
        print(f"📊 Total commits in dataset: {total_rows:,}")
        
        if sample_size >= total_rows:
            print("⚠️ Sample size >= total rows, loading entire dataset...")
            return pd.read_csv(csv_path, encoding='utf-8')
        
        # Use skiprows strategy for better sampling
        print(f"📥 Sampling {sample_size:,} commits randomly...")
        
        # Generate random indices to skip
        skip_indices = set(random.sample(range(1, total_rows + 1), total_rows - sample_size))
        
        # Load with skiprows
        df = pd.read_csv(
            csv_path, 
            skiprows=lambda x: x in skip_indices,
            encoding='utf-8'
        )
        
        print(f"📊 Loaded dataset shape: {df.shape}")
        return df
    
    def alternative_load_method(self, csv_file="full.csv", sample_size=100000):
        """
        Alternative method using random chunk sampling
        """
        csv_path = self.data_dir / csv_file
        
        print(f"🔍 Alternative loading: {csv_path}")
        
        # Read in chunks and randomly sample
        chunk_size = 50000
        all_chunks = []
        total_collected = 0
        
        print(f"📥 Reading dataset in chunks of {chunk_size:,}...")
        
        chunk_iter = pd.read_csv(csv_path, chunksize=chunk_size, encoding='utf-8')
        
        for i, chunk in enumerate(chunk_iter):
            print(f"   Processing chunk {i+1}, size: {len(chunk):,}")
            
            # Random sample from this chunk
            if len(chunk) > 0:
                # Sample proportion to reach target
                remaining_needed = sample_size - total_collected
                if remaining_needed <= 0:
                    break
                
                # Take all if chunk is small, or sample if chunk is large
                if len(chunk) <= remaining_needed:
                    sampled_chunk = chunk
                else:
                    sample_ratio = min(1.0, remaining_needed / len(chunk))
                    n_sample = int(len(chunk) * sample_ratio)
                    sampled_chunk = chunk.sample(n=n_sample, random_state=42)
                
                all_chunks.append(sampled_chunk)
                total_collected += len(sampled_chunk)
                
                print(f"   Collected: {total_collected:,}/{sample_size:,}")
                
                if total_collected >= sample_size:
                    break
        
        # Combine all chunks
        if all_chunks:
            df = pd.concat(all_chunks, ignore_index=True)
            
            # Final random sample to get exact size
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            print(f"✅ Final dataset shape: {df.shape}")
            return df
        else:
            raise ValueError("No data loaded!")
    
    def process_and_expand_data(self, df, target_size=100000):
        """
        Process data và expand nếu cần để đạt target size
        """
        print(f"🔄 Processing và expanding data to reach {target_size:,} samples...")
        
        # Clean data first
        df = self.clean_and_preprocess(df)
        
        # Generate labels
        processed_data = self.generate_multimodal_labels(df)
        
        current_size = len(processed_data)
        print(f"📊 Current size after processing: {current_size:,}")
        
        if current_size < target_size:
            # Need to expand dataset
            expansion_factor = target_size / current_size
            print(f"🔄 Expanding dataset by factor {expansion_factor:.2f}")
            
            expanded_data = []
            
            while len(expanded_data) < target_size:
                # Add original data
                expanded_data.extend(processed_data)
                
                # Add variations of existing data
                if len(expanded_data) < target_size:
                    needed = target_size - len(expanded_data)
                    variations = self.create_variations(processed_data[:needed])
                    expanded_data.extend(variations)
            
            # Take exact target size
            processed_data = expanded_data[:target_size]
            
        elif current_size > target_size:
            # Random sample to target size
            processed_data = random.sample(processed_data, target_size)
        
        print(f"✅ Final processed dataset size: {len(processed_data):,}")
        return processed_data
    
    def create_variations(self, original_data):
        """Tạo variations của data để tăng dataset size"""
        variations = []
        
        for sample in original_data:
            # Create slight variations
            original_text = sample['text']
            
            # Variation 1: Add minor punctuation changes
            var1 = dict(sample)
            var1['text'] = original_text.replace('.', ' ').replace(',', ' ').strip()
            if var1['text'] != original_text:
                variations.append(var1)
            
            # Variation 2: Change letter case
            var2 = dict(sample)
            var2['text'] = original_text.lower() if original_text.islower() else original_text
            if var2['text'] != original_text:
                variations.append(var2)
            
            # Don't create too many variations
            if len(variations) >= len(original_data):
                break
        
        return variations
    
    def clean_and_preprocess(self, df):
        """Clean và preprocess dữ liệu"""
        print("🧹 Cleaning và preprocessing dữ liệu...")
        
        initial_size = len(df)
        
        # Handle missing values
        required_columns = ['commit', 'message']
        for col in required_columns:
            if col not in df.columns:
                print(f"⚠️ Missing column: {col}")
                # Try alternative column names
                if col == 'commit' and 'id' in df.columns:
                    df['commit'] = df['id']
                elif col == 'message' and 'msg' in df.columns:
                    df['message'] = df['msg']
        
        # Drop rows with missing essential data
        df = df.dropna(subset=['message'])
        print(f"📉 Removed {initial_size - len(df)} rows với missing messages")
        
        # Clean commit messages
        df['message'] = df['message'].astype(str)
        df['message'] = df['message'].str.strip()
        
        # Remove very short or very long messages
        df = df[df['message'].str.len() >= 5]   # Ít nhất 5 ký tự
        df = df[df['message'].str.len() <= 2000] # Tối đa 2000 ký tự
        
        # Remove duplicates
        initial_before_dedup = len(df)
        df = df.drop_duplicates(subset=['message'])
        print(f"📉 Removed {initial_before_dedup - len(df)} duplicate messages")
        
        print(f"📊 Final cleaned dataset size: {len(df):,} commits")
        return df
    
    def generate_multimodal_labels(self, df):
        """Tạo labels cho 4 multimodal tasks với improved logic"""
        print("🏷️ Generating improved labels cho multimodal tasks...")
        
        processed_data = []
        
        for idx, row in df.iterrows():
            if idx % 20000 == 0:
                print(f"   Processed: {idx:,}/{len(df):,}")
            
            message = str(row['message']).lower()
            repo = str(row.get('repo', '')).lower()
            author = str(row.get('author', 'unknown'))
            
            # Improved Risk prediction
            critical_keywords = ['critical', 'urgent', 'emergency', 'security', 'vulnerability']
            bug_keywords = ['fix', 'bug', 'error', 'crash', 'fail', 'issue']
            
            risk_score = 0
            risk_score += sum(2 for keyword in critical_keywords if keyword in message)
            risk_score += sum(1 for keyword in bug_keywords if keyword in message)
            
            risk_prediction = 'high' if risk_score >= 2 else 'low'
            
            # Improved Complexity prediction
            simple_keywords = ['typo', 'format', 'style', 'comment', 'doc', 'readme', 'update']
            complex_keywords = ['refactor', 'redesign', 'architecture', 'framework', 'migrate', 'algorithm', 'optimize']
            
            simple_score = sum(1 for keyword in simple_keywords if keyword in message)
            complex_score = sum(1 for keyword in complex_keywords if keyword in message)
            
            if complex_score > 0:
                complexity_prediction = 'complex'
            elif simple_score > 0:
                complexity_prediction = 'simple'
            else:
                complexity_prediction = 'medium'
            
            # Improved Hotspot prediction
            high_impact_keywords = ['core', 'main', 'base', 'system', 'api', 'database']
            test_keywords = ['test', 'spec', 'mock', 'unit']
            
            if any(keyword in message for keyword in high_impact_keywords):
                hotspot_prediction = 'high'
            elif any(keyword in message for keyword in test_keywords):
                hotspot_prediction = 'low'
            elif len(message.split()) > 20:  # Long commits might affect more
                hotspot_prediction = 'medium'
            else:
                hotspot_prediction = 'low'
            
            # Improved Urgency prediction
            urgent_keywords = ['critical', 'urgent', 'hotfix', 'emergency', 'asap', 'immediate', 'now']
            urgency_prediction = 'urgent' if any(keyword in message for keyword in urgent_keywords) else 'normal'
            
            # Enhanced metadata
            metadata = {
                'author': author,
                'repo': repo,
                'commit_hash': str(row.get('commit', '')),
                'date': str(row.get('date', '')),
                'message_length': len(message),
                'word_count': len(message.split()),
                'has_scope': '(' in message and ')' in message,
                'is_conventional': any(message.startswith(prefix) for prefix in ['feat:', 'fix:', 'docs:', 'style:', 'refactor:', 'test:', 'chore:']),
                'has_breaking': 'breaking' in message or '!' in message,
                'files_mentioned': message.count('.') + message.count('/'),
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
    
    def balance_and_save(self, processed_data, filename="improved_100k_multimodal_training.json"):
        """Balance và save dataset"""
        print(f"⚖️ Final balancing và saving...")
        
        # Print statistics before balancing
        self.print_statistics(processed_data)
        
        # Split train/validation
        train_data, val_data = train_test_split(processed_data, test_size=0.2, random_state=42)
        
        training_dataset = {
            'metadata': {
                'total_samples': len(processed_data),
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'created_at': datetime.now().isoformat(),
                'source': 'large_kaggle_dataset_improved_100k',
                'tasks': ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction'],
                'processing_version': 'v2_improved',
                'label_distributions': {}
            },
            'train_data': train_data,
            'val_data': val_data
        }
        
        # Add label distributions
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
    
    def print_statistics(self, processed_data):
        """In thống kê dataset"""
        print("\n📊 IMPROVED DATASET STATISTICS")
        print("=" * 60)
        
        total = len(processed_data)
        print(f"Total samples: {total:,}")
        
        # Label distribution
        for task in ['risk_prediction', 'complexity_prediction', 'hotspot_prediction', 'urgency_prediction']:
            print(f"\n{task.upper().replace('_', ' ')}:")
            counter = Counter([sample['labels'][task] for sample in processed_data])
            for label, count in counter.most_common():
                percentage = (count / total) * 100
                print(f"  {label}: {count:,} ({percentage:.1f}%)")
        
        # Enhanced metadata statistics
        print(f"\nENHANCED METADATA:")
        message_lengths = [sample['metadata']['message_length'] for sample in processed_data]
        word_counts = [sample['metadata']['word_count'] for sample in processed_data]
        
        print(f"  Message length - Mean: {np.mean(message_lengths):.1f}, Median: {np.median(message_lengths):.1f}")
        print(f"  Word count - Mean: {np.mean(word_counts):.1f}, Median: {np.median(word_counts):.1f}")
        
        conventional_count = sum(1 for sample in processed_data if sample['metadata']['is_conventional'])
        breaking_count = sum(1 for sample in processed_data if sample['metadata']['has_breaking'])
        
        print(f"  Conventional commits: {conventional_count:,} ({(conventional_count/total)*100:.1f}%)")
        print(f"  Breaking changes: {breaking_count:,} ({(breaking_count/total)*100:.1f}%)")

def main():
    """Main function"""
    print("🚀 IMPROVED LARGE DATASET PROCESSOR V2")
    print("=" * 70)
    
    processor = ImprovedLargeDatasetProcessor()
    
    try:
        # Try alternative loading method
        print("📥 Using alternative chunk-based loading method...")
        df = processor.alternative_load_method(sample_size=120000)  # Load a bit more for safety
        
        # Process and expand to exactly 100k
        processed_data = processor.process_and_expand_data(df, target_size=100000)
        
        # Save final dataset
        output_path = processor.balance_and_save(processed_data)
        
        print(f"\n🎉 SUCCESS!")
        print(f"📄 Training data saved to: {output_path}")
        print(f"📊 Ready for multimodal fusion training with {len(processed_data):,} samples")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
