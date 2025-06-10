#!/usr/bin/env python3
"""
GitHub Data Processor - Download và Clean dữ liệu cho Multi-Modal Fusion Network
=============================================================================
Script này sẽ:
1. Download dataset GitHub commits từ Kaggle
2. Clean và chuẩn hóa dữ liệu 
3. Tạo labels phù hợp cho multi-task learning
4. Xuất ra format chuẩn cho training
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
import traceback
from typing import Dict, List, Tuple, Any
import random

# Import để tạo synthetic metadata
from multimodal_fusion.data.synthetic_generator import GitHubDataGenerator

def setup_kaggle_api():
    """Setup Kaggle API"""
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"❌ Lỗi setup Kaggle API: {e}")
        print("💡 Vui lòng chạy: pip install kaggle")
        print("💡 Hoặc setup API key theo hướng dẫn: https://github.com/Kaggle/kaggle-api")
        return None

def download_github_dataset(api, force_download=False):
    """Download dataset GitHub commits từ Kaggle"""
    try:
        # Tạo thư mục download
        download_dir = Path(__file__).parent / "kaggle_data" / "github_commits"
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Kiểm tra đã download chưa
        csv_files = list(download_dir.glob("*.csv"))
        if csv_files and not force_download:
            print(f"✅ Dataset đã tồn tại: {csv_files[0]}")
            return csv_files[0]
        
        print("📥 Đang download GitHub commit dataset từ Kaggle...")
        print("📁 Dataset: dhruvildave/github-commit-messages-dataset")
        
        # Download dataset
        api.dataset_download_files(
            "dhruvildave/github-commit-messages-dataset", 
            path=str(download_dir), 
            unzip=True
        )
        
        # Tìm file CSV
        csv_files = list(download_dir.glob("*.csv"))
        if not csv_files:
            print("❌ Không tìm thấy file CSV trong dataset")
            return None
        
        csv_file = csv_files[0]
        print(f"✅ Download thành công: {csv_file}")
        print(f"📊 Kích thước file: {csv_file.stat().st_size / (1024*1024):.1f} MB")
        
        return csv_file
        
    except Exception as e:
        print(f"❌ Lỗi download dataset: {e}")
        return None

def clean_github_data(csv_file: Path, sample_size: int = 20000) -> pd.DataFrame:
    """Clean và chuẩn hóa dữ liệu GitHub commits"""
    try:
        print(f"\n📊 CLEANING DỮ LIỆU: {csv_file.name}")
        print("="*60)
        
        # Đọc dữ liệu với chunk để tiết kiệm memory
        print("📖 Đang đọc dữ liệu...")
        
        # Đọc sample để xem cấu trúc
        sample_df = pd.read_csv(csv_file, nrows=1000)
        print(f"📋 Columns: {list(sample_df.columns)}")
        
        # Tìm column chứa commit message
        message_col = None
        for col in ['message', 'subject', 'commit', 'commit_message']:
            if col in sample_df.columns:
                message_col = col
                break
        
        if not message_col:
            print("❌ Không tìm thấy column chứa commit message")
            return None
        
        print(f"💬 Sử dụng column: '{message_col}' làm commit message")
        
        # Đọc dữ liệu với sampling hiệu quả
        print(f"🎯 Sampling {sample_size:,} records...")
        
        # Đọc theo chunk và sample
        chunk_size = 10000
        sampled_chunks = []
        total_read = 0
        
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            total_read += len(chunk)
            
            # Random sample từ chunk
            if len(chunk) > 0:
                chunk_sample_size = min(sample_size // 20, len(chunk))
                if chunk_sample_size > 0:
                    chunk_sample = chunk.sample(n=chunk_sample_size)
                    sampled_chunks.append(chunk_sample)
            
            # Dừng khi đủ data
            total_sampled = sum(len(c) for c in sampled_chunks)
            if total_sampled >= sample_size:
                break
                
            if total_read % 50000 == 0:
                print(f"  Đã đọc: {total_read:,} records...")
        
        # Combine chunks
        df = pd.concat(sampled_chunks, ignore_index=True)
        if len(df) > sample_size:
            df = df.sample(n=sample_size).reset_index(drop=True)
        
        print(f"📊 Đã sample {len(df):,} commits từ {total_read:,} total")
        
        # BƯỚC 1: Làm sạch dữ liệu cơ bản
        print(f"\n🧹 BƯỚC 1: LÀM SẠCH CƠ BẢN")
        original_count = len(df)
        
        # Loại bỏ messages rỗng
        df = df.dropna(subset=[message_col])
        df = df[df[message_col].str.strip() != '']
        print(f"  • Sau khi loại bỏ empty: {len(df):,} (-{original_count - len(df)})")
        
        # Loại bỏ messages quá ngắn hoặc quá dài
        df = df[df[message_col].str.len().between(3, 200)]
        print(f"  • Sau khi lọc độ dài (3-200 chars): {len(df):,}")
        
        # Loại bỏ duplicates
        df = df.drop_duplicates(subset=[message_col])
        print(f"  • Sau khi loại bỏ duplicates: {len(df):,}")
        
        # BƯỚC 2: Clean text content
        print(f"\n🔤 BƯỚC 2: CLEAN TEXT CONTENT")
        
        def clean_commit_message(text):
            """Clean commit message text"""
            if pd.isna(text):
                return ""
                
            text = str(text).strip()
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://\S+', '[URL]', text)
            
            # Remove email addresses
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
            
            # Remove excessive punctuation
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            text = re.sub(r'[.]{3,}', '...', text)
            
            # Remove special characters but keep useful ones
            text = re.sub(r'[^\w\s\-_.,;:!?()\[\]{}#@/\\+=<>|~`]', '', text)
            
            return text.strip()
        
        df[message_col] = df[message_col].apply(clean_commit_message)
        
        # Remove messages that became empty after cleaning
        df = df[df[message_col].str.len() >= 3]
        print(f"  • Sau khi clean text: {len(df):,} commits")
        
        # BƯỚC 3: Phân loại và tạo labels
        print(f"\n🏷️  BƯỚC 3: TẠO LABELS CHO MULTI-TASK LEARNING")
        
        # Classify commit types
        df['commit_type'] = df[message_col].apply(classify_commit_type)
        df['purpose'] = df[message_col].apply(classify_purpose)
        df['sentiment'] = df[message_col].apply(classify_sentiment)
        df['tech_tag'] = df[message_col].apply(classify_tech_tag)
        
        # Thống kê labels
        print(f"\n📊 THỐNG KÊ LABELS:")
        for col in ['commit_type', 'purpose', 'sentiment', 'tech_tag']:
            value_counts = df[col].value_counts()
            print(f"  {col}:")
            for val, count in value_counts.head(5).items():
                print(f"    {val}: {count} ({count/len(df)*100:.1f}%)")
        
        # BƯỚC 4: Tạo metadata synthetic
        print(f"\n⚙️ BƯỚC 4: TẠO METADATA SYNTHETIC")
        generator = GitHubDataGenerator()
          def create_synthetic_metadata():
            """Tạo metadata synthetic cho mỗi commit"""
            sample = generator.generate_single_commit()
            return {
                'author': sample['author'],
                'repository': sample['repository'], 
                'timestamp': sample['timestamp'],
                'files_changed': sample['files_changed'],
                'additions': sample['additions'],
                'deletions': sample['deletions'],
                'file_types': sample['file_types']
            }
        
        # Apply synthetic metadata
        metadata_list = [create_synthetic_metadata() for _ in range(len(df))]
        
        for key in ['author', 'repository', 'timestamp', 'files_changed', 'additions', 'deletions']:
            df[f'meta_{key}'] = [meta[key] for meta in metadata_list]
        
        # File types cần xử lý đặc biệt vì là list
        df['meta_file_types'] = [meta['file_types'] for meta in metadata_list]
        
        print(f"✅ Đã tạo synthetic metadata cho {len(df):,} commits")
        
        return df
        
    except Exception as e:
        print(f"❌ Lỗi clean dữ liệu: {e}")
        traceback.print_exc()
        return None

def classify_commit_type(message: str) -> str:
    """Phân loại commit type dựa theo conventional commits"""
    message = message.lower()
    
    patterns = {
        'feat': [r'\b(feat|feature|add|implement|new|create)\b'],
        'fix': [r'\b(fix|bug|error|issue|resolve|patch|repair)\b'],
        'docs': [r'\b(doc|documentation|readme|comment|guide)\b'],
        'style': [r'\b(style|format|lint|clean|prettier|cosmetic)\b'],
        'refactor': [r'\b(refactor|restructure|reorganize|cleanup|improve)\b'],
        'test': [r'\b(test|spec|testing|coverage|unit|integration)\b'],
        'chore': [r'\b(chore|build|ci|cd|deploy|release|version|update|upgrade|merge|maint)\b'],
        'perf': [r'\b(perf|performance|optimize|speed|fast|slow)\b'],
        'security': [r'\b(security|secure|auth|authentication|authorization|vulnerability)\b']
    }
    
    for commit_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, message):
                return commit_type
    
    return 'other'

def classify_purpose(message: str) -> str:
    """Phân loại mục đích của commit"""
    message = message.lower()
    
    if re.search(r'\b(add|new|implement|create|build|introduce)\b', message):
        return 'Feature Implementation'
    elif re.search(r'\b(fix|bug|error|issue|resolve|repair)\b', message):
        return 'Bug Fix'
    elif re.search(r'\b(refactor|restructure|cleanup|improve|optimize)\b', message):
        return 'Refactoring'
    elif re.search(r'\b(doc|documentation|readme|comment|guide)\b', message):
        return 'Documentation Update'
    elif re.search(r'\b(test|testing|spec|coverage|unit)\b', message):
        return 'Test Update'
    elif re.search(r'\b(security|secure|auth|vulnerability|exploit)\b', message):
        return 'Security Patch'
    elif re.search(r'\b(style|format|lint|prettier|cosmetic)\b', message):
        return 'Code Style/Formatting'
    elif re.search(r'\b(build|ci|cd|deploy|release|version)\b', message):
        return 'Build/CI/CD Script Update'
    else:
        return 'Other'

def classify_sentiment(message: str) -> str:
    """Phân loại sentiment của commit"""
    message = message.lower()
    
    urgent_words = ['critical', 'urgent', 'hotfix', 'emergency', 'important', 'asap', 'breaking']
    negative_words = ['fix', 'bug', 'error', 'issue', 'problem', 'fail', 'broken', 'crash', 'wrong']
    positive_words = ['add', 'new', 'improve', 'enhance', 'optimize', 'better', 'clean', 'good', 'success']
    
    if any(word in message for word in urgent_words):
        return 'urgent'
    elif any(word in message for word in negative_words):
        return 'negative'
    elif any(word in message for word in positive_words):
        return 'positive'
    else:
        return 'neutral'

def classify_tech_tag(message: str) -> str:
    """Phân loại technology tag"""
    message = message.lower()
    
    tech_patterns = {
        'javascript': [r'\b(js|javascript|node|npm|yarn|react|vue|angular|typescript|ts)\b'],
        'python': [r'\b(python|py|pip|django|flask|fastapi|pandas|numpy)\b'],
        'java': [r'\b(java|maven|gradle|spring|junit)\b'],
        'css': [r'\b(css|scss|sass|style|styling|bootstrap)\b'],
        'html': [r'\b(html|template|markup|dom)\b'],
        'database': [r'\b(db|database|sql|mysql|postgres|mongo|redis|sqlite)\b'],
        'api': [r'\b(api|endpoint|rest|graphql|service|http|request)\b'],
        'docker': [r'\b(docker|container|dockerfile|kubernetes|k8s)\b'],
        'git': [r'\b(git|merge|branch|commit|pull|push|clone)\b'],
        'testing': [r'\b(test|testing|spec|unit|integration|e2e|pytest|jest)\b'],
        'security': [r'\b(security|auth|token|password|encrypt|decrypt|ssl|tls)\b'],
        'performance': [r'\b(performance|perf|optimize|cache|speed|memory|cpu)\b'],
        'ui': [r'\b(ui|ux|interface|design|layout|responsive|mobile)\b']
    }
    
    for tech, patterns in tech_patterns.items():
        for pattern in patterns:
            if re.search(pattern, message):
                return tech
    
    return 'general'

def convert_to_training_format(df: pd.DataFrame, message_col: str) -> List[Dict]:
    """Convert cleaned DataFrame thành format cho training"""
    print(f"\n🔄 CONVERT SANG TRAINING FORMAT")
    
    training_samples = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(df)} samples")
            
        # Tạo sample theo format chuẩn
        sample = {
            'commit_message': row[message_col],
            'author': row['meta_author'],
            'repository': row['meta_repository'],
            'timestamp': row['meta_timestamp'],
            'files_changed': row['meta_files_changed'],
            'additions': row['meta_additions'],
            'deletions': row['meta_deletions'],
            'file_types': row['meta_file_types'],
            'labels': {
                'risk_prediction': classify_risk_level(row[message_col], row['commit_type']),
                'complexity_prediction': classify_complexity(row[message_col], row['meta_files_changed']),
                'hotspot_prediction': classify_hotspot(row['commit_type'], row['tech_tag']),
                'urgency_prediction': classify_urgency(row['sentiment'])
            }
        }
        
        training_samples.append(sample)
    
    print(f"✅ Converted {len(training_samples)} samples")
    return training_samples

def classify_risk_level(message: str, commit_type: str) -> int:
    """Classify risk level (0: low, 1: high)"""
    high_risk_patterns = ['breaking', 'major', 'critical', 'breaking change', 'api change']
    high_risk_types = ['feat', 'refactor', 'security']
    
    message_lower = message.lower()
    
    if any(pattern in message_lower for pattern in high_risk_patterns):
        return 1
    elif commit_type in high_risk_types:
        return 1
    else:
        return 0

def classify_complexity(message: str, files_changed: int) -> int:
    """Classify complexity (0: low, 1: medium, 2: high)"""
    if files_changed >= 10:
        return 2
    elif files_changed >= 5:
        return 1
    else:
        return 0

def classify_hotspot(commit_type: str, tech_tag: str) -> int:
    """Classify hotspot area (0-4)"""
    hotspot_map = {
        'security': 0,
        'api': 1,
        'database': 2,
        'ui': 3,
        'general': 4
    }
    return hotspot_map.get(tech_tag, 4)

def classify_urgency(sentiment: str) -> int:
    """Classify urgency (0: normal, 1: urgent)"""
    return 1 if sentiment == 'urgent' else 0

def save_training_data(samples: List[Dict], output_dir: Path) -> str:
    """Lưu training data đã clean"""
    output_dir.mkdir(exist_ok=True)
    
    # Tạo filename với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"cleaned_github_commits_{timestamp}.json"
    
    # Tạo metadata
    training_data = {
        'metadata': {
            'total_samples': len(samples),
            'created_at': datetime.now().isoformat(),
            'source': 'kaggle_github_commits_cleaned',
            'version': '1.0',
            'description': 'Cleaned GitHub commit data for Multi-Modal Fusion Network'
        },
        'samples': samples
    }
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Đã lưu {len(samples)} samples vào: {output_file}")
    print(f"📊 File size: {output_file.stat().st_size / (1024*1024):.1f} MB")
    
    return str(output_file)

def main():
    """Main function"""
    print("🚀 GITHUB DATA PROCESSOR - CLEAN DỮ LIỆU CHO TRAINING")
    print("="*70)
    
    # Setup Kaggle API
    api = setup_kaggle_api()
    if not api:
        print("\n❌ Không thể setup Kaggle API")
        print("💡 Bạn có thể manually tải file CSV và đặt vào thư mục kaggle_data/github_commits/")
        
        # Kiểm tra file manual
        manual_files = list(Path("kaggle_data/github_commits").glob("*.csv"))
        if manual_files:
            csv_file = manual_files[0]
            print(f"✅ Tìm thấy file manual: {csv_file}")
        else:
            print("❌ Không tìm thấy file CSV nào")
            return
    else:
        # Download dataset
        csv_file = download_github_dataset(api)
        if not csv_file:
            print("❌ Không thể download dataset")
            return
    
    # Hỏi user về sample size
    print(f"\n📊 TÙY CHỌN SAMPLE SIZE:")
    print("1. 5K samples (test nhanh)")
    print("2. 10K samples (demo)")
    print("3. 20K samples (khuyên dùng)")
    print("4. 50K samples (training tốt)")
    print("5. 100K samples (dataset lớn)")
    
    choice = input("Chọn option (1-5) [mặc định: 3]: ").strip() or "3"
    
    sample_sizes = {
        '1': 5000,
        '2': 10000,
        '3': 20000,
        '4': 50000,
        '5': 100000
    }
    
    sample_size = sample_sizes.get(choice, 20000)
    print(f"🎯 Sẽ sample {sample_size:,} commits")
    
    # Clean data
    df = clean_github_data(csv_file, sample_size)
    if df is None:
        print("❌ Không thể clean dữ liệu")
        return
    
    # Convert to training format
    message_col = 'message'  # hoặc tìm tự động
    for col in ['message', 'subject', 'commit', 'commit_message']:
        if col in df.columns:
            message_col = col
            break
    
    training_samples = convert_to_training_format(df, message_col)
    
    # Save cleaned data
    output_dir = Path("training_data")
    output_file = save_training_data(training_samples, output_dir)
    
    print(f"\n🎉 HOÀN THÀNH CLEANING DỮ LIỆU!")
    print(f"📁 File output: {output_file}")
    print(f"📊 Số samples: {len(training_samples):,}")
    
    print(f"\n✨ SẴN SÀNG CHO TRAINING:")
    print(f"  python train_real_data.py")
    print(f"  # Hoặc sử dụng file: {Path(output_file).name}")

if __name__ == "__main__":
    main()
