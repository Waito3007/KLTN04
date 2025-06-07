"""
Download và xử lý dataset GitHub Commit Messages từ Kaggle
Dataset: mrisdal/github-commit-messages
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import re
from collections import Counter
import zipfile

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
        print("Vui lòng chạy: python quick_kaggle_setup.py")
        return None

def download_dataset(api, dataset_name="dhruvildave/github-commit-messages-dataset", force_download=False):
    """Download dataset từ Kaggle"""
    try:
        # Tạo thư mục download
        download_dir = Path(__file__).parent / "kaggle_data" / "github_commits"
        download_dir.mkdir(parents=True, exist_ok=True)
        
        # Kiểm tra đã download chưa
        csv_files = list(download_dir.glob("*.csv"))
        if csv_files and not force_download:
            print(f"✅ Dataset đã tồn tại trong {download_dir}")
            return download_dir, csv_files[0]
        
        print(f"📥 Đang download dataset: {dataset_name}")
        print(f"📁 Vào thư mục: {download_dir}")
        
        # Download dataset
        api.dataset_download_files(
            dataset_name, 
            path=str(download_dir), 
            unzip=True
        )
        
        # Tìm file CSV
        csv_files = list(download_dir.glob("*.csv"))
        if not csv_files:
            print("❌ Không tìm thấy file CSV trong dataset")
            return None, None
        
        csv_file = csv_files[0]
        print(f"✅ Download thành công: {csv_file}")
        print(f"📊 Kích thước file: {csv_file.stat().st_size / (1024*1024):.1f} MB")
        
        return download_dir, csv_file
        
    except Exception as e:
        print(f"❌ Lỗi download dataset: {e}")
        return None, None

def analyze_commit_data(csv_file, sample_size=None):
    """Phân tích dữ liệu commit"""
    try:
        print(f"\n📊 PHÂN TÍCH DỮ LIỆU: {csv_file.name}")
        print("="*60)
        
        # Đọc dữ liệu với chunk để tiết kiệm memory
        print("📖 Đang đọc dữ liệu...")
        
        # Đọc một phần nhỏ trước để xem cấu trúc
        sample_df = pd.read_csv(csv_file, nrows=1000)
        print(f"📋 Columns: {list(sample_df.columns)}")
        print(f"📏 Sample shape: {sample_df.shape}")
        
        # Đọc toàn bộ hoặc sample
        if sample_size:
            # Đếm tổng số dòng
            total_lines = sum(1 for line in open(csv_file, 'r', encoding='utf-8'))
            skip_rows = np.random.choice(range(1, total_lines), 
                                       size=max(0, total_lines-sample_size-1), 
                                       replace=False)
            df = pd.read_csv(csv_file, skiprows=skip_rows)
            print(f"📊 Đã sample {len(df)} commits từ {total_lines} total")
        else:
            df = pd.read_csv(csv_file)
            print(f"📊 Đã đọc {len(df)} commits")
        
        # Hiển thị thông tin cơ bản
        print(f"\n📈 THỐNG KÊ CƠ BẢN:")
        print(f"  • Tổng số commits: {len(df):,}")
        print(f"  • Columns: {df.shape[1]}")
        print(f"  • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Kiểm tra columns quan trọng
        important_cols = ['message', 'commit', 'subject', 'body', 'author', 'repo']
        available_cols = [col for col in important_cols if col in df.columns]
        print(f"  • Available important columns: {available_cols}")
        
        # Tìm column chứa commit message
        message_col = None
        for col in ['message', 'subject', 'commit']:
            if col in df.columns:
                message_col = col
                break
        
        if not message_col:
            print("❌ Không tìm thấy column chứa commit message")
            return None
        
        print(f"  • Sử dụng column: '{message_col}' làm commit message")
        
        # Làm sạch dữ liệu
        print(f"\n🧹 LÀM SẠCH DỮ LIỆU:")
        original_count = len(df)
        
        # Loại bỏ messages rỗng
        df = df.dropna(subset=[message_col])
        df = df[df[message_col].str.strip() != '']
        print(f"  • Sau khi loại bỏ empty: {len(df):,} (-{original_count - len(df)})")
        
        # Loại bỏ messages quá ngắn hoặc quá dài
        df = df[df[message_col].str.len().between(5, 500)]
        print(f"  • Sau khi lọc độ dài (5-500 chars): {len(df):,}")
        
        # Loại bỏ duplicates
        df = df.drop_duplicates(subset=[message_col])
        print(f"  • Sau khi loại bỏ duplicates: {len(df):,}")
        
        # Phân tích nội dung
        print(f"\n📝 PHÂN TÍCH NỘI DUNG:")
        messages = df[message_col].astype(str)
        
        # Thống kê độ dài
        lengths = messages.str.len()
        print(f"  • Độ dài trung bình: {lengths.mean():.1f} chars")
        print(f"  • Độ dài median: {lengths.median():.1f} chars")
        print(f"  • Min/Max: {lengths.min()}/{lengths.max()} chars")
        
        # Top words
        all_words = []
        for msg in messages.head(10000):  # Sample để tránh quá tải
            words = re.findall(r'\b[a-zA-Z]+\b', msg.lower())
            all_words.extend(words)
        
        word_counts = Counter(all_words).most_common(20)
        print(f"\n🔤 TOP 20 WORDS:")
        for word, count in word_counts:
            print(f"    {word}: {count}")
        
        return df, message_col
        
    except Exception as e:
        print(f"❌ Lỗi phân tích dữ liệu: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def classify_commits(df, message_col):
    """Phân loại commits theo các tiêu chí"""
    print(f"\n🏷️  PHÂN LOẠI COMMITS:")
    print("="*60)
    
    messages = df[message_col].astype(str).str.lower()
    classifications = []
    
    for idx, message in enumerate(messages):
        if idx % 10000 == 0:
            print(f"  Đã xử lý: {idx:,}/{len(messages):,}")
        
        labels = {
            'commit_type': classify_commit_type(message),
            'purpose': classify_purpose(message),
            'sentiment': classify_sentiment(message),
            'tech_tag': classify_tech_tag(message)
        }
        classifications.append(labels)
    
    # Thống kê phân loại
    print(f"\n📊 THỐNG KÊ PHÂN LOẠI:")
    
    for category in ['commit_type', 'purpose', 'sentiment', 'tech_tag']:
        values = [c[category] for c in classifications]
        counter = Counter(values)
        print(f"\n{category.upper()}:")
        for value, count in counter.most_common(10):
            percentage = (count / len(values)) * 100
            print(f"    {value}: {count:,} ({percentage:.1f}%)")
    
    return classifications

def classify_commit_type(message):
    """Phân loại commit type"""
    patterns = {
        'feat': [r'\b(feat|feature|add|implement|new)\b'],
        'fix': [r'\b(fix|bug|error|issue|resolve|patch)\b'],
        'docs': [r'\b(doc|documentation|readme|comment)\b'],
        'style': [r'\b(style|format|lint|clean|prettier)\b'],
        'refactor': [r'\b(refactor|restructure|reorganize|cleanup)\b'],
        'test': [r'\b(test|spec|testing|coverage)\b'],
        'chore': [r'\b(chore|build|ci|cd|deploy|release|version|update|upgrade|merge)\b'],
        'perf': [r'\b(perf|performance|optimize|speed)\b'],
        'security': [r'\b(security|secure|auth|authentication|authorization)\b']
    }
    
    for commit_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            if re.search(pattern, message):
                return commit_type
    
    return 'other'

def classify_purpose(message):
    """Phân loại mục đích"""
    if re.search(r'\b(add|new|implement|create|build)\b', message):
        return 'Feature Implementation'
    elif re.search(r'\b(fix|bug|error|issue|resolve)\b', message):
        return 'Bug Fix'
    elif re.search(r'\b(refactor|restructure|cleanup|improve)\b', message):
        return 'Refactoring'
    elif re.search(r'\b(doc|documentation|readme|comment)\b', message):
        return 'Documentation Update'
    elif re.search(r'\b(test|testing|spec|coverage)\b', message):
        return 'Test Update'
    elif re.search(r'\b(security|secure|auth|vulnerability)\b', message):
        return 'Security Patch'
    elif re.search(r'\b(style|format|lint|prettier)\b', message):
        return 'Code Style/Formatting'
    elif re.search(r'\b(build|ci|cd|deploy|release)\b', message):
        return 'Build/CI/CD Script Update'
    else:
        return 'Other'

def classify_sentiment(message):
    """Phân loại sentiment"""
    positive_words = ['add', 'new', 'improve', 'enhance', 'optimize', 'better', 'clean', 'good']
    negative_words = ['fix', 'bug', 'error', 'issue', 'problem', 'fail', 'broken', 'bad']
    urgent_words = ['critical', 'urgent', 'hotfix', 'emergency', 'important', 'asap']
    
    if any(word in message for word in urgent_words):
        return 'urgent'
    elif any(word in message for word in negative_words):
        return 'negative'
    elif any(word in message for word in positive_words):
        return 'positive'
    else:
        return 'neutral'

def classify_tech_tag(message):
    """Phân loại technology tag"""
    tech_patterns = {
        'javascript': [r'\b(js|javascript|node|npm|yarn|react|vue|angular)\b'],
        'python': [r'\b(python|py|pip|django|flask|fastapi)\b'],
        'java': [r'\b(java|maven|gradle|spring)\b'],
        'css': [r'\b(css|scss|sass|style|styling)\b'],
        'html': [r'\b(html|template|markup)\b'],
        'database': [r'\b(db|database|sql|mysql|postgres|mongo)\b'],
        'api': [r'\b(api|endpoint|rest|graphql|service)\b'],
        'docker': [r'\b(docker|container|dockerfile)\b'],
        'git': [r'\b(git|merge|branch|commit|pull)\b'],
        'testing': [r'\b(test|testing|spec|unit|integration)\b'],
        'security': [r'\b(security|auth|token|password|encrypt)\b'],
        'performance': [r'\b(performance|perf|optimize|cache|speed)\b'],
        'ui': [r'\b(ui|ux|interface|design|layout|responsive)\b']
    }
    
    for tech, patterns in tech_patterns.items():
        for pattern in patterns:
            if re.search(pattern, message):
                return tech
    
    return 'general'

def save_processed_data(df, message_col, classifications, output_dir):
    """Lưu dữ liệu đã xử lý"""
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo training data format
        training_data = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            if idx >= len(classifications):
                break
                
            labels = classifications[idx]
            
            training_data.append({
                'text': row[message_col],
                'labels': labels,
                'metadata': {
                    'source': 'github-commit-messages',
                    'original_index': idx
                }
            })
        
        # Tạo metadata
        metadata = {
            'total_samples': len(training_data),
            'created_at': datetime.now().isoformat(),
            'source_dataset': 'mrisdal/github-commit-messages',
            'message_column': message_col,
            'statistics': {}
        }
        
        # Thống kê cho metadata
        for category in ['commit_type', 'purpose', 'sentiment', 'tech_tag']:
            values = [item['labels'][category] for item in training_data]
            metadata['statistics'][category] = dict(Counter(values))
        
        # Save training data
        output_file = output_dir / 'github_commits_training_data.json'
        final_data = {
            'metadata': metadata,
            'data': training_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 ĐÃ LÀU DỮ LIỆU:")
        print(f"  📁 File: {output_file}")
        print(f"  📊 Samples: {len(training_data):,}")
        print(f"  📏 Size: {output_file.stat().st_size / (1024*1024):.1f} MB")
        
        # Lưu sample để preview
        sample_file = output_dir / 'sample_preview.json'
        sample_data = {
            'metadata': metadata,
            'data': training_data[:100]  # 100 samples đầu
        }
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        print(f"  🔍 Sample: {sample_file}")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Lỗi lưu dữ liệu: {e}")
        return None

def main():
    """Main function"""
    print("🚀 GITHUB COMMIT MESSAGES DOWNLOADER")
    print("="*60)
    
    # Setup Kaggle API
    api = setup_kaggle_api()
    if not api:
        return
    
    # Download dataset
    download_dir, csv_file = download_dataset(api)
    if not csv_file:
        return
    
    # Hỏi user về sample size
    print(f"\n📊 TÙY CHỌN PROCESSING:")
    print("1. Xử lý toàn bộ dataset (có thể mất nhiều thời gian)")
    print("2. Sample 50K commits (khuyên dùng)")
    print("3. Sample 10K commits (nhanh)")
    print("4. Sample 1K commits (test)")
    
    choice = input("Chọn option (1-4): ").strip()
    
    sample_sizes = {
        '1': None,
        '2': 50000,
        '3': 10000,
        '4': 1000
    }
    
    sample_size = sample_sizes.get(choice, 10000)
    
    # Analyze data
    df, message_col = analyze_commit_data(csv_file, sample_size)
    if df is None:
        return
    
    # Classify commits
    classifications = classify_commits(df, message_col)
    
    # Save processed data
    output_dir = Path(__file__).parent / "training_data"
    output_file = save_processed_data(df, message_col, classifications, output_dir)
    
    if output_file:
        print(f"\n🎉 HOÀN THÀNH!")
        print(f"📋 Bây giờ bạn có thể:")
        print(f"  • Train HAN: python train_han_with_kaggle.py")
        print(f"  • Train XGBoost: python train_xgboost.py")
        print(f"  📁 Data file: {output_file}")

if __name__ == "__main__":
    main()
