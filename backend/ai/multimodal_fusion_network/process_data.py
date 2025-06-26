"""
Script xử lý dữ liệu commit từ file JSON.
"""
import os
import json
import random
from datetime import datetime

def load_commits(json_path):
    """Đọc dữ liệu commit từ file JSON."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Lấy phần data nếu có
    if isinstance(data, dict) and 'data' in data:
        return data['data']
    return data

def clean_text(text):
    """Làm sạch văn bản."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip().lower()
    text = ' '.join(text.split())  # Xóa khoảng trắng thừa
    return text

def process_commit(commit):
    """Xử lý một commit."""
    # Lấy text (commit message)
    text = None
    if 'message' in commit:
        text = clean_text(commit['message'])
    elif 'commit_message' in commit:
        text = clean_text(commit['commit_message'])
    elif 'text' in commit:
        text = clean_text(commit['text'])
    
    if not text:
        return None  # Bỏ qua commit không có message
    
    # Lấy metadata
    metadata = commit.get('metadata', {})
    if not metadata:
        # Tạo metadata từ các trường khác nếu có
        metadata = {}
        for key, value in commit.items():
            if key not in ['message', 'commit_message', 'text']:
                metadata[key] = value
    
    # Tạo features
    features = {}
    # Chuyển đổi metadata thành features phù hợp
    for key, value in metadata.items():
        if isinstance(value, (int, float)):
            features[key] = value
        elif isinstance(value, str):
            features[key] = 1  # One-hot
        elif isinstance(value, list):
            features[key + '_count'] = len(value)
    
    # Gán nhãn tự động (ví dụ)
    labels = {}
    # Nhãn dựa trên độ dài message
    if len(text.split()) < 5:
        labels['message_quality'] = 0  # Thấp
    elif len(text.split()) < 15:
        labels['message_quality'] = 1  # Trung bình
    else:
        labels['message_quality'] = 2  # Cao
    
    # Nhãn dựa trên từ khóa trong message
    if 'fix' in text or 'bug' in text or 'issue' in text:
        labels['commit_type'] = 0  # Fix
    elif 'add' in text or 'feature' in text or 'implement' in text:
        labels['commit_type'] = 1  # Feature
    elif 'update' in text or 'improve' in text or 'refactor' in text:
        labels['commit_type'] = 2  # Refactor
    else:
        labels['commit_type'] = 3  # Khác
    
    return {
        'text': text,
        'features': features,
        'labels': labels
    }

def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Chia tập dữ liệu."""
    random.seed(seed)
    random.shuffle(data)
    
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train = data[:n_train]
    val = data[n_train:n_train+n_val]
    test = data[n_train+n_val:]
    
    return train, val, test

def save_json(data, path):
    """Lưu dữ liệu vào file JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    # Đường dẫn đến file dữ liệu thô
    input_file = "data/parallel_commits/github_commits_batch1_20250623_225619.json"
    
    # Thư mục đầu ra
    output_dir = "output/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Đọc dữ liệu
    print(f"Đang đọc dữ liệu từ {input_file}...")
    commits = load_commits(input_file)
    total_raw = len(commits)
    print(f"Đã đọc {total_raw} commits")
    
    # Xử lý dữ liệu
    print("Đang xử lý dữ liệu...")
    processed = []
    for commit in commits:
        result = process_commit(commit)
        if result:
            processed.append(result)
    
    total_processed = len(processed)
    print(f"Đã xử lý {total_processed} commits (loại bỏ {total_raw - total_processed})")
    
    # Chia tập dữ liệu
    print("Đang chia tập dữ liệu...")
    train, val, test = split_data(processed)
    
    # Tạo metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        'processed_at': timestamp,
        'input_file': input_file,
        'total_raw': total_raw,
        'total_processed': total_processed,
        'removed': total_raw - total_processed,
        'train_size': len(train),
        'val_size': len(val),
        'test_size': len(test)
    }
    
    # Lưu dữ liệu
    print("Đang lưu dữ liệu...")
    train_path = os.path.join(output_dir, f"train_{timestamp}.json")
    val_path = os.path.join(output_dir, f"val_{timestamp}.json")
    test_path = os.path.join(output_dir, f"test_{timestamp}.json")
    stats_path = os.path.join(output_dir, f"stats_{timestamp}.json")
    
    save_json({'metadata': metadata, 'data': train}, train_path)
    save_json({'metadata': metadata, 'data': val}, val_path)
    save_json({'metadata': metadata, 'data': test}, test_path)
    save_json(metadata, stats_path)
    
    print(f"Đã lưu dữ liệu vào:")
    print(f"  Train: {train_path} ({len(train)} samples)")
    print(f"  Validation: {val_path} ({len(val)} samples)")
    print(f"  Test: {test_path} ({len(test)} samples)")
    print(f"  Stats: {stats_path}")

if __name__ == "__main__":
    main()
