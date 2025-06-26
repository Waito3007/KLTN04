"""
Script xử lý dữ liệu commit từ file JSON cho mô hình multimodal fusion.
"""
import os
import json
import random
from datetime import datetime

def load_commits(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Nếu là dạng {"metadata":..., "data":[...]} thì lấy data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data

def clean_text(text):
    if not text or not isinstance(text, str):
        return None
    text = text.strip().replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())  # Xóa khoảng trắng thừa
    text = text.lower()  # (Tùy chọn) chuyển về chữ thường
    return text

def deduplicate_commits(commits):
    seen = set()
    unique = []
    for c in commits:
        key = (c.get('text'),
               c.get('metadata', {}).get('author_email'),
               c.get('metadata', {}).get('timestamp'))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique

def extract_fields(commits):
    processed = []
    for c in commits:
        text = c.get("text") or c.get("commit_message") or c.get("message")
        text = clean_text(text)
        if not text:
            continue  # Bỏ qua commit không có message
        metadata = c.get("metadata", {})
        if not metadata:
            continue  # Bỏ qua commit không có metadata
        
        # Tạo tính năng tự động
        features = {}
        for k, v in metadata.items():
            if isinstance(v, (int, float)):
                features[k] = v
            elif isinstance(v, str) and v:
                features[k] = 1  # Đơn giản hóa thành one-hot
            elif isinstance(v, list):
                features[k + "_count"] = len(v)
        
        # Tạo nhãn tự động
        labels = {}
        # Phân loại theo độ dài commit message
        words = len(text.split())
        if words < 5:
            labels["message_quality"] = 0  # Ngắn
        elif words < 15:
            labels["message_quality"] = 1  # Trung bình
        else:
            labels["message_quality"] = 2  # Dài
        
        # Phân loại theo từ khóa
        if "fix" in text or "bug" in text or "issue" in text:
            labels["commit_type"] = 0  # Fix
        elif "add" in text or "feature" in text or "implement" in text:
            labels["commit_type"] = 1  # Feature
        elif "refactor" in text or "clean" in text or "improve" in text:
            labels["commit_type"] = 2  # Refactor
        else:
            labels["commit_type"] = 3  # Khác
        
        item = {
            "text": text,
            "metadata": metadata,
            "features": features,
            "labels": labels
        }
        processed.append(item)
    
    processed = deduplicate_commits(processed)
    return processed

def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_path = "data/parallel_commits/github_commits_batch1_20250623_225619.json"
    output_dir = "output/processed"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Đang đọc dữ liệu từ {input_path}...")
    commits = load_commits(input_path)
    total_raw = len(commits)
    print(f"Đã đọc {total_raw} commits")
    
    print("Đang xử lý dữ liệu...")
    processed = extract_fields(commits)
    total_clean = len(processed)
    print(f"Đã xử lý {total_clean} commits (loại bỏ {total_raw - total_clean})")
    
    print("Đang chia tập dữ liệu...")
    train, val, test = split_data(processed)

    # Tạo metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "processed_at": timestamp,
        "input_file": input_path,
        "total_raw": total_raw,
        "total_clean": total_clean,
        "removed": total_raw - total_clean,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test)
    }
    
    # Lưu dữ liệu
    print("Đang lưu dữ liệu...")
    train_path = os.path.join(output_dir, f"train_{timestamp}.json")
    val_path = os.path.join(output_dir, f"val_{timestamp}.json")
    test_path = os.path.join(output_dir, f"test_{timestamp}.json")
    stats_path = os.path.join(output_dir, f"stats_{timestamp}.json")
    
    save_json({"metadata": metadata, "data": train}, train_path)
    save_json({"metadata": metadata, "data": val}, val_path)
    save_json({"metadata": metadata, "data": test}, test_path)
    save_json(metadata, stats_path)
    
    print(f"Đã lưu dữ liệu vào:")
    print(f"  Train: {train_path} ({len(train)} samples)")
    print(f"  Validation: {val_path} ({len(val)} samples)")
    print(f"  Test: {test_path} ({len(test)} samples)")
    print(f"  Stats: {stats_path}")
