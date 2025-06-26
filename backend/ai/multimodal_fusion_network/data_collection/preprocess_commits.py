import json
import os
import random

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
    # Có thể thêm các bước làm sạch khác nếu cần
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
        item = {
            "text": text,
            "metadata": metadata,
            # Nếu có nhãn, thêm vào đây. Ví dụ: "label": c.get("label")
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
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_path = "../data/parallel_commits/github_commits_batch2_20250623_232635.json"  # hoặc tên file thực tế của bạn
    # input_path = "../data/parallel_commits/github_commits_batch1_20250623_225619.json"  # hoặc tên file thực tế của bạn
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)

    commits = load_commits(input_path)
    total_raw = len(commits)
    processed = extract_fields(commits)
    total_clean = len(processed)
    train, val, test = split_data(processed)

    save_json(train, f"{output_dir}/train.json")
    save_json(val, f"{output_dir}/val.json")
    save_json(test, f"{output_dir}/test.json")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Tổng số commit gốc: {total_raw}")
    print(f"Tổng số commit đã làm sạch: {total_clean}")
    print(f"Số commit bị loại bỏ: {total_raw - total_clean}")
    # Xuất file thống kê
    stats = {
        "total_raw": total_raw,
        "total_clean": total_clean,
        "removed": total_raw - total_clean,
        "train": len(train),
        "val": len(val),
        "test": len(test)
    }
    save_json(stats, f"{output_dir}/stats.json")
