import json
import sys
from typing import Dict, Any

def extract_features(commit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hàm sinh features đơn giản từ commit. Có thể mở rộng thêm logic tại đây.
    """
    features = {}
    # Số lượng ký tự, số từ trong message
    text = commit.get('text') or commit.get('commit_message') or commit.get('message')
    if text:
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
    # Metadata (nếu có)
    metadata = commit.get('metadata', {})
    if isinstance(metadata, dict):
        for k, v in metadata.items():
            if isinstance(v, (int, float)):
                features[k] = v
            elif isinstance(v, str):
                features[k] = 1  # one-hot đơn giản cho string, sẽ ghi đè bởi author_id nếu là author
            elif isinstance(v, list):
                features[k + '_count'] = len(v)
    return features

def add_features_and_encode_author(input_path: str, output_path: str, mapping_path: str = None):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Xác định key chứa danh sách commit
    if isinstance(data, dict) and 'data' in data:
        commits = data['data']
        meta = data.get('metadata', {})
    else:
        commits = data
        meta = {}

    # Tạo mapping author -> author_id từ metadata
    author_set = set()
    for commit in commits:
        author = commit.get('metadata', {}).get('author', None)
        if author is not None:
            author_set.add(author)
    author2id = {author: idx for idx, author in enumerate(sorted(author_set))}

    # Thêm features và author_id cho từng commit
    for commit in commits:
        features = extract_features(commit)
        author = commit.get('metadata', {}).get('author', None)
        if author is not None:
            features['author_id'] = author2id[author]
        else:
            features['author_id'] = -1
        commit['features'] = features

    # Lưu lại file mới
    with open(output_path, 'w', encoding='utf-8') as f:
        if isinstance(data, dict) and 'data' in data:
            json.dump({'metadata': meta, 'data': commits}, f, ensure_ascii=False, indent=2)
        else:
            json.dump(commits, f, ensure_ascii=False, indent=2)
    print(f"Đã thêm features và author_id cho {len(commits)} commits và lưu vào {output_path}")

    # Lưu mapping nếu cần
    if mapping_path:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(author2id, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu mapping author → author_id vào {mapping_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Cách dùng: python add_features_and_encode_author.py <input_file> <output_file> [mapping_file]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    mapping_path = sys.argv[3] if len(sys.argv) > 3 else None
    add_features_and_encode_author(input_path, output_path, mapping_path)
