import json
import sys
from collections import defaultdict

def encode_author(input_path, output_path, mapping_path=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'data' in data:
        samples = data['data']
        meta = data.get('metadata', {})
    else:
        samples = data
        meta = {}
    # Tạo mapping author -> id
    author_set = set()
    for s in samples:
        author = s.get('metadata', {}).get('author', None)
        if author is not None:
            author_set.add(author)
    author_list = sorted(list(author_set))
    author2id = {a: i for i, a in enumerate(author_list)}
    print(f"Tổng số author: {len(author2id)}")
    # Gán author_id vào features
    for s in samples:
        author = s.get('metadata', {}).get('author', None)
        if author is not None:
            if 'features' not in s:
                s['features'] = {}
            s['features']['author_id'] = author2id[author]
    # Lưu file mới
    if isinstance(data, dict) and 'data' in data:
        data['data'] = samples
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu dataset với author_id vào {output_path}")
    # Lưu mapping
    if mapping_path:
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(author2id, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu mapping author -> id vào {mapping_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Cách dùng: python encode_author_label.py <input_file> <output_file> [mapping_file]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    mapping_path = sys.argv[3] if len(sys.argv) > 3 else None
    encode_author(input_path, output_path, mapping_path)
