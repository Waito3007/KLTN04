import json
import sys
from typing import Dict, Any
from collections import defaultdict

def build_label_maps(commits, label_keys):
    label_maps = {k: {} for k in label_keys}
    label_counters = {k: 0 for k in label_keys}
    for commit in commits:
        labels = commit.get('labels', {})
        for k in label_keys:
            v = labels.get(k)
            if isinstance(v, str) and v not in label_maps[k]:
                label_maps[k][v] = label_counters[k]
                label_counters[k] += 1
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, str) and x not in label_maps[k]:
                        label_maps[k][x] = label_counters[k]
                        label_counters[k] += 1
    return label_maps

def encode_labels(commits, label_maps):
    # Chuẩn bị số lượng class cho từng trường nhãn
    label_dims = {k: len(v) for k, v in label_maps.items()}
    for commit in commits:
        labels = commit.get('labels', {})
        for k in label_maps:
            v = labels.get(k, [])
            # Lấy các chỉ số nhãn hợp lệ
            indices = []
            if isinstance(v, str) and v in label_maps[k]:
                indices = [label_maps[k][v]]
            elif isinstance(v, list):
                indices = [label_maps[k][x] for x in v if x in label_maps[k]]
            # Tạo multi-hot vector
            multi_hot = [0] * label_dims[k]
            for idx in indices:
                if 0 <= idx < label_dims[k]:
                    multi_hot[idx] = 1
            labels[k] = multi_hot
        commit['labels'] = labels
    return commits

def auto_encode_labels(input_path, output_path, label_keys=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'data' in data:
        commits = data['data']
        meta = data.get('metadata', {})
    else:
        commits = data
        meta = {}
    # Xác định các key nhãn cần encode
    if label_keys is None:
        # Lấy tất cả key trong labels của sample đầu tiên
        for c in commits:
            if 'labels' in c:
                label_keys = list(c['labels'].keys())
                break
    # Build mapping
    label_maps = build_label_maps(commits, label_keys)
    # Encode
    commits = encode_labels(commits, label_maps)
    # Loại bỏ sample nếu tất cả nhãn đều là list rỗng hoặc chỉ chứa -1
    filtered_commits = []
    for commit in commits:
        labels = commit.get('labels', {})
        has_valid_label = False
        for v in labels.values():
            if isinstance(v, list) and any((isinstance(x, int) and x >= 0) for x in v):
                has_valid_label = True
                break
        if has_valid_label:
            filtered_commits.append(commit)
    print(f"Đã loại bỏ {len(commits) - len(filtered_commits)} sample có tất cả nhãn không hợp lệ.")
    commits = filtered_commits
    # Lưu lại file mới
    with open(output_path, 'w', encoding='utf-8') as f:
        if isinstance(data, dict) and 'data' in data:
            json.dump({'metadata': meta, 'data': commits}, f, ensure_ascii=False, indent=2)
        else:
            json.dump(commits, f, ensure_ascii=False, indent=2)
    # Lưu mapping để tham khảo
    with open(output_path + '.label_map.json', 'w', encoding='utf-8') as f:
        json.dump(label_maps, f, ensure_ascii=False, indent=2)
    print(f"Đã encode labels cho {len(commits)} commits và lưu vào {output_path}")
    print(f"Đã lưu mapping nhãn vào {output_path}.label_map.json")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Cách dùng: python auto_encode_labels.py <input_file> <output_file>")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    auto_encode_labels(input_path, output_path)
