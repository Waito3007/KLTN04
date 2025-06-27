import json
import os
import random

def split_dataset(input_file, out_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    commits = data['data']
    random.seed(seed)
    random.shuffle(commits)
    n = len(commits)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = commits[:n_train]
    val = commits[n_train:n_train+n_val]
    test = commits[n_train+n_val:]
    os.makedirs(out_dir, exist_ok=True)
    for name, subset in zip(['train', 'val', 'test'], [train, val, test]):
        out_path = os.path.join(out_dir, f'{name}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'data': subset, 'metadata': {'total_samples': len(subset)}}, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu {name}: {len(subset)} samples -> {out_path}")
    print(f"Tổng số mẫu: {n} | train: {len(train)} | val: {len(val)} | test: {len(test)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Chia lại train/val/test từ file tổng hợp.')
    parser.add_argument('--input', type=str, default='data/all_labeled_commits.json', help='File tổng hợp JSON')
    parser.add_argument('--out_dir', type=str, default='data/processed', help='Thư mục lưu train/val/test')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    split_dataset(args.input, args.out_dir, args.train_ratio, args.val_ratio, args.seed)
