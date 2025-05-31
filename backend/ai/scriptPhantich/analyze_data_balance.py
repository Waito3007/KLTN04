import json
from collections import Counter
import pandas as pd
import os

# Đọc dữ liệu huấn luyện
file_path = "c:\\SAN\\KLTN\\KLTN04\\backend\\ai\\training_data\\han_training_samples.json"
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

samples = data['samples']

# Khởi tạo counters
purpose_counter = Counter()
sentiment_counter = Counter()
tech_tags_counter = Counter()
author_counter = Counter()
repo_counter = Counter()

# Đếm số lượng của mỗi nhãn
for sample in samples:
    purpose_counter[sample.get('purpose', 'Unknown')] += 1
    sentiment_counter[sample.get('sentiment', 'Unknown')] += 1
    
    tech_tags = sample.get('tech_tags', [])
    for tag in tech_tags:
        tech_tags_counter[tag] += 1
        
    metadata = sample.get('metadata', {})
    author_counter[metadata.get('author', 'Unknown')] += 1
    repo_counter[metadata.get('source_repo', 'Unknown')] += 1

# In phân tích
print("\n=== Phân tích cân bằng dữ liệu ===")
print(f"\nTổng số mẫu: {len(samples)}")

print("\n--- Purpose Distribution ---")
for purpose, count in purpose_counter.most_common():
    print(f"{purpose}: {count} ({count/len(samples)*100:.1f}%)")

print("\n--- Sentiment Distribution ---")
for sentiment, count in sentiment_counter.most_common():
    print(f"{sentiment}: {count} ({count/len(samples)*100:.1f}%)")

print("\n--- Top 10 Tech Tags ---")
for tag, count in tech_tags_counter.most_common(10):
    print(f"{tag}: {count}")

print("\n--- Author Distribution ---")
for author, count in author_counter.most_common():
    print(f"{author}: {count} ({count/len(samples)*100:.1f}%)")

print("\n--- Repository Distribution ---")
for repo, count in repo_counter.most_common():
    print(f"{repo}: {count} ({count/len(samples)*100:.1f}%)")

# Tính toán các chỉ số cân bằng
def calculate_imbalance(counter):
    total = sum(counter.values())
    proportions = [count/total for count in counter.values()]
    max_prop = max(proportions)
    min_prop = min(proportions)
    imbalance_ratio = max_prop / min_prop if min_prop > 0 else float('inf')
    return imbalance_ratio

print("\n=== Tỉ lệ mất cân bằng (max/min) ===")
print(f"Purpose Imbalance: {calculate_imbalance(purpose_counter):.2f}")
print(f"Sentiment Imbalance: {calculate_imbalance(sentiment_counter):.2f}")
print(f"Author Imbalance: {calculate_imbalance(author_counter):.2f}")
print(f"Repository Imbalance: {calculate_imbalance(repo_counter):.2f}")
