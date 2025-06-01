import json
import random
from collections import Counter, defaultdict

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def balance_dataset(data, min_samples_per_class=25):
    samples = data['samples']
    # Group samples by different attributes
    by_purpose = defaultdict(list)
    by_sentiment = defaultdict(list)
    by_author = defaultdict(list)
    by_repo = defaultdict(list)
    
    # First pass - collect statistics and group samples
    for sample in samples:
        purpose = sample['purpose']
        sentiment = sample['sentiment']
        author = sample['metadata']['author']
        repo_id = sample['metadata']['source_repo']
        
        by_purpose[purpose].append(sample)
        by_sentiment[sentiment].append(sample)
        by_author[author].append(sample)
        by_repo[repo_id].append(sample)
      # Balance purposes first
    balanced_samples = []
    purposes = ['Feature Implementation', 'Bug Fix', 'Refactoring', 'Other']
    
    # Find minimum samples across purposes that have data
    min_purpose = float('inf')
    for purpose in purposes:
        if by_purpose[purpose]:  # Only consider non-empty categories
            min_purpose = min(min_purpose, len(by_purpose[purpose]))
    
    # Set target size per purpose, ensuring at least min_samples_per_class
    target_size = max(min_purpose, min_samples_per_class)
    
    for purpose in purposes:
        available = by_purpose[purpose]
        if not available:
            continue
            
        if len(available) < target_size:
            # If we don't have enough samples, use bootstrapping
            needed = target_size - len(available)
            selected = available + [random.choice(available) for _ in range(needed)]
        else:
            # Sample randomly
            selected = random.sample(available, target_size)
        balanced_samples.extend(selected)
      # Now balance sentiments while preserving purpose balance
    sentiments = ['positive', 'negative', 'neutral']
    sentiment_grouped = defaultdict(list)
    for sample in balanced_samples:
        sentiment = sample['sentiment']
        sentiment_grouped[sentiment].append(sample)
    
    # Find average sentiment count for non-empty categories
    total_samples = 0
    non_empty_cats = 0
    for sentiment in sentiments:
        if sentiment_grouped[sentiment]:
            total_samples += len(sentiment_grouped[sentiment])
            non_empty_cats += 1
    
    target_sentiment = max(total_samples // non_empty_cats, min_samples_per_class//2)
    
    final_samples = []
    for sentiment in sentiments:
        available = sentiment_grouped[sentiment]
        if len(available) < target_sentiment:
            final_samples.extend(available)
        else:
            final_samples.extend(random.sample(available, target_sentiment))
    
    # Shuffle final samples
    random.shuffle(final_samples)
    return final_samples

def main():
    # Load original and existing balanced data
    original_data = load_json('c:/SAN/KLTN/KLTN04/backend/ai/training_data/han_training_samples.json')
    
    # Balance the dataset
    balanced_samples = balance_dataset(original_data)
    
    # Save balanced dataset
    balanced_data = {'samples': balanced_samples}
    save_json(balanced_data, 'c:/SAN/KLTN/KLTN04/backend/ai/training_data/han_training_samples_balanced.json')
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(balanced_data['samples'])}")
    
    print("\nPurpose Distribution:")
    purposes = Counter(s['purpose'] for s in balanced_data['samples'])
    for purpose, count in purposes.most_common():
        print(f"{purpose}: {count} ({count/len(balanced_data['samples'])*100:.1f}%)")
    
    print("\nSentiment Distribution:")
    sentiments = Counter(s['sentiment'] for s in balanced_data['samples'])
    for sentiment, count in sentiments.most_common():
        print(f"{sentiment}: {count} ({count/len(balanced_data['samples'])*100:.1f}%)")
    
    print("\nTop Authors:")
    authors = Counter(s['metadata']['author'] for s in balanced_data['samples'])
    for author, count in authors.most_common(5):
        print(f"{author}: {count} ({count/len(balanced_data['samples'])*100:.1f}%)")
    
    print("\nRepository Distribution:")
    repos = Counter(s['metadata']['source_repo'] for s in balanced_data['samples'])
    for repo, count in repos.most_common():
        print(f"Repo {repo}: {count} ({count/len(balanced_data['samples'])*100:.1f}%)")

if __name__ == "__main__":
    main()