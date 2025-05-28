import os
import json
from ai.modelAi.generate_data import classify_commit_purpose, is_suspicious_commit, extract_tech_tags, classify_sentiment

def build_training_data(collected_dir, output_path):
    all_samples = []
    for file in os.listdir(collected_dir):
        if not file.endswith('.json'):
            continue
        with open(os.path.join(collected_dir, file), 'r', encoding='utf-8') as f:
            items = json.load(f)
        for item in items:
            text = item.get('raw_text', '')
            # Gán nhãn tự động
            labels = {
                'purpose': classify_commit_purpose(text),
                'suspicious': is_suspicious_commit(text),
                'tech_tag': extract_tech_tags(text),
                'sentiment': classify_sentiment(text)
            }
            item['labels'] = labels
            all_samples.append(item)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_samples)} samples to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    collected_dir = os.path.join(base_dir, "collected_data")
    output_path = os.path.join(base_dir, "training_data", "han_training_samples.json")
    build_training_data(
        collected_dir=collected_dir,
        output_path=output_path
    )
