import json
import os

print("Checking data format...")

data_path = 'training_data/improved_100k_multimodal_training.json'
if not os.path.exists(data_path):
    print(f"File not found: {data_path}")
    exit(1)

try:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Data type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Data keys: {list(data.keys())}")
        print(f"Number of top-level keys: {len(data)}")
        
        # Check each key
        for key, value in list(data.items())[:3]:
            print(f"Key '{key}': type={type(value)}, length={len(value) if hasattr(value, '__len__') else 'N/A'}")
            if isinstance(value, (list, dict)) and len(value) > 0:
                first_item = value[0] if isinstance(value, list) else list(value.values())[0]
                print(f"  First item type: {type(first_item)}")
                if isinstance(first_item, dict):
                    print(f"  First item keys: {list(first_item.keys())[:5]}")
        
        # Detailed inspection of first training sample
        if 'train_data' in data and len(data['train_data']) > 0:
            print("\n=== DETAILED FIRST TRAINING SAMPLE ===")
            first_sample = data['train_data'][0]
            print(f"Sample keys: {list(first_sample.keys())}")
            
            # Check each field
            for field, value in first_sample.items():
                print(f"\nField '{field}': type={type(value)}")
                if isinstance(value, dict):
                    print(f"  Dict keys: {list(value.keys())}")
                    for subkey, subvalue in list(value.items())[:3]:
                        print(f"    {subkey}: {type(subvalue)} = {str(subvalue)[:100]}")
                elif isinstance(value, (str, int, float, bool)):
                    print(f"  Value: {str(value)[:200]}")
                elif isinstance(value, list):
                    print(f"  List length: {len(value)}")
                    if len(value) > 0:
                        print(f"  First item: {type(value[0])} = {str(value[0])[:100]}")
                        
    elif isinstance(data, list):
        print(f"Data is a list with {len(data)} items")
        if len(data) > 0:
            print(f"First item type: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"First item keys: {list(data[0].keys())}")
    else:
        print(f"Unexpected data format: {type(data)}")
        
except Exception as e:
    print(f"Error loading data: {e}")
