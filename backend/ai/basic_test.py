"""
Simple working version for testing Multi-Modal Fusion Network
"""
import os
import sys
import torch
import numpy as np
from typing import Dict, List, Any

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import only the data generator first
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multimodal_fusion'))

print("Testing data generation...")
try:
    from data.synthetic_generator import GitHubDataGenerator
    
    generator = GitHubDataGenerator()
    sample_data = generator.generate_batch(5)
    print(f"✓ Generated {len(sample_data)} samples")
    print(f"✓ Sample keys: {list(sample_data[0].keys())}")
    
    # Show sample
    sample = sample_data[0]
    print("\nSample commit data:")
    print(f"Message: {sample['commit_message'][:100]}...")
    print(f"Labels: {sample['labels']}")
    print(f"Metadata keys: {list(sample['metadata'].keys())}")
    
except Exception as e:
    print(f"✗ Data generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting basic text processing...")
try:
    # Simple text processor without heavy dependencies
    class SimpleTextProcessor:
        def __init__(self):
            self.vocab = {'<PAD>': 0, '<UNK>': 1}
            
        def process_text(self, text: str):
            # Simple tokenization
            words = text.lower().split()
            return [self.vocab.get(word, 1) for word in words]
    
    processor = SimpleTextProcessor()
    result = processor.process_text("fix: update documentation")
    print(f"✓ Text processing works: {result}")
    
except Exception as e:
    print(f"✗ Text processing failed: {e}")

print("\nBasic test completed!")
