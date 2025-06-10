"""
Simple test script for Multi-Modal Fusion Network
Test basic functionality without heavy dependencies
"""
import sys
import os
import torch
import numpy as np
from typing import Dict, List, Any

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test basic torch functionality
print("Testing basic PyTorch functionality...")
x = torch.randn(2, 3)
print(f"✓ PyTorch tensor creation: {x.shape}")

# Test if we can create a simple model
class SimpleTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleTestModel()
test_input = torch.randn(1, 10)
output = model(test_input)
print(f"✓ Simple model forward pass: {output.shape}")

# Test data generation capability
from multimodal_fusion.data.synthetic_generator import GitHubDataGenerator

print("Testing synthetic data generation...")
generator = GitHubDataGenerator()
sample_data = generator.generate_batch(5)
print(f"✓ Generated {len(sample_data)} samples")
print(f"✓ Sample keys: {list(sample_data[0].keys())}")

print("\nAll basic tests passed! Ready for more complex testing.")
