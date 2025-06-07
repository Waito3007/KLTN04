"""
Simple test script for XGBoost model
"""
import sys
import os

print("Starting XGBoost test...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

try:
    import xgboost as xgb
    print(f"XGBoost version: {xgb.__version__}")
except ImportError as e:
    print(f"XGBoost import error: {e}")
    sys.exit(1)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("scikit-learn import successful")
except ImportError as e:
    print(f"scikit-learn import error: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"Pandas import error: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy import error: {e}")
    sys.exit(1)

# Test data path
data_path = os.path.join(os.path.dirname(__file__), 'training_data', 'han_training_samples.json')
print(f"Looking for training data at: {data_path}")
print(f"Training data exists: {os.path.exists(data_path)}")

if os.path.exists(data_path):
    try:
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Training data loaded: {len(data)} samples")
        if len(data) > 0:
            print(f"Sample data structure: {list(data[0].keys())}")
    except Exception as e:
        print(f"Error loading training data: {e}")

print("Test completed successfully!")
