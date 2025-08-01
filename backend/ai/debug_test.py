#!/usr/bin/env python3
"""
Simple Commit Analyzer Test - Debug version
"""

import os
import sys
import json
import torch
from pathlib import Path

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

def test_model_loading():
    """Test basic model loading"""
    print("🚀 SIMPLE COMMIT ANALYZER TEST")
    print("="*50)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Check model file
    model_path = Path(__file__).parent / "models" / "han_github_model" / "best_model.pth"
    print(f"📦 Model path: {model_path}")
    print(f"📦 Model exists: {model_path.exists()}")
    
    if not model_path.exists():
        print("❌ Model file not found!")
        return False
    
    try:
        # Load checkpoint
        print("📥 Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location=device)
        
        print("✅ Checkpoint loaded successfully!")
        print(f"   Keys: {list(checkpoint.keys())}")
        
        if 'num_classes' in checkpoint:
            print(f"   Tasks: {list(checkpoint['num_classes'].keys())}")
            print(f"   Classes per task: {checkpoint['num_classes']}")
        
        if 'val_accuracy' in checkpoint:
            print(f"   Best accuracy: {checkpoint['val_accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

def test_simple_prediction():
    """Test a simple prediction"""
    try:
        from train_han_github import SimpleHANModel, SimpleTokenizer
        print("\n🧪 Testing simple prediction...")
        
        # Load model components
        model_path = Path(__file__).parent / "models" / "han_github_model" / "best_model.pth"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(model_path, map_location=device)
        tokenizer = checkpoint['tokenizer']
        num_classes = checkpoint['num_classes']
        label_encoders = checkpoint['label_encoders']
        
        # Create reverse label encoders
        reverse_encoders = {}
        for task, encoder in label_encoders.items():
            reverse_encoders[task] = {v: k for k, v in encoder.items()}
        
        # Initialize model
        vocab_size = len(tokenizer.word_to_idx)
        model = SimpleHANModel(
            vocab_size=vocab_size,
            embed_dim=100,
            hidden_dim=128,
            num_classes=num_classes
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Model initialized with vocab size: {vocab_size}")
        
        # Test prediction
        test_text = "fix: resolve authentication bug in login endpoint"
        print(f"📝 Testing text: '{test_text}'")
        
        # Tokenize
        input_ids = tokenizer.encode_text(test_text, max_sentences=10, max_words=50)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            
            print(f"🔍 Predictions:")
            for task, output in outputs.items():
                probs = torch.softmax(output, dim=1)
                confidence, pred_idx = torch.max(probs, 1)
                
                pred_idx = pred_idx.item()
                confidence = confidence.item()
                
                predicted_label = reverse_encoders[task][pred_idx]
                print(f"   {task}: {predicted_label} (confidence: {confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in prediction test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting debug tests...\n")
    
    # Test 1: Model loading
    if test_model_loading():
        print("\n" + "="*50)
        # Test 2: Simple prediction
        test_simple_prediction()
    
    print("\n🎯 Debug tests completed!")
