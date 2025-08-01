#!/usr/bin/env python3
"""
Debug script để kiểm tra vấn đề phân loại commit type
"""

import torch
import os
import sys
from test_commit_analyzer import CommitAnalyzer

def test_problematic_commits():
    """Test các commit messages có vấn đề phân loại"""
    
    print("🔍 DEBUGGING COMMIT CLASSIFICATION ISSUES")
    print("="*60)
    
    # Initialize analyzer với model path
    model_path = r"C:\SAN\KLTN\KLTN04\backend\ai\models\han_github_model\best_model.pth"
    analyzer = CommitAnalyzer(model_path)
    
    # Test cases có vấn đề
    test_cases = [
        {
            "message": "docs: fix typo in configuration guide", 
            "expected": "docs",
            "author": "Test User"
        },
        {
            "message": "docs: update installation instructions",
            "expected": "docs", 
            "author": "Test User"
        },
        {
            "message": "test: add unit tests for user service",
            "expected": "test",
            "author": "Test User" 
        },
        {
            "message": "test: fix failing integration tests",
            "expected": "test",
            "author": "Test User"
        },
        {
            "message": "fix: typo in variable name",
            "expected": "fix",
            "author": "Test User"
        },
        {
            "message": "feat: add new documentation system", 
            "expected": "feat",
            "author": "Test User"
        },
        {
            "message": "chore: update dependencies",
            "expected": "chore", 
            "author": "Test User"
        },
        {
            "message": "style: fix code formatting",
            "expected": "style",
            "author": "Test User"
        }
    ]
    
    print(f"🧪 Testing {len(test_cases)} problematic commit messages...")
    print()
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        message = case["message"]
        expected = case["expected"]
        author = case["author"]
        
        print(f"{i}. Testing: '{message}'")
        print(f"   Expected: {expected}")
        
        # Analyze commit
        analysis = analyzer.predict_commit(message, author)
        predicted = analysis.predicted_labels.get('commit_type', 'unknown')
        confidence = analysis.confidence_scores.get('commit_type', 0.0)
        
        print(f"   Predicted: {predicted} (confidence: {confidence:.3f})")
        
        # Check if correct
        is_correct = predicted == expected
        if is_correct:
            print("   ✅ CORRECT")
            correct_predictions += 1
        else:
            print("   ❌ WRONG")
            
            # Analyze why it's wrong - show all predictions for this commit
            print("   🔍 Full predictions:")
            for task, pred in analysis.predicted_labels.items():
                conf = analysis.confidence_scores.get(task, 0.0)
                print(f"      {task}: {pred} ({conf:.3f})")
        
        print()
    
    # Summary
    accuracy = correct_predictions / total_predictions * 100
    print("="*60)
    print(f"📊 CLASSIFICATION ACCURACY: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
    
    if accuracy < 80:
        print("🚨 LOW ACCURACY DETECTED!")
        print("💡 Possible issues:")
        print("   - Model wasn't trained properly on commit prefixes")
        print("   - Training data didn't have enough conventional commit examples")
        print("   - Model is focusing on content words rather than prefixes")
        print("   - Need to retrain with better conventional commit dataset")
    else:
        print("✅ Classification accuracy looks good!")
    
    return accuracy

def analyze_model_attention():
    """Phân tích xem model đang chú ý vào phần nào của commit message"""
    
    print("\n🧠 ANALYZING MODEL ATTENTION PATTERNS")
    print("="*60)
    
    model_path = r"C:\SAN\KLTN\KLTN04\backend\ai\models\han_github_model\best_model.pth"
    analyzer = CommitAnalyzer(model_path)
    
    # Test với các variations
    test_variations = [
        "docs: fix typo in configuration guide",
        "fix typo in configuration guide",  # Không có prefix
        "docs: update configuration guide",  # Không có từ "fix"
        "fix: typo in configuration guide"   # Prefix khác
    ]
    
    print("Testing how model responds to different parts of the message:")
    print()
    
    for i, message in enumerate(test_variations, 1):
        print(f"{i}. '{message}'")
        analysis = analyzer.predict_commit(message, "Test User")
        predicted = analysis.predicted_labels.get('commit_type', 'unknown')
        confidence = analysis.confidence_scores.get('commit_type', 0.0)
        print(f"   → {predicted} ({confidence:.3f})")
        print()
    
    return test_variations

if __name__ == "__main__":
    try:
        accuracy = test_problematic_commits()
        analyze_model_attention()
        
        print("\n💡 RECOMMENDATIONS:")
        print("="*60)
        
        if accuracy < 80:
            print("🔧 TO FIX CLASSIFICATION ISSUES:")
            print("1. Retrain model with more conventional commit examples")
            print("2. Add prefix-aware preprocessing")
            print("3. Use rule-based fallback for obvious prefixes")
            print("4. Create training data with equal distribution of commit types")
            print("5. Consider ensemble model (ML + rule-based)")
        
        print("\n✅ Debug completed!")
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()
