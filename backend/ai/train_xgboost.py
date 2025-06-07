"""
Training script for XGBoost commit classifier.
This script trains an XGBoost model with similar functionality to the spaCy model.
"""

import os
import sys
import json
from datetime import datetime

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from ai.xgboost_model import XGBoostCommitClassifier

def train_xgboost_model():
    """
    Train and save the XGBoost commit classifier model.
    """
    print("=" * 60)
    print("XGBoost Commit Classifier Training")
    print("=" * 60)
    print(f"Training started at: {datetime.now()}")
    
    try:
        # Initialize classifier
        print("\n1. Initializing XGBoost classifier...")
        classifier = XGBoostCommitClassifier(
            max_features=5000,
            min_df=2,
            max_df=0.9
        )
        
        # Define data path
        data_path = os.path.join(
            os.path.dirname(__file__), 
            'training_data', 
            'han_training_samples.json'
        )
        
        if not os.path.exists(data_path):
            print(f"❌ Training data not found at {data_path}")
            return False
        
        # Load training data
        print(f"\n2. Loading training data from {data_path}...")
        texts, labels_list = classifier.load_training_data(data_path)
        
        if not texts:
            print("❌ No training data loaded")
            return False
            
        print(f"✅ Loaded {len(texts)} training samples")
        
        # Show label distribution
        print("\n3. Analyzing label distribution...")
        all_labels = [label for labels in labels_list for label in labels]
        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("Label distribution:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count}")
        
        # Train the model
        print(f"\n4. Training XGBoost model...")
        classifier.train(texts, labels_list, test_size=0.2)
        
        # Create models directory
        model_dir = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_commit_classifier')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        print(f"\n5. Saving model to {model_dir}...")
        classifier.save_model(model_dir)
        
        # Test with sample data
        print(f"\n6. Testing with sample commit messages...")
        sample_texts = [
            "Add new user authentication feature with JWT tokens",
            "Fix critical bug in payment processing module",
            "Update API documentation for user endpoints",
            "Refactor database connection and query optimization",
            "Add comprehensive unit tests for user service",
            "Style: format code and fix linting issues",
            "Chore: update dependencies and configuration",
            "feat: implement search functionality for products",
            "docs: add README for installation guide",
            "ui: improve dashboard layout and responsiveness"
        ]
        
        predictions = classifier.predict_top_labels(sample_texts, top_k=3)
        
        print("\nSample predictions:")
        for i, (text, pred) in enumerate(zip(sample_texts, predictions)):
            print(f"\n📝 Text: {text}")
            print("🎯 Top predictions:")
            for label, prob in pred:
                print(f"   {label}: {prob:.3f}")
        
        print(f"\n✅ Training completed successfully!")
        print(f"⏰ Training finished at: {datetime.now()}")
        print(f"📁 Model saved to: {model_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_spacy():
    """
    Compare XGBoost model performance with spaCy model (if available).
    """
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    
    try:
        # Load XGBoost model
        model_dir = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_commit_classifier')
        if not os.path.exists(model_dir):
            print("❌ XGBoost model not found. Train it first.")
            return
        
        xgb_classifier = XGBoostCommitClassifier.load_model(model_dir)
        
        # Test texts
        test_texts = [
            "Add user registration and login functionality",
            "Fix memory leak in data processing",
            "Update documentation for new API",
            "Refactor code for better performance",
            "Add tests for authentication module"
        ]
        
        print("XGBoost predictions:")
        xgb_predictions = xgb_classifier.predict_top_labels(test_texts, top_k=2)
        
        for i, (text, pred) in enumerate(zip(test_texts, xgb_predictions)):
            print(f"\n📝 {text}")
            print(f"🤖 XGBoost: {pred[0][0]} ({pred[0][1]:.3f}), {pred[1][0]} ({pred[1][1]:.3f})")
        
        # Try to load spaCy model for comparison
        try:
            import spacy
            spacy_model_path = os.path.join(os.path.dirname(__file__), 'modelAi')
            if os.path.exists(spacy_model_path):
                nlp = spacy.load(spacy_model_path)
                print("\nspaCy predictions:")
                
                for text in test_texts:
                    doc = nlp(text)
                    cats = doc.cats
                    top_cats = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:2]
                    print(f"📝 {text}")
                    print(f"🧠 spaCy: {top_cats[0][0]} ({top_cats[0][1]:.3f}), {top_cats[1][0]} ({top_cats[1][1]:.3f})")
            else:
                print("ℹ️  spaCy model not found for comparison")
                
        except ImportError:
            print("ℹ️  spaCy not available for comparison")
            
    except Exception as e:
        print(f"❌ Comparison failed: {str(e)}")

if __name__ == "__main__":
    print("Starting XGBoost model training process...")
    
    success = train_xgboost_model()
    
    if success:
        compare_with_spacy()
    else:
        print("❌ Training failed. Please check the errors above.")
