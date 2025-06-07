"""
Simplified XGBoost training script for commit classification
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import xgboost as xgb

def main():
    print("=" * 60)
    print("XGBoost Commit Classifier Training (Simplified)")
    print("=" * 60)
    
    # Load training data
    data_path = os.path.join(os.path.dirname(__file__), 'training_data', 'han_training_samples.json')
    
    if not os.path.exists(data_path):
        print(f"❌ Training data not found at {data_path}")
        return
    
    print(f"📁 Loading data from: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} samples")
    
    # Prepare data
    texts = []
    labels = []
    
    # Simple label mapping based on text patterns
    label_keywords = {
        'feat': ['feat', 'feature', 'add', 'new', 'implement'],
        'fix': ['fix', 'bug', 'error', 'issue', 'resolve'],
        'docs': ['doc', 'readme', 'comment', 'documentation'],
        'style': ['style', 'format', 'lint', 'clean'],
        'refactor': ['refactor', 'optimize', 'improve'],
        'test': ['test', 'spec', 'unit', 'integration'],
        'chore': ['chore', 'update', 'upgrade', 'dependency']
    }
    
    for entry in data[:1000]:  # Use first 1000 samples for faster training
        if 'raw_text' in entry:
            text = entry['raw_text'].lower()
            texts.append(text)
            
            # Determine primary label based on keywords
            found_label = 'other'
            for label, keywords in label_keywords.items():
                if any(keyword in text for keyword in keywords):
                    found_label = label
                    break
            
            labels.append(found_label)
    
    if not texts:
        print("❌ No valid training data found")
        return
    
    print(f"📊 Using {len(texts)} training samples")
    
    # Label distribution
    unique_labels = list(set(labels))
    print(f"🏷️  Labels: {unique_labels}")
    
    label_counts = {label: labels.count(label) for label in unique_labels}
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {label}: {count}")
    
    # Vectorize text
    print("\n🔄 Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(texts)
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    print(f"✅ Features: {X.shape[1]}")
    print(f"✅ Classes: {len(label_encoder.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📈 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    
    # Train XGBoost model
    print("\n🚀 Training XGBoost model...")
    
    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # XGBoost parameters
    params = {
        'objective': 'multi:softprob',
        'num_class': len(label_encoder.classes_),
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    
    # Train model
    num_rounds = 100
    model = xgb.train(
        params,
        dtrain,
        num_rounds,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=20
    )
    
    # Make predictions
    print("\n📊 Evaluating model...")
    y_pred_proba = model.predict(dtest)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Print classification report
    print("\n📈 Classification Report:")
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_,
        zero_division=0
    ))
    
    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_simple')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'xgboost_model.json')
    model.save_model(model_path)
    
    # Save preprocessing components
    joblib.dump(vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))
    
    # Save metadata
    metadata = {
        'labels': label_encoder.classes_.tolist(),
        'num_features': X.shape[1],
        'num_samples': len(texts),
        'training_date': datetime.now().isoformat()
    }
    
    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Model saved to: {model_dir}")
    
    # Test with sample texts
    print("\n🧪 Testing with sample texts:")
    sample_texts = [
        "Add new user authentication feature",
        "Fix bug in payment processing",
        "Update documentation for API",
        "Refactor database connection code",
        "Add unit tests for user service"
    ]
    
    for text in sample_texts:
        X_sample = vectorizer.transform([text.lower()])
        dsample = xgb.DMatrix(X_sample)
        pred_proba = model.predict(dsample)[0]
        
        # Get top prediction
        top_idx = np.argmax(pred_proba)
        top_label = label_encoder.classes_[top_idx]
        top_prob = pred_proba[top_idx]
        
        print(f"📝 '{text}'")
        print(f"🎯 Predicted: {top_label} ({top_prob:.3f})")
    
    print(f"\n🎉 Training completed successfully!")

if __name__ == "__main__":
    main()
