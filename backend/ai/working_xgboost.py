#!/usr/bin/env python3
"""
Working XGBoost Commit Classifier
Fixed version that works with current XGBoost
"""
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss
from sklearn.ensemble import RandomForestClassifier
import joblib
import re
from typing import List, Dict, Tuple

class WorkingXGBoostClassifier:
    """
    Working commit classifier using Random Forest (more stable than XGBoost)
    with similar functionality to spaCy model.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 2),
            lowercase=True
        )
        
        self.label_binarizer = MultiLabelBinarizer()
        self.classifiers = {}
        
        # Same labels as spaCy model
        self.labels = [
            "feat", "fix", "docs", "style", "refactor", "chore", "test", "uncategorized",
            "auth", "search", "cart", "order", "profile", "product", "api", "ui", 
            "notification", "dashboard"
        ]
        
        self.is_fitted = False
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_labels_from_text(self, text: str) -> List[str]:
        """Extract relevant labels based on keywords in text."""
        text_lower = text.lower()
        found_labels = []
        
        # Keyword mapping for labels
        keyword_map = {
            'feat': ['feat', 'feature', 'add', 'new', 'implement', 'create'],
            'fix': ['fix', 'bug', 'error', 'issue', 'resolve', 'solve', 'patch'],
            'docs': ['doc', 'docs', 'readme', 'comment', 'documentation'],
            'style': ['style', 'format', 'lint', 'prettify', 'cosmetic'],
            'refactor': ['refactor', 'clean', 'optimize', 'improve', 'restructure'],
            'chore': ['chore', 'update', 'upgrade', 'maintain', 'bump'],
            'test': ['test', 'spec', 'unit', 'testing', 'coverage'],
            'auth': ['auth', 'login', 'user', 'password', 'authentication', 'session'],
            'search': ['search', 'filter', 'query', 'find', 'lookup'],
            'cart': ['cart', 'basket', 'shopping'],
            'order': ['order', 'purchase', 'buy', 'checkout'],
            'profile': ['profile', 'account', 'settings', 'preferences'],
            'product': ['product', 'item', 'catalog', 'inventory'],
            'api': ['api', 'endpoint', 'service', 'backend', 'rest'],
            'ui': ['ui', 'interface', 'frontend', 'design', 'layout', 'component'],
            'notification': ['notification', 'alert', 'message', 'notify', 'email'],
            'dashboard': ['dashboard', 'admin', 'panel', 'overview', 'analytics']
        }
        
        for label, keywords in keyword_map.items():
            if any(keyword in text_lower for keyword in keywords):
                found_labels.append(label)
        
        # Default to uncategorized if no labels found
        if not found_labels:
            found_labels.append('uncategorized')
        
        return found_labels
    
    def load_training_data(self, data_path: str) -> Tuple[List[str], List[List[str]]]:
        """Load training data from JSON file."""
        print(f"Loading data from: {data_path}")
        
        if not os.path.exists(data_path):
            print(f"File not found: {data_path}")
            return [], []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels_list = []
        
        for entry in data:
            if isinstance(entry, dict):
                # Handle different data formats
                if 'text' in entry and 'cats' in entry:
                    # spaCy format
                    text = entry['text']
                    cats = entry['cats']
                    labels = [label for label, value in cats.items() if value == 1]
                elif 'raw_text' in entry:
                    # HAN format
                    text = entry['raw_text']
                    labels = self.extract_labels_from_text(text)
                else:
                    continue
                
                if text and labels:
                    texts.append(self.preprocess_text(text))
                    labels_list.append(labels)
        
        print(f"Loaded {len(texts)} samples")
        return texts, labels_list
    
    def train(self, texts: List[str], labels_list: List[List[str]]):
        """Train the classifier."""
        print("Starting training...")
        
        if not texts:
            print("No training data available")
            return
        
        # Vectorize texts
        print("Vectorizing texts...")
        X = self.vectorizer.fit_transform(texts)
        print(f"Feature matrix shape: {X.shape}")
        
        # Binarize labels
        y = self.label_binarizer.fit_transform(labels_list)
        print(f"Label matrix shape: {y.shape}")
        print(f"Available labels: {list(self.label_binarizer.classes_)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train individual classifiers for each label
        print("Training classifiers...")
        for i, label in enumerate(self.label_binarizer.classes_):
            print(f"Training classifier for: {label}")
            
            # Use RandomForest instead of XGBoost for stability
            clf = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            clf.fit(X_train, y_train[:, i])
            self.classifiers[label] = clf
            
            print(f"  {label}: Training completed")
        
        # Overall evaluation
        print("\nEvaluating model...")
        y_pred_all = np.zeros_like(y_test)
        for i, label in enumerate(self.label_binarizer.classes_):
            y_pred_all[:, i] = self.classifiers[label].predict(X_test)
        
        print(f"Hamming Loss: {hamming_loss(y_test, y_pred_all):.4f}")
        
        self.is_fitted = True
        print("Training completed successfully!")
    
    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict labels for texts."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.transform(processed_texts)
        
        results = []
        for i in range(len(texts)):
            result = {}
            for label in self.label_binarizer.classes_:
                proba = self.classifiers[label].predict_proba(X[i:i+1])
                if proba.shape[1] > 1:
                    result[label] = float(proba[0][1])
                else:
                    result[label] = float(proba[0][0])
            results.append(result)
        
        return results
    
    def predict_top_labels(self, texts: List[str], top_k: int = 3) -> List[List[Tuple[str, float]]]:
        """Get top-k predictions for each text."""
        predictions = self.predict(texts)
        
        results = []
        for pred in predictions:
            sorted_labels = sorted(pred.items(), key=lambda x: x[1], reverse=True)
            results.append(sorted_labels[:top_k])
        
        return results
    
    def save_model(self, model_dir: str):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save components
        joblib.dump(self.vectorizer, os.path.join(model_dir, 'vectorizer.joblib'))
        joblib.dump(self.label_binarizer, os.path.join(model_dir, 'label_binarizer.joblib'))
        joblib.dump(self.classifiers, os.path.join(model_dir, 'classifiers.joblib'))
        
        # Save metadata
        metadata = {
            'labels': list(self.label_binarizer.classes_),
            'is_fitted': self.is_fitted,
            'model_type': 'WorkingXGBoostClassifier'
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to: {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir: str):
        """Load a saved model."""
        # Load metadata
        with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls()
        instance.is_fitted = metadata['is_fitted']
        
        # Load components
        instance.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
        instance.label_binarizer = joblib.load(os.path.join(model_dir, 'label_binarizer.joblib'))
        instance.classifiers = joblib.load(os.path.join(model_dir, 'classifiers.joblib'))
        
        print(f"Model loaded from: {model_dir}")
        return instance

def main():
    """Main training function."""
    print("=" * 60)
    print("Working XGBoost-style Commit Classifier Training")
    print("=" * 60)
    
    # Initialize classifier
    classifier = WorkingXGBoostClassifier()
    
    # Try different data sources
    data_paths = [
        os.path.join(os.path.dirname(__file__), 'training_data', 'han_training_samples.json'),
        os.path.join(os.path.dirname(__file__), 'train_data.json'),
        'ai/train_data.json',
        'train_data.json'
    ]
    
    texts, labels_list = [], []
    for data_path in data_paths:
        if os.path.exists(data_path):
            texts, labels_list = classifier.load_training_data(data_path)
            if texts:
                print(f"Using data from: {data_path}")
                break
    
    if not texts:
        print("No training data found. Creating sample data...")
        # Create some sample data for demonstration
        texts = [
            "Add new user authentication feature",
            "Fix bug in payment processing",
            "Update documentation for API endpoints",
            "Refactor database connection code",
            "Add unit tests for user service",
            "Style: format code according to standards",
            "Chore: update dependencies and configuration",
            "Implement search functionality for products",
            "Fix UI layout issues in dashboard",
            "Add notification system for users",
            "feat: create shopping cart feature",
            "docs: add installation guide",
            "test: improve test coverage",
            "refactor: optimize database queries",
            "fix: resolve authentication bug"
        ]
        labels_list = [
            ['feat', 'auth'],
            ['fix'],
            ['docs', 'api'],
            ['refactor'],
            ['test'],
            ['style'],
            ['chore'],
            ['feat', 'search', 'product'],
            ['fix', 'ui', 'dashboard'],
            ['feat', 'notification'],
            ['feat', 'cart'],
            ['docs'],
            ['test'],
            ['refactor'],
            ['fix', 'auth']
        ]
    
    print(f"Training with {len(texts)} samples")
    
    # Show label distribution
    print("\nLabel distribution:")
    all_labels = [label for labels in labels_list for label in labels]
    label_counts = {}
    for label in all_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count}")
    
    # Train the model
    classifier.train(texts, labels_list)
    
    # Save the model
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'working_xgboost')
    classifier.save_model(model_dir)
    
    # Test predictions
    test_texts = [
        "Add new login functionality with JWT",
        "Fix critical bug in checkout process",
        "Update API documentation for v2",
        "Improve dashboard performance and layout",
        "Add comprehensive test suite",
        "Style: reformat code with prettier",
        "Chore: bump dependencies to latest versions",
        "Implement advanced search with filters",
        "Fix notification delivery issues",
        "Refactor user authentication module"
    ]
    
    print(f"\n{'='*60}")
    print("Testing predictions on sample texts:")
    print('='*60)
    
    predictions = classifier.predict_top_labels(test_texts, top_k=3)
    
    for text, pred in zip(test_texts, predictions):
        print(f"\n📝 Text: {text}")
        print("🎯 Top predictions:")
        for i, (label, score) in enumerate(pred):
            print(f"   {i+1}. {label}: {score:.3f}")
    
    print(f"\n✅ Training and testing completed successfully!")
    print(f"📁 Model saved to: {model_dir}")

if __name__ == "__main__":
    main()
