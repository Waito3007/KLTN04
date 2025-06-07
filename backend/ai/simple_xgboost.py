"""
Simple XGBoost model with similar functionality to spaCy for commit classification.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss
import joblib
import re
from typing import List, Dict, Tuple

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available, using sklearn GradientBoosting instead")
    from sklearn.ensemble import GradientBoostingClassifier
    XGBOOST_AVAILABLE = False

class SimpleXGBoostClassifier:
    """
    Simple XGBoost commit classifier similar to spaCy functionality.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            min_df=2,
            max_df=0.8,
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
            'feat': ['feat', 'feature', 'add', 'new', 'implement'],
            'fix': ['fix', 'bug', 'error', 'issue', 'resolve', 'solve'],
            'docs': ['doc', 'docs', 'readme', 'comment', 'documentation'],
            'style': ['style', 'format', 'lint', 'prettify'],
            'refactor': ['refactor', 'clean', 'optimize', 'improve'],
            'chore': ['chore', 'update', 'upgrade', 'maintain'],
            'test': ['test', 'spec', 'unit', 'testing'],
            'auth': ['auth', 'login', 'user', 'password', 'authentication'],
            'search': ['search', 'filter', 'query', 'find'],
            'cart': ['cart', 'basket', 'shopping'],
            'order': ['order', 'purchase', 'buy', 'checkout'],
            'profile': ['profile', 'account', 'settings'],
            'product': ['product', 'item', 'catalog'],
            'api': ['api', 'endpoint', 'service', 'backend'],
            'ui': ['ui', 'interface', 'frontend', 'design', 'layout'],
            'notification': ['notification', 'alert', 'message', 'notify'],
            'dashboard': ['dashboard', 'admin', 'panel', 'overview']
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
        """Train the XGBoost model."""
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
            
            if XGBOOST_AVAILABLE:
                clf = xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss'
                )
            else:
                clf = GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42
                )
            
            clf.fit(X_train, y_train[:, i])
            self.classifiers[label] = clf
            
            # Quick evaluation
            y_pred = clf.predict(X_test)
            print(f"  {label}: Training completed")
        
        # Overall evaluation
        print("\nEvaluating model...")
        y_pred_all = np.zeros_like(y_test)
        for i, label in enumerate(self.label_binarizer.classes_):
            y_pred_all[:, i] = self.classifiers[label].predict(X_test)
        
        print(f"Hamming Loss: {hamming_loss(y_test, y_pred_all):.4f}")
        
        self.is_fitted = True
        print("Training completed!")
    
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
                if hasattr(self.classifiers[label], 'predict_proba'):
                    proba = self.classifiers[label].predict_proba(X[i:i+1])
                    if proba.shape[1] > 1:
                        result[label] = float(proba[0][1])
                    else:
                        result[label] = float(proba[0][0])
                else:
                    result[label] = float(self.classifiers[label].predict(X[i:i+1])[0])
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
            'model_type': 'SimpleXGBoostClassifier',
            'xgboost_available': XGBOOST_AVAILABLE
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
    print("=" * 50)
    print("Simple XGBoost Commit Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = SimpleXGBoostClassifier()
    
    # Try different data sources
    data_paths = [
        'ai/train_data.json',
        'ai/training_data/han_training_samples.json',
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
            "Update documentation for API",
            "Refactor database connection code",
            "Add unit tests for user service",
            "Style: format code according to standards",
            "Chore: update dependencies",
            "Implement search functionality",
            "Fix UI layout issues",
            "Add notification system"
        ]
        labels_list = [
            ['feat', 'auth'],
            ['fix'],
            ['docs', 'api'],
            ['refactor'],
            ['test'],
            ['style'],
            ['chore'],
            ['feat', 'search'],
            ['fix', 'ui'],
            ['feat', 'notification']
        ]
    
    # Train the model
    classifier.train(texts, labels_list)
    
    # Save the model
    model_dir = 'ai/models/simple_xgboost'
    classifier.save_model(model_dir)
    
    # Test predictions
    test_texts = [
        "Add new login functionality",
        "Fix critical bug in checkout process",
        "Update API documentation",
        "Improve dashboard performance"
    ]
    
    print("\nTesting predictions:")
    predictions = classifier.predict_top_labels(test_texts, top_k=2)
    
    for text, pred in zip(test_texts, predictions):
        print(f"\nText: {text}")
        print(f"Predictions: {pred[0][0]} ({pred[0][1]:.3f}), {pred[1][0]} ({pred[1][1]:.3f})")

if __name__ == "__main__":
    main()
