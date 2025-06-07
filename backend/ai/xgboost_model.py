import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import json
import joblib
import re
from typing import List, Dict, Tuple
import os

class XGBoostCommitClassifier:
    """
    XGBoost-based commit message classifier with similar functionality to spaCy model.
    Supports multi-label classification for commit types and categories.
    """
    
    def __init__(self, max_features: int = 5000, min_df: int = 2, max_df: float = 0.9):
        """
        Initialize the XGBoost commit classifier.
        
        Args:
            max_features: Maximum number of features for TF-IDF vectorizer
            min_df: Minimum document frequency for TF-IDF
            max_df: Maximum document frequency for TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        self.label_binarizer = MultiLabelBinarizer()
        
        # XGBoost classifier with multi-output support
        self.classifier = MultiOutputClassifier(
            xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        )
        
        # Labels similar to spaCy model
        self.labels = [
            "feat", "fix", "docs", "style", "refactor", "chore", "test", "uncategorized",
            "auth", "search", "cart", "order", "profile", "product", "api", "ui", 
            "notification", "dashboard"
        ]
        
        self.is_fitted = False
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess commit message text.
        
        Args:
            text: Raw commit message
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def extract_labels_from_commit(self, text: str, cats: Dict) -> List[str]:
        """
        Extract labels from commit message and categories.
        
        Args:
            text: Commit message
            cats: Category dictionary from training data
            
        Returns:
            List of applicable labels
        """
        labels = []
        text_lower = text.lower()
        
        # Rule-based label extraction based on keywords
        if any(keyword in text_lower for keyword in ['feat', 'feature', 'add', 'new']):
            labels.append('feat')
        if any(keyword in text_lower for keyword in ['fix', 'bug', 'error', 'issue']):
            labels.append('fix')
        if any(keyword in text_lower for keyword in ['doc', 'readme', 'comment']):
            labels.append('docs')
        if any(keyword in text_lower for keyword in ['style', 'format', 'lint']):
            labels.append('style')
        if any(keyword in text_lower for keyword in ['refactor', 'clean', 'optimize']):
            labels.append('refactor')
        if any(keyword in text_lower for keyword in ['chore', 'update', 'upgrade']):
            labels.append('chore')
        if any(keyword in text_lower for keyword in ['test', 'spec', 'unit']):
            labels.append('test')
        if any(keyword in text_lower for keyword in ['auth', 'login', 'user', 'password']):
            labels.append('auth')
        if any(keyword in text_lower for keyword in ['search', 'filter', 'query']):
            labels.append('search')
        if any(keyword in text_lower for keyword in ['cart', 'basket', 'shopping']):
            labels.append('cart')
        if any(keyword in text_lower for keyword in ['order', 'purchase', 'buy']):
            labels.append('order')
        if any(keyword in text_lower for keyword in ['profile', 'account', 'settings']):
            labels.append('profile')
        if any(keyword in text_lower for keyword in ['product', 'item', 'catalog']):
            labels.append('product')
        if any(keyword in text_lower for keyword in ['api', 'endpoint', 'service']):
            labels.append('api')
        if any(keyword in text_lower for keyword in ['ui', 'interface', 'frontend', 'design']):
            labels.append('ui')
        if any(keyword in text_lower for keyword in ['notification', 'alert', 'message']):
            labels.append('notification')
        if any(keyword in text_lower for keyword in ['dashboard', 'admin', 'panel']):
            labels.append('dashboard')
        
        # Add labels from cats if available
        if cats:
            for label, value in cats.items():
                if value == 1 and label in self.labels:
                    labels.append(label)
        
        # Default to uncategorized if no labels found
        if not labels:
            labels.append('uncategorized')
            
        return list(set(labels))  # Remove duplicates
    
    def load_training_data(self, json_file_path: str) -> Tuple[List[str], List[List[str]]]:
        """
        Load training data from JSON file.
        
        Args:
            json_file_path: Path to training data JSON file
            
        Returns:
            Tuple of (texts, labels)
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels_list = []
        
        for entry in data:
            if isinstance(entry, dict):
                # Handle different data formats
                if 'raw_text' in entry:
                    text = entry['raw_text']
                    # Extract labels from entry
                    labels = self.extract_labels_from_commit(text, entry.get('labels', {}))
                elif 'text' in entry:
                    text = entry['text']
                    cats = entry.get('cats', {})
                    labels = self.extract_labels_from_commit(text, cats)
                else:
                    continue
                    
                texts.append(self.preprocess_text(text))
                labels_list.append(labels)
        
        return texts, labels_list
    
    def train(self, texts: List[str], labels_list: List[List[str]], test_size: float = 0.2):
        """
        Train the XGBoost model.
        
        Args:
            texts: List of commit messages
            labels_list: List of label lists for each commit
            test_size: Size of test set for evaluation
        """
        print("Starting XGBoost model training...")
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        
        # Binarize labels
        y = self.label_binarizer.fit_transform(labels_list)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"Feature dimensions: {X_train.shape[1]}")
        print(f"Number of labels: {len(self.label_binarizer.classes_)}")
        
        # Train the model
        self.classifier.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.classifier.predict(X_test)
        
        print("\nEvaluation Results:")
        for i, label in enumerate(self.label_binarizer.classes_):
            print(f"\nLabel: {label}")
            print(classification_report(y_test[:, i], y_pred[:, i], target_names=['No', 'Yes']))
        
        self.is_fitted = True
        print("Training completed successfully!")
    
    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Predict labels for given texts.
        
        Args:
            texts: List of commit messages
            
        Returns:
            List of dictionaries with label probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        X = self.vectorizer.transform(processed_texts)
        
        # Get probabilities
        probabilities = self.classifier.predict_proba(X)
        
        results = []
        for i in range(len(texts)):
            result = {}
            for j, label in enumerate(self.label_binarizer.classes_):
                # Handle both binary and multi-class probability outputs
                if hasattr(probabilities[j][i], '__len__') and len(probabilities[j][i]) > 1:
                    result[label] = float(probabilities[j][i][1])  # Probability of positive class
                else:
                    result[label] = float(probabilities[j][i])
            results.append(result)
        
        return results
    
    def predict_top_labels(self, texts: List[str], top_k: int = 3) -> List[List[Tuple[str, float]]]:
        """
        Predict top-k labels for given texts.
        
        Args:
            texts: List of commit messages
            top_k: Number of top labels to return
            
        Returns:
            List of lists containing (label, probability) tuples
        """
        predictions = self.predict(texts)
        
        results = []
        for pred in predictions:
            # Sort labels by probability
            sorted_labels = sorted(pred.items(), key=lambda x: x[1], reverse=True)
            results.append(sorted_labels[:top_k])
        
        return results
    
    def save_model(self, model_dir: str):
        """
        Save the trained model to disk.
        
        Args:
            model_dir: Directory to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save components
        joblib.dump(self.classifier, os.path.join(model_dir, 'xgboost_classifier.joblib'))
        joblib.dump(self.vectorizer, os.path.join(model_dir, 'vectorizer.joblib'))
        joblib.dump(self.label_binarizer, os.path.join(model_dir, 'label_binarizer.joblib'))
        
        # Save metadata
        metadata = {
            'labels': self.labels,
            'is_fitted': self.is_fitted,
            'model_type': 'XGBoostCommitClassifier'
        }
        
        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir: str):
        """
        Load a trained model from disk.
        
        Args:
            model_dir: Directory containing the saved model
            
        Returns:
            Loaded XGBoostCommitClassifier instance
        """
        # Load metadata
        with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls()
        instance.labels = metadata['labels']
        instance.is_fitted = metadata['is_fitted']
        
        # Load components
        instance.classifier = joblib.load(os.path.join(model_dir, 'xgboost_classifier.joblib'))
        instance.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
        instance.label_binarizer = joblib.load(os.path.join(model_dir, 'label_binarizer.joblib'))
        
        print(f"Model loaded from {model_dir}")
        return instance

def main():
    """
    Main function to train and save the XGBoost model.
    """
    # Initialize classifier
    classifier = XGBoostCommitClassifier()
    
    # Load training data
    data_path = os.path.join(os.path.dirname(__file__), 'training_data', 'han_training_samples.json')
    
    if not os.path.exists(data_path):
        print(f"Training data not found at {data_path}")
        return
    
    print("Loading training data...")
    texts, labels_list = classifier.load_training_data(data_path)
    
    print(f"Loaded {len(texts)} training samples")
    
    # Train the model
    classifier.train(texts, labels_list)
    
    # Save the model
    model_dir = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_commit_classifier')
    classifier.save_model(model_dir)
    
    # Test with some sample texts
    sample_texts = [
        "Add new user authentication feature",
        "Fix bug in payment processing",
        "Update documentation for API endpoints",
        "Refactor database connection logic",
        "Add unit tests for user service"
    ]
    
    print("\nTesting with sample texts:")
    predictions = classifier.predict_top_labels(sample_texts, top_k=3)
    
    for i, (text, pred) in enumerate(zip(sample_texts, predictions)):
        print(f"\nText: {text}")
        print("Top predictions:")
        for label, prob in pred:
            print(f"  {label}: {prob:.3f}")

if __name__ == "__main__":
    main()
