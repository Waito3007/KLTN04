# KLTN04\backend\models\commit_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

class CommitClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = RandomForestClassifier()
        self.labels = ['normal', 'critical']  # 0: normal, 1: critical/bugfix

    def train(self, df: pd.DataFrame):
        """Huấn luyện model từ dataframe"""
        X = self.vectorizer.fit_transform(df['message'])
        y = df['is_critical']  # Cột nhãn (0/1)
        self.model.fit(X, y)
        
    def predict(self, new_messages: list):
        """Dự đoán commit quan trọng cần review"""
        X_new = self.vectorizer.transform(new_messages)
        return self.model.predict(X_new)
    
    def save(self, path='models/commit_classifier.joblib'):
        """Lưu model"""
        joblib.dump({
            'vectorizer': self.vectorizer,
            'model': self.model
        }, path)
    
    @classmethod
    def load(cls, path='models/commit_classifier.joblib'):
        """Load model đã lưu"""
        data = joblib.load(path)
        classifier = cls()
        classifier.vectorizer = data['vectorizer']
        classifier.model = data['model']
        return classifier