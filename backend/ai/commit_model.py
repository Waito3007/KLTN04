# AI commit model for TaskFlowAI
import os
import joblib
from pathlib import Path

class CommitClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
    
    @classmethod
    def load(cls):
        """Load the trained commit classifier model"""
        instance = cls()
        
        # Đường dẫn tới model trong thư mục AI
        model_dir = Path(__file__).parent / "trained_models"
        model_path = model_dir / "commit_classifier.joblib"
        
        try:
            if model_path.exists():
                data = joblib.load(model_path)
                if isinstance(data, dict):
                    instance.model = data.get('model')
                    instance.vectorizer = data.get('vectorizer')
                else:
                    instance.model = data
                print(f"✅ Loaded commit classifier from {model_path}")
            else:
                print(f"⚠️ Model file not found at {model_path}")
                # Tạo mock model để tránh lỗi
                instance._create_mock_model()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            instance._create_mock_model()
        
        return instance
    
    def _create_mock_model(self):
        """Create a mock model for development"""
        print("🔧 Creating mock commit classifier...")
        # Mock implementation
        self.model = None
        self.vectorizer = None
    
    def classify(self, commit_message):
        """Classify a commit message"""
        if self.model is None:
            # Mock classification
            return {
                'category': 'feature',
                'confidence': 0.85,
                'description': 'Mock classification result'
            }
        
        try:
            # Real classification logic would go here
            if self.vectorizer:
                features = self.vectorizer.transform([commit_message])
                prediction = self.model.predict(features)[0]
                confidence = self.model.predict_proba(features).max()
                
                return {
                    'category': prediction,
                    'confidence': confidence,
                    'description': f'Classified as {prediction}'
                }
        except Exception as e:
            print(f"Classification error: {e}")
        
        # Fallback
        return {
            'category': 'other',
            'confidence': 0.5,
            'description': 'Classification failed, using fallback'
        }
    
    def save(self, path=None):
        """Save the model"""
        if path is None:
            model_dir = Path(__file__).parent / "trained_models"
            model_dir.mkdir(exist_ok=True)
            path = model_dir / "commit_classifier.joblib"
        
        try:
            if self.model and self.vectorizer:
                data = {
                    'model': self.model,
                    'vectorizer': self.vectorizer
                }
                joblib.dump(data, path)
                print(f"✅ Model saved to {path}")
            else:
                print("⚠️ No model to save")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
