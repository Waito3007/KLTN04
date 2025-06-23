# KLTN04\backend\services\model_loader.py
import joblib
from pathlib import Path
from typing import Optional, Union
import logging
from functools import lru_cache
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    _instance = None
    
    def __init__(self):
        try:
            model_path = self._get_model_path()
            logger.info(f"Loading model from {model_path}")
            
            self.model_data = joblib.load(model_path)
            self.model = self.model_data['model']
            self.vectorizer = self.model_data['vectorizer']
            
            # Warm-up predict
            self._warm_up()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.exception("Failed to load model")
            raise

    @staticmethod
    def _get_model_path() -> Path:
        """Validate and return model path"""
        model_path = Path(__file__).parent.parent / "models" / "commit_classifier_v1.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return model_path

    def _warm_up(self):
        """Warm-up model with sample input"""
        sample = "fix: critical security vulnerability"
        self.predict(sample)
        
    @lru_cache(maxsize=1000)
    def vectorize(self, message: str) -> np.ndarray:
        """Cache vectorized results for frequent messages"""
        return self.vectorizer.transform([message])

    def predict(self, message: str) -> int:
        """Predict if commit is critical (with input validation)"""
        if not message or not isinstance(message, str):
            raise ValueError("Input must be non-empty string")
            
        X = self.vectorize(message.strip())
        return int(self.model.predict(X)[0])

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def predict_commit(message: str) -> dict:
    """Public API for commit prediction
    
    Returns:
        {
            "prediction": 0|1,
            "confidence": float,
            "error": str|None
        }
    """
    try:
        loader = ModelLoader.get_instance()
        proba = loader.model.predict_proba(loader.vectorize(message))[0]
        return {
            "prediction": loader.predict(message),
            "confidence": float(np.max(proba)),
            "error": None
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {
            "prediction": -1,
            "confidence": 0.0,
            "error": str(e)
        }