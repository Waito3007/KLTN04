# backend/ai/han_commit_analyzer.py
"""
HAN Commit Analyzer - Load and inference for HAN model
"""
import os
import torch
import torch.nn as nn
import json
from typing import Dict, Any

class HANCommitAnalyzer:
    def __init__(self, model_dir=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), 'models', 'han_github_model')
        self.model_path = os.path.join(self.model_dir, 'best_model.pth')
        self.is_loaded = False
        self.task_labels = {
            'risk': ['low', 'medium', 'high'],
            'complexity': ['low', 'medium', 'high'], 
            'hotspot': ['no', 'yes', 'critical'],
            'urgency': ['low', 'medium', 'high']
        }
        self._load_model()

    def _load_model(self):
        """Load HAN model with error handling"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Model file not found: {self.model_path}")
                return
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            config = checkpoint.get('model_config', {
                'vocab_size': 10000,
                'embedding_dim': 128,
                'hidden_dim': 64,
                'num_classes': {'risk': 3, 'complexity': 3, 'hotspot': 3, 'urgency': 3}
            })
            
            self.task_labels = config.get('task_labels', self.task_labels)
            self.model = self._build_model(config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print("HAN model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load HAN model: {e}")
            self.is_loaded = False

    def _build_model(self, config):
        """Build HAN model architecture"""
        class HANModel(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
                
                # Multi-task heads
                self.classifiers = nn.ModuleDict()
                for task, ncls in num_classes.items():
                    self.classifiers[task] = nn.Linear(hidden_dim * 2, ncls)
                    
            def forward(self, x):
                x = self.embedding(x)
                x, _ = self.lstm(x)
                x = x.mean(dim=1)  # Global average pooling
                
                outputs = {}
                for task, classifier in self.classifiers.items():
                    outputs[task] = classifier(x)
                return outputs
                
        return HANModel(
            config.get('vocab_size', 10000),
            config.get('embedding_dim', 128),
            config.get('hidden_dim', 64),
            config.get('num_classes', {'risk': 3, 'complexity': 3, 'hotspot': 3, 'urgency': 3})
        )

    def preprocess(self, message: str):
        """Simple tokenization and preprocessing"""
        # Simple hash-based tokenization
        tokens = [abs(hash(w)) % 10000 for w in message.lower().split()]
        
        # Pad or truncate to fixed length
        max_length = 32
        if len(tokens) < max_length:
            tokens += [0] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
            
        return torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)

    def predict_commit_analysis(self, message: str) -> Dict[str, Any]:
        """Predict multi-task analysis for commit message"""
        if not self.is_loaded:
            # Return mock prediction when model is not loaded
            return self._mock_prediction(message)
            
        try:
            x = self.preprocess(message)
            
            with torch.no_grad():
                logits_dict = self.model(x)
                result = {}
                
                for task, logits in logits_dict.items():
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    labels = self.task_labels.get(task, [str(i) for i in range(len(probs))])
                    pred = int(probs.argmax())
                    
                    result[task] = labels[pred]
                    result[f'{task}_probs'] = {k: float(v) for k, v in zip(labels, probs)}
                
                result['input'] = message
                return result
                
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}', 'input': message}

    def _mock_prediction(self, message: str) -> Dict[str, Any]:
        """Mock prediction when model is not available"""
        import random
        
        message_lower = message.lower()
        
        # Simple keyword-based mock analysis
        if any(word in message_lower for word in ['fix', 'bug', 'error', 'critical']):
            risk_pred = 'high'
            complexity_pred = 'medium'
            urgency_pred = 'high'
            hotspot_pred = 'yes'
        elif any(word in message_lower for word in ['feat', 'feature', 'add', 'new']):
            risk_pred = 'medium'
            complexity_pred = 'high'
            urgency_pred = 'medium'
            hotspot_pred = 'no'
        elif any(word in message_lower for word in ['docs', 'doc', 'readme', 'comment']):
            risk_pred = 'low'
            complexity_pred = 'low'
            urgency_pred = 'low'
            hotspot_pred = 'no'
        else:
            risk_pred = 'medium'
            complexity_pred = 'medium'
            urgency_pred = 'medium'
            hotspot_pred = 'no'
        
        result = {
            'risk': risk_pred,
            'complexity': complexity_pred,
            'urgency': urgency_pred,
            'hotspot': hotspot_pred,
            'input': message,
            'mock': True  # Indicate this is a mock prediction
        }
        
        # Add probability distributions
        for task in ['risk', 'complexity', 'urgency']:
            labels = self.task_labels[task]
            pred_idx = labels.index(result[task])
            probs = [0.1, 0.1, 0.1]
            probs[pred_idx] = 0.7
            result[f'{task}_probs'] = {k: v for k, v in zip(labels, probs)}
        
        # Hotspot probabilities
        hotspot_labels = self.task_labels['hotspot']
        hotspot_idx = hotspot_labels.index(result['hotspot'])
        hotspot_probs = [0.1, 0.1, 0.1]
        hotspot_probs[hotspot_idx] = 0.8
        result['hotspot_probs'] = {k: v for k, v in zip(hotspot_labels, hotspot_probs)}
        
        return result

    def load_model(self):
        """Public method to load model (for compatibility)"""
        self._load_model()
