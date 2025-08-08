# backend/ai/han_commit_analyzer.py
"""
HAN Commit Analyzer - Load and inference for HAN model
"""
import os
import torch
import torch.nn as nn
import json
from typing import Dict, Any

# Import classes from training script for model loading
try:
    from .train_han_github import SimpleHANModel, SimpleTokenizer
except ImportError:
    print("Warning: Could not import from train_han_github, using fallback classes")
    SimpleHANModel = None
    SimpleTokenizer = None

class HANCommitAnalyzer:
    def __init__(self, model_dir=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        # Cập nhật đường dẫn đến model HAN thực tế
        self.model_dir = model_dir or os.path.join(os.path.dirname(__file__), 'models', 'HanModel')
        self.model_path = os.path.join(self.model_dir, 'HAN.pth')
        self.is_loaded = False
        self.task_labels = {
            'commit_type': ['feat', 'fix', 'docs', 'test', 'refactor', 'style', 'perf', 'chore'],
            'purpose': ['Feature Implementation', 'Bug Fix', 'Documentation', 'Testing', 'Code Improvement'],
            'sentiment': ['positive', 'negative', 'neutral', 'urgent'],
            'tech_tag': ['authentication', 'database', 'api', 'security', 'ui', 'performance', 'general']
        }
        self._load_model()

    def _load_model(self):
        """Load HAN model with error handling"""
        try:
            if not os.path.exists(self.model_path):
                print(f"Model file not found: {self.model_path}")
                print(f"Please ensure HAN.pth exists at: {self.model_path}")
                self.is_loaded = False
                return
            
            print(f"Loading HAN model from: {self.model_path}")
            
            # Try loading with weights_only=True first for security
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            except Exception:
                # Fallback to weights_only=False if tokenizer/other objects are in checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract model components - cập nhật để phù hợp với format HAN.pth
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            else:
                # Nếu checkpoint chỉ chứa state_dict trực tiếp
                model_state = checkpoint
            
            # Lấy config từ checkpoint hoặc dùng config mặc định
            config = checkpoint.get('model_config', {
                'vocab_size': 10000,
                'embed_dim': 100,
                'hidden_dim': 128,
                'num_classes': {
                    'commit_type': len(self.task_labels['commit_type']),
                    'purpose': len(self.task_labels['purpose']),
                    'sentiment': len(self.task_labels['sentiment']),
                    'tech_tag': len(self.task_labels['tech_tag'])
                }
            })
            
            # Load tokenizer nếu có
            if 'tokenizer' in checkpoint:
                self.tokenizer = checkpoint['tokenizer']
            else:
                print("Warning: No tokenizer found in checkpoint, using mock tokenizer")
                self.tokenizer = self._create_mock_tokenizer()
            
            # Load label encoders nếu có
            if 'label_encoders' in checkpoint:
                self.label_encoders = checkpoint['label_encoders']
                # Cập nhật task_labels từ label_encoders
                for task, encoder in self.label_encoders.items():
                    if hasattr(encoder, 'classes_'):
                        self.task_labels[task] = list(encoder.classes_)
            
            # Build và load model
            self.model = self._build_model(config)
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            
            print("✅ HAN model loaded successfully!")
            print(f"   📊 Model config: {config}")
            print(f"   🏷️ Task labels: {list(self.task_labels.keys())}")
            print(f"   🔧 Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load HAN model: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False

    def _create_mock_tokenizer(self):
        """Create a mock tokenizer when not available in checkpoint"""
        class MockTokenizer:
            def __init__(self):
                self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
                self.vocab_size = 10000
                
            def encode_text(self, text, max_sentences=10, max_words=50):
                # Simple tokenization for mock
                words = text.lower().split()[:max_words]
                # Convert to indices (mock implementation)
                indices = [self.word_to_idx.get(word, 1) for word in words]
                # Pad to max_words
                while len(indices) < max_words:
                    indices.append(0)
                return indices[:max_words]
        
        return MockTokenizer()

    def _build_model(self, config):
        """Build HAN model architecture"""
        # Sử dụng SimpleHANModel từ training script nếu có
        if SimpleHANModel is not None:
            return SimpleHANModel(
                vocab_size=config.get('vocab_size', 10000),
                embed_dim=config.get('embed_dim', 100),
                hidden_dim=config.get('hidden_dim', 128),
                num_classes=config.get('num_classes', {
                    'commit_type': len(self.task_labels['commit_type']),
                    'purpose': len(self.task_labels['purpose']),
                    'sentiment': len(self.task_labels['sentiment']),
                    'tech_tag': len(self.task_labels['tech_tag'])
                })
            )
        else:
            # Fallback implementation
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
                config.get('num_classes', {
                    'commit_type': len(self.task_labels['commit_type']),
                    'purpose': len(self.task_labels['purpose']),
                    'sentiment': len(self.task_labels['sentiment']),
                    'tech_tag': len(self.task_labels['tech_tag'])
                })
            )

    def preprocess(self, message: str):
        """Tokenization and preprocessing using loaded tokenizer"""
        if hasattr(self, 'tokenizer') and self.tokenizer:
            # Sử dụng tokenizer từ checkpoint
            tokens = self.tokenizer.encode_text(message, max_sentences=10, max_words=50)
            return torch.tensor([tokens], dtype=torch.long).to(self.device)
        else:
            # Fallback: Simple hash-based tokenization
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
            # Preprocess input
            input_tensor = self.preprocess(message)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
            
            # Decode predictions
            predictions = {}
            for task, output in outputs.items():
                if task in self.task_labels:
                    # Get probabilities and predicted class
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    # Get label name
                    if hasattr(self, 'label_encoders') and task in self.label_encoders:
                        # Use label encoder if available
                        label = list(self.label_encoders[task].keys())[predicted_idx.item()]
                    else:
                        # Use task_labels
                        label = self.task_labels[task][predicted_idx.item()]
                    
                    predictions[task] = {
                        'label': label,
                        'confidence': confidence.item()
                    }
            
            return predictions
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return self._mock_prediction(message)

    def _mock_prediction(self, message: str) -> Dict[str, Any]:
        """Mock prediction when model is not available"""
        text = message.lower()
        
        # Simple rule-based classification for fallback
        if any(word in text for word in ['feat:', 'feature:', 'add', 'implement']):
            commit_type = {'label': 'feat', 'confidence': 0.85}
        elif any(word in text for word in ['fix:', 'bug:', 'resolve']):
            commit_type = {'label': 'fix', 'confidence': 0.85}
        elif any(word in text for word in ['docs:', 'documentation']):
            commit_type = {'label': 'docs', 'confidence': 0.85}
        else:
            commit_type = {'label': 'other', 'confidence': 0.60}
        
        return {
            'commit_type': commit_type,
            'purpose': {'label': 'Feature Implementation', 'confidence': 0.75},
            'sentiment': {'label': 'neutral', 'confidence': 0.70},
            'tech_tag': {'label': 'general', 'confidence': 0.65}
        }

    def load_model(self):
        """Public method to load model (for compatibility)"""
        return self._load_model()
