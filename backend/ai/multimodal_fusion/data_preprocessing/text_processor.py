"""
Text Processor for Multi-Modal Fusion Network
Xử lý và chuẩn bị dữ liệu văn bản từ commit messages
"""

import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter

# Optional transformers import
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Using simple tokenization only.")

# Optional NLTK import
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Using simple text processing.")

class TextProcessor:
    """
    Lớp xử lý văn bản cho commit messages
    Hỗ trợ cả tokenization đơn giản và pre-trained embeddings
    """
    
    def __init__(self, 
                 method: str = "lstm",  # "lstm", "distilbert", "transformer"
                 vocab_size: int = 10000,
                 max_length: int = 128,
                 pretrained_model: str = "distilbert-base-uncased"):
        """
        Args:
            method: Phương pháp xử lý ("lstm", "distilbert", "transformer")
            vocab_size: Kích thước vocabulary cho LSTM
            max_length: Độ dài tối đa của sequence
            pretrained_model: Tên pre-trained model nếu dùng transformer
        """
        self.method = method
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pretrained_model = pretrained_model
          # Initialize components based on method
        if method in ["distilbert", "transformer"] and TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
                self.model = AutoModel.from_pretrained(pretrained_model)
                self.embed_dim = self.model.config.hidden_size
            except Exception as e:
                print(f"Failed to load pretrained model: {e}. Falling back to LSTM method.")
                self.method = "lstm"
                self._init_lstm_components()
        else:
            # LSTM method
            self._init_lstm_components()
    
    def _init_lstm_components(self):
        """Initialize components for LSTM method"""
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.word_counts = Counter()
        self.embed_dim = 128  # Default embedding dimension
            
        # Download NLTK data if needed
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('punkt')
                nltk.download('stopwords')
                
            self.stop_words = set(stopwords.words('english'))
        else:
            # Simple fallback stopwords
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'])
        
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text with fallback if NLTK not available
        """
        if NLTK_AVAILABLE:
            return word_tokenize(text)
        else:
            # Simple tokenization fallback
            import string            # Remove punctuation and split by whitespace
            text = text.translate(str.maketrans('', '', string.punctuation))
            return text.split()
    
    def build_vocab(self, texts: List[str], vocab_size: int = None):
        """
        Build vocabulary from a list of texts
        """
        if vocab_size:
            self.vocab_size = vocab_size
            
        # Count words
        for text in texts:
            cleaned_text = self.clean_commit_message(text)
            tokens = self._tokenize_text(cleaned_text.lower())
            self.word_counts.update(tokens)
        
        # Build vocabulary with most common words
        most_common = self.word_counts.most_common(self.vocab_size - 4)  # Reserve space for special tokens
        
        for word, count in most_common:
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        print(f"Built vocabulary with {len(self.word_to_idx)} words")
        
    @property
    def vocab(self):
        """Return vocabulary dictionary"""
        return self.word_to_idx
    
    def clean_commit_message(self, text: str) -> str:
        """
        Làm sạch commit message
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove commit hashes (SHA)
        text = re.sub(r'\b[0-9a-fA-F]{7,40}\b', '', text)
        
        # Remove issue/PR references like #123, Fixes #456
        text = re.sub(r'(closes?|fixes?|resolves?)\s*#\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'#\d+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\:\;\-\(\)]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_commit_features(self, text: str) -> Dict[str, any]:
        """
        Trích xuất các đặc trưng từ commit message
        """
        features = {}
        
        # Basic features
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len([c for c in text if c.isalpha()])
        features['digit_count'] = len([c for c in text if c.isdigit()])
        features['upper_count'] = len([c for c in text if c.isupper()])
        
        # Commit type detection
        commit_types = ['feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore', 'perf']
        features['has_commit_type'] = any(text.lower().startswith(ct + ':') for ct in commit_types)
        
        # Extract commit type if present
        for ct in commit_types:
            if text.lower().startswith(ct + ':'):
                features['commit_type_prefix'] = ct
                break
        else:
            features['commit_type_prefix'] = 'none'
            
        # Keywords detection
        bug_keywords = ['fix', 'bug', 'error', 'issue', 'problem', 'resolve', 'solve']
        feature_keywords = ['add', 'implement', 'create', 'new', 'feature', 'support']
        doc_keywords = ['doc', 'readme', 'comment', 'documentation']
        
        features['has_bug_keywords'] = any(keyword in text.lower() for keyword in bug_keywords)
        features['has_feature_keywords'] = any(keyword in text.lower() for keyword in feature_keywords)
        features['has_doc_keywords'] = any(keyword in text.lower() for keyword in doc_keywords)
        
        # Sentiment indicators
        positive_words = ['improve', 'enhance', 'optimize', 'better', 'good', 'success']
        negative_words = ['remove', 'delete', 'deprecated', 'broken', 'fail', 'error']
        urgent_words = ['urgent', 'critical', 'hotfix', 'emergency', 'asap']
        
        features['positive_sentiment'] = any(word in text.lower() for word in positive_words)
        features['negative_sentiment'] = any(word in text.lower() for word in negative_words)
        features['urgent_sentiment'] = any(word in text.lower() for word in urgent_words)
        
        return features
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Xây dựng vocabulary cho LSTM method
        """
        if self.method != "lstm":
            return
            
        print("🔤 Building vocabulary for text processing...")
        for text in texts:
            cleaned_text = self.clean_commit_message(text)
            words = self._tokenize_text(cleaned_text.lower())
            # Filter out stop words and very short words
            words = [w for w in words if w not in self.stop_words and len(w) > 1]
            self.word_counts.update(words)
        
        # Keep most frequent words
        most_common = self.word_counts.most_common(self.vocab_size - 4)  # -4 for special tokens
        
        for i, (word, count) in enumerate(most_common):
            idx = i + 4  # Start from 4 (after special tokens)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        print(f"✅ Vocabulary built with {len(self.word_to_idx)} words")
    
    def encode_text_lstm(self, text: str) -> torch.Tensor:
        """
        Encode text cho LSTM method
        """
        cleaned_text = self.clean_commit_message(text)
        words = self._tokenize_text(cleaned_text.lower())
        words = [w for w in words if w not in self.stop_words and len(w) > 1]
        
        # Convert to indices
        indices = [self.word_to_idx.get(word, 1) for word in words]  # 1 is UNK
        
        # Add start and end tokens
        indices = [2] + indices + [3]  # 2 is START, 3 is END
        
        # Pad or truncate
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices.extend([0] * (self.max_length - len(indices)))  # 0 is PAD
        
        return torch.tensor(indices, dtype=torch.long)
    
    def encode_text_transformer(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Encode text cho transformer method
        """
        cleaned_text = self.clean_commit_message(text)
        
        encoding = self.tokenizer(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
    
    def get_text_embeddings(self, texts: List[str], device: str = 'cpu') -> torch.Tensor:
        """
        Lấy embeddings cho list of texts
        """
        if self.method == "lstm":
            # Return token indices for LSTM
            embeddings = []
            for text in texts:
                embeddings.append(self.encode_text_lstm(text))
            return torch.stack(embeddings)
        
        elif self.method in ["distilbert", "transformer"]:
            # Get contextual embeddings
            self.model.eval()
            self.model.to(device)
            
            embeddings = []
            with torch.no_grad():
                for text in texts:
                    encoding = self.encode_text_transformer(text)
                    input_ids = encoding['input_ids'].unsqueeze(0).to(device)
                    attention_mask = encoding['attention_mask'].unsqueeze(0).to(device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    # Use [CLS] token embedding or mean pooling
                    embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
                    embeddings.append(embedding.cpu())
            
            return torch.cat(embeddings, dim=0)
    
    def process_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Xử lý một batch texts
        """
        results = {
            'text_features': [],
            'embeddings': None,
            'metadata_features': []
        }
        
        # Extract text features
        for text in texts:
            features = self.extract_commit_features(text)
            results['text_features'].append(features)
        
        # Get embeddings
        if self.method == "lstm":
            embeddings = []
            for text in texts:
                embeddings.append(self.encode_text_lstm(text))
            results['embeddings'] = torch.stack(embeddings)
        else:
            results['embeddings'] = self.get_text_embeddings(texts)
        
        return results
    
    def fit(self, texts: List[str]) -> 'TextProcessor':
        """
        Fit the text processor to the training data
        This method builds vocabulary for LSTM method and prepares the processor
        """
        if self.method == "lstm":
            self.build_vocabulary(texts)
        # For transformer methods, no fitting is needed as they use pre-trained models
        return self
    
    def get_vocab_size(self) -> int:
        """
        Trả về kích thước vocabulary
        """
        if self.method == "lstm":
            return len(self.word_to_idx)
        else:
            return self.tokenizer.vocab_size
    
    def get_embedding_dim(self) -> int:
        """
        Trả về dimension của embeddings
        """
        return self.embed_dim
