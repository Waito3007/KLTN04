"""
Minimal Enhanced Text Processor
Provides basic NLTK functionality without complex dependencies
"""

import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import string
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try minimal NLTK imports
try:
    # Only import basic tokenization without sklearn dependencies
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import nltk
    
    # Download only essential data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    NLTK_BASIC = True
    logger.info("Basic NLTK functionality available")
except Exception as e:
    NLTK_BASIC = False
    logger.warning(f"NLTK basic features not available: {e}")

# Try TextBlob for sentiment
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
    logger.info("TextBlob available for sentiment analysis")
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available")

# Optional transformers import
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")


class MinimalEnhancedTextProcessor:
    """
    Minimal Enhanced Text Processor with basic NLTK support
    Focuses on essential improvements without complex dependencies
    """
    
    def __init__(self, 
                 method: str = "lstm",
                 vocab_size: int = 10000,
                 max_length: int = 128,
                 pretrained_model: str = "distilbert-base-uncased",
                 enable_sentiment: bool = True,
                 enable_advanced_cleaning: bool = True):
        """
        Args:
            method: Processing method ("lstm", "distilbert", "transformer")
            vocab_size: Vocabulary size for LSTM
            max_length: Maximum sequence length
            pretrained_model: Pre-trained model name
            enable_sentiment: Enable sentiment analysis with TextBlob
            enable_advanced_cleaning: Enable advanced text cleaning
        """
        self.method = method
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pretrained_model = pretrained_model
        self.enable_sentiment = enable_sentiment
        self.enable_advanced_cleaning = enable_advanced_cleaning
        
        # Initialize stopwords
        self._init_stopwords()
        
        # Initialize model components
        if method in ["distilbert", "transformer"] and TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
                self.model = AutoModel.from_pretrained(pretrained_model)
                self.embed_dim = self.model.config.hidden_size
                logger.info(f"Initialized {pretrained_model} with embedding dimension {self.embed_dim}")
            except Exception as e:
                logger.error(f"Failed to load pretrained model: {e}. Falling back to LSTM method.")
                self.method = "lstm"
                self._init_lstm_components()
        else:
            self._init_lstm_components()
    
    def _init_stopwords(self):
        """Initialize stopwords with fallback"""
        if NLTK_BASIC:
            try:
                self.stop_words = set(stopwords.words('english'))
                logger.info(f"Loaded {len(self.stop_words)} NLTK stopwords")
            except Exception as e:
                logger.warning(f"Failed to load NLTK stopwords: {e}")
                self.stop_words = self._get_basic_stopwords()
        else:
            self.stop_words = self._get_basic_stopwords()
    
    def _get_basic_stopwords(self) -> Set[str]:
        """Basic stopwords list"""
        return set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours'
        ])
    
    def _init_lstm_components(self):
        """Initialize LSTM components"""
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.word_counts = Counter()
        self.embed_dim = 128
    
    def enhanced_tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization with NLTK if available"""
        if NLTK_BASIC:
            try:
                tokens = word_tokenize(text.lower())
                return [token for token in tokens if token.isalpha() and len(token) > 1]
            except Exception as e:
                logger.warning(f"NLTK tokenization failed: {e}")
        
        # Fallback to simple tokenization
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return [word for word in text.split() if len(word) > 1]
    
    def advanced_clean_commit_message(self, text: str) -> str:
        """Advanced commit message cleaning"""
        if not text or not isinstance(text, str):
            return ""
        
        original_text = text
        
        # Basic cleaning
        text = self.clean_commit_message(text)
        
        if not self.enable_advanced_cleaning:
            return text
        
        # Advanced cleaning patterns
        advanced_patterns = [
            # Remove version numbers
            (r'\bv?\d+\.\d+(\.\d+)?(-\w+)?\b', ''),
            # Remove file extensions in isolation
            (r'\b\w+\.(js|py|html|css|md|txt|json|xml|yml|yaml)\b', ''),
            # Remove common dev terms that add noise
            (r'\b(eslint|prettier|webpack|babel|npm|yarn|pip)\b', ''),
            # Remove brackets with single words
            (r'\[\w+\]', ''),
            # Remove parentheses with single words
            (r'\(\w+\)', ''),
            # Clean up multiple spaces and special chars
            (r'[^\w\s\.\!\?\,\:\;\-]', ' '),
            (r'\s+', ' '),
        ]
        
        for pattern, replacement in advanced_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        text = text.strip()
        
        # If cleaning removed too much, return original cleaned version
        if len(text) < len(original_text) * 0.3:
            return self.clean_commit_message(original_text)
        
        return text
    
    def clean_commit_message(self, text: str) -> str:
        """Basic commit message cleaning"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove commit hashes
        text = re.sub(r'\b[0-9a-fA-F]{7,40}\b', '', text)
        
        # Remove issue/PR references
        text = re.sub(r'(closes?|fixes?|resolves?|addresses?)\s*#\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'#\d+', '', text)
        
        # Remove co-authored-by lines
        text = re.sub(r'co-authored-by:.*', '', text, flags=re.IGNORECASE)
        
        # Remove merge commit patterns
        text = re.sub(r'merge (branch|pull request) .* into .*', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_enhanced_features(self, text: str) -> Dict[str, any]:
        """Extract enhanced features with sentiment analysis"""
        features = self.extract_basic_features(text)
        
        if not text or not isinstance(text, str):
            return features
        
        # Add sentiment analysis if available
        if self.enable_sentiment and TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                features['sentiment_polarity'] = blob.sentiment.polarity
                features['sentiment_subjectivity'] = blob.sentiment.subjectivity
                
                # Categorize sentiment
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    features['sentiment_category'] = 'positive'
                elif polarity < -0.1:
                    features['sentiment_category'] = 'negative'
                else:
                    features['sentiment_category'] = 'neutral'
                    
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                features['sentiment_polarity'] = 0.0
                features['sentiment_subjectivity'] = 0.0
                features['sentiment_category'] = 'neutral'
        
        # Enhanced text statistics
        words = text.split()
        if words:
            features['avg_word_length'] = np.mean([len(word) for word in words])
            features['max_word_length'] = max(len(word) for word in words)
            features['unique_word_ratio'] = len(set(words)) / len(words)
        else:
            features['avg_word_length'] = 0
            features['max_word_length'] = 0
            features['unique_word_ratio'] = 0
        
        # Enhanced keyword detection
        technical_keywords = ['api', 'database', 'server', 'client', 'config', 'auth', 'security', 'performance']
        ui_keywords = ['ui', 'interface', 'design', 'layout', 'style', 'theme', 'responsive']
        testing_keywords = ['test', 'spec', 'mock', 'coverage', 'unit', 'integration', 'e2e']
        
        text_lower = text.lower()
        features['has_technical_keywords'] = any(kw in text_lower for kw in technical_keywords)
        features['has_ui_keywords'] = any(kw in text_lower for kw in ui_keywords)
        features['has_testing_keywords'] = any(kw in text_lower for kw in testing_keywords)
        
        return features
    
    def extract_basic_features(self, text: str) -> Dict[str, any]:
        """Extract basic text features"""
        features = {}
        
        if not text or not isinstance(text, str):
            return {key: 0 for key in [
                'length', 'word_count', 'char_count', 'digit_count', 'upper_count',
                'punctuation_count', 'has_commit_type', 'commit_type_prefix',
                'has_bug_keywords', 'has_feature_keywords', 'has_doc_keywords'
            ]}
        
        # Basic statistics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len([c for c in text if c.isalpha()])
        features['digit_count'] = len([c for c in text if c.isdigit()])
        features['upper_count'] = len([c for c in text if c.isupper()])
        features['punctuation_count'] = len([c for c in text if c in string.punctuation])
        
        # Commit type detection
        commit_types = ['feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore', 'perf', 'ci', 'build']
        text_lower = text.lower()
        features['has_commit_type'] = any(text_lower.startswith(ct + ':') or text_lower.startswith(ct + '(') for ct in commit_types)
        
        # Extract commit type
        for ct in commit_types:
            if text_lower.startswith(ct + ':') or text_lower.startswith(ct + '('):
                features['commit_type_prefix'] = ct
                break
        else:
            features['commit_type_prefix'] = 'none'
        
        # Keyword detection
        bug_keywords = ['fix', 'bug', 'error', 'issue', 'problem', 'resolve', 'patch', 'hotfix']
        feature_keywords = ['add', 'implement', 'create', 'new', 'feature', 'support', 'introduce']
        doc_keywords = ['doc', 'readme', 'comment', 'documentation', 'guide']
        
        features['has_bug_keywords'] = any(kw in text_lower for kw in bug_keywords)
        features['has_feature_keywords'] = any(kw in text_lower for kw in feature_keywords)
        features['has_doc_keywords'] = any(kw in text_lower for kw in doc_keywords)
        
        return features
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary for LSTM method"""
        if self.method != "lstm":
            return
        
        logger.info("🔤 Building enhanced vocabulary...")
        
        for text in texts:
            cleaned_text = self.advanced_clean_commit_message(text)
            tokens = self.enhanced_tokenize(cleaned_text)
            tokens = [token for token in tokens if token not in self.stop_words]
            self.word_counts.update(tokens)
        
        # Build vocabulary
        most_common = self.word_counts.most_common(self.vocab_size - 4)
        
        for i, (word, count) in enumerate(most_common):
            idx = i + 4
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        logger.info(f"✅ Enhanced vocabulary built with {len(self.word_to_idx)} words")
    
    def encode_text_lstm(self, text: str) -> torch.Tensor:
        """Enhanced LSTM encoding"""
        cleaned_text = self.advanced_clean_commit_message(text)
        tokens = self.enhanced_tokenize(cleaned_text)
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Convert to indices
        indices = [self.word_to_idx.get(token, 1) for token in tokens]
        
        # Add start and end tokens
        indices = [2] + indices + [3]
        
        # Pad or truncate
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices.extend([0] * (self.max_length - len(indices)))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def fit(self, texts: List[str]) -> 'MinimalEnhancedTextProcessor':
        """Fit the processor to training data"""
        logger.info("🚀 Fitting minimal enhanced text processor...")
        
        if self.method == "lstm":
            self.build_vocabulary(texts)
        
        logger.info("✅ Text processor fitted successfully")
        return self
    
    def process_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Process a batch of texts"""
        results = {
            'text_features': [],
            'embeddings': None,
            'enhanced_features': []
        }
        
        for text in texts:
            basic_features = self.extract_basic_features(text)
            enhanced_features = self.extract_enhanced_features(text)
            
            results['text_features'].append(basic_features)
            results['enhanced_features'].append(enhanced_features)
        
        # Get embeddings
        if self.method == "lstm":
            embeddings = []
            for text in texts:
                embeddings.append(self.encode_text_lstm(text))
            results['embeddings'] = torch.stack(embeddings)
        
        return results
    
    # Keep essential methods from original processor
    def encode_text_transformer(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode text for transformer method"""
        cleaned_text = self.advanced_clean_commit_message(text)
        
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
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size"""
        if self.method == "lstm":
            return len(self.word_to_idx)
        else:
            return self.tokenizer.vocab_size
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimension"""
        return self.embed_dim

    @property
    def vocab(self):
        """Return vocabulary dictionary"""
        return self.word_to_idx
