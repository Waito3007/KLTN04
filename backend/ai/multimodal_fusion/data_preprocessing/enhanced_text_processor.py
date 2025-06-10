"""
Enhanced Text Processor with NLTK Support
Provides advanced natural language processing capabilities for commit message analysis
"""

import re
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
import string
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional transformers import
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers library available")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Using simple tokenization only.")

# Enhanced NLTK import with more features
try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.tokenize import word_tokenize, sent_tokenize, TreebankWordTokenizer
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import opinion_lexicon
    from nltk.probability import FreqDist
    from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
    from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
    NLTK_AVAILABLE = True
    logger.info("NLTK library available with advanced features")
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available. Using simple text processing.")

# Optional TextBlob for additional sentiment analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
    logger.info("TextBlob available for sentiment analysis")
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available.")


class EnhancedTextProcessor:
    """
    Enhanced Text Processor with comprehensive NLTK support
    Provides advanced NLP features for commit message analysis
    """
    
    def __init__(self, 
                 method: str = "lstm",  # "lstm", "distilbert", "transformer"
                 vocab_size: int = 10000,
                 max_length: int = 128,
                 pretrained_model: str = "distilbert-base-uncased",
                 enable_stemming: bool = True,
                 enable_lemmatization: bool = True,
                 enable_pos_tagging: bool = True,
                 enable_sentiment_analysis: bool = True,
                 enable_ngrams: bool = True):
        """
        Args:
            method: Processing method ("lstm", "distilbert", "transformer")
            vocab_size: Vocabulary size for LSTM
            max_length: Maximum sequence length
            pretrained_model: Pre-trained model name for transformers
            enable_stemming: Enable word stemming
            enable_lemmatization: Enable word lemmatization
            enable_pos_tagging: Enable part-of-speech tagging
            enable_sentiment_analysis: Enable sentiment analysis
            enable_ngrams: Enable n-gram extraction
        """
        self.method = method
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pretrained_model = pretrained_model
        
        # NLTK feature flags
        self.enable_stemming = enable_stemming
        self.enable_lemmatization = enable_lemmatization
        self.enable_pos_tagging = enable_pos_tagging
        self.enable_sentiment_analysis = enable_sentiment_analysis
        self.enable_ngrams = enable_ngrams
        
        # Initialize NLTK components
        self._init_nltk_components()
        
        # Initialize model components based on method
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
            # LSTM method
            self._init_lstm_components()
    
    def _init_nltk_components(self):
        """Initialize NLTK components and download required data"""
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available. Advanced text processing features disabled.")
            self.stop_words = self._get_basic_stopwords()
            return
        
        # Download required NLTK data
        required_data = [
            'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'maxent_ne_chunker', 'words', 'vader_lexicon', 'opinion_lexicon',
            'omw-1.4'
        ]
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{data}')
                except LookupError:
                    try:
                        nltk.data.find(f'taggers/{data}')
                    except LookupError:
                        try:
                            nltk.data.find(f'chunkers/{data}')
                        except LookupError:
                            try:
                                nltk.download(data, quiet=True)
                                logger.info(f"Downloaded NLTK data: {data}")
                            except Exception as e:
                                logger.warning(f"Failed to download {data}: {e}")
        
        # Initialize NLTK tools
        try:
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer() if self.enable_stemming else None
            self.lemmatizer = WordNetLemmatizer() if self.enable_lemmatization else None
            self.sentiment_analyzer = SentimentIntensityAnalyzer() if self.enable_sentiment_analysis else None
            self.tokenizer_nltk = TreebankWordTokenizer()
            
            # Load opinion lexicon for additional sentiment analysis
            try:
                self.positive_words = set(opinion_lexicon.positive())
                self.negative_words = set(opinion_lexicon.negative())
            except Exception as e:
                logger.warning(f"Failed to load opinion lexicon: {e}")
                self.positive_words = set()
                self.negative_words = set()
                
            logger.info("NLTK components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLTK components: {e}")
            self.stop_words = self._get_basic_stopwords()
    
    def _get_basic_stopwords(self) -> Set[str]:
        """Get basic stopwords if NLTK is not available"""
        return set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'
        ])
    
    def _init_lstm_components(self):
        """Initialize components for LSTM method"""
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.word_counts = Counter()
        self.embed_dim = 128  # Default embedding dimension
        
        # Additional LSTM-specific features
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()
        self.pos_tag_counts = Counter()
        
    def advanced_tokenize(self, text: str) -> List[str]:
        """
        Advanced tokenization with NLTK features
        """
        if not NLTK_AVAILABLE:
            # Fallback to simple tokenization
            text = text.translate(str.maketrans('', '', string.punctuation))
            return text.split()
        
        try:
            # Use TreebankWordTokenizer for better handling of contractions and punctuation
            tokens = self.tokenizer_nltk.tokenize(text)
            
            # Apply stemming or lemmatization
            if self.enable_lemmatization and self.lemmatizer:
                # Get POS tags for better lemmatization
                pos_tags = pos_tag(tokens) if self.enable_pos_tagging else [(token, 'NN') for token in tokens]
                tokens = [self._lemmatize_with_pos(token, pos) for token, pos in pos_tags]
            elif self.enable_stemming and self.stemmer:
                tokens = [self.stemmer.stem(token) for token in tokens]
            
            return tokens
            
        except Exception as e:
            logger.warning(f"Advanced tokenization failed: {e}. Using simple fallback.")
            text = text.translate(str.maketrans('', '', string.punctuation))
            return text.split()
    
    def _lemmatize_with_pos(self, word: str, pos_tag: str) -> str:
        """
        Lemmatize word with POS tag context
        """
        if not self.lemmatizer:
            return word
            
        # Convert POS tag to WordNet format
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }
        
        wordnet_pos = tag_dict.get(pos_tag[0], wordnet.NOUN)
        return self.lemmatizer.lemmatize(word, wordnet_pos)
    
    def extract_advanced_features(self, text: str) -> Dict[str, any]:
        """
        Extract advanced features using NLTK capabilities
        """
        features = {}
        
        # Basic features
        features.update(self.extract_basic_features(text))
        
        if not NLTK_AVAILABLE:
            return features
        
        try:
            # Tokenize for advanced analysis
            tokens = self.advanced_tokenize(text.lower())
            
            # Sentiment analysis
            if self.enable_sentiment_analysis and self.sentiment_analyzer:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
                features.update({
                    'sentiment_positive': sentiment_scores['pos'],
                    'sentiment_negative': sentiment_scores['neg'],
                    'sentiment_neutral': sentiment_scores['neu'],
                    'sentiment_compound': sentiment_scores['compound']
                })
                
                # TextBlob sentiment as additional feature
                if TEXTBLOB_AVAILABLE:
                    blob = TextBlob(text)
                    features['textblob_polarity'] = blob.sentiment.polarity
                    features['textblob_subjectivity'] = blob.sentiment.subjectivity
                
                # Opinion lexicon features
                positive_count = sum(1 for word in tokens if word in self.positive_words)
                negative_count = sum(1 for word in tokens if word in self.negative_words)
                features['positive_word_count'] = positive_count
                features['negative_word_count'] = negative_count
                features['sentiment_ratio'] = (positive_count - negative_count) / max(len(tokens), 1)
            
            # POS tagging features
            if self.enable_pos_tagging:
                pos_tags = pos_tag(tokens)
                pos_counts = Counter(tag for _, tag in pos_tags)
                
                # Important POS categories for commit analysis
                features['noun_count'] = pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + pos_counts.get('NNP', 0)
                features['verb_count'] = pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0)
                features['adjective_count'] = pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + pos_counts.get('JJS', 0)
                features['adverb_count'] = pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + pos_counts.get('RBS', 0)
                
                # POS diversity
                features['pos_diversity'] = len(pos_counts) / max(len(tokens), 1)
            
            # N-gram features
            if self.enable_ngrams and len(tokens) > 1:
                # Bigrams
                bigrams = list(nltk.bigrams(tokens))
                features['unique_bigrams'] = len(set(bigrams))
                features['bigram_ratio'] = len(set(bigrams)) / max(len(bigrams), 1)
                
                # Trigrams (if enough tokens)
                if len(tokens) > 2:
                    trigrams = list(nltk.trigrams(tokens))
                    features['unique_trigrams'] = len(set(trigrams))
                    features['trigram_ratio'] = len(set(trigrams)) / max(len(trigrams), 1)
                else:
                    features['unique_trigrams'] = 0
                    features['trigram_ratio'] = 0
            
            # Lexical diversity
            features['lexical_diversity'] = len(set(tokens)) / max(len(tokens), 1)
            
            # Average word length
            features['avg_word_length'] = np.mean([len(word) for word in tokens]) if tokens else 0
            
        except Exception as e:
            logger.warning(f"Advanced feature extraction failed: {e}")
        
        return features
    
    def extract_basic_features(self, text: str) -> Dict[str, any]:
        """
        Extract basic features (fallback when NLTK not available)
        """
        features = {}
        
        if not text or not isinstance(text, str):
            return {key: 0 for key in [
                'length', 'word_count', 'char_count', 'digit_count', 'upper_count',
                'has_commit_type', 'commit_type_prefix', 'has_bug_keywords',
                'has_feature_keywords', 'has_doc_keywords', 'positive_sentiment',
                'negative_sentiment', 'urgent_sentiment'
            ]}
        
        # Basic text statistics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len([c for c in text if c.isalpha()])
        features['digit_count'] = len([c for c in text if c.isdigit()])
        features['upper_count'] = len([c for c in text if c.isupper()])
        features['punctuation_count'] = len([c for c in text if c in string.punctuation])
        
        # Commit type detection
        commit_types = ['feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore', 'perf', 'ci', 'build']
        features['has_commit_type'] = any(text.lower().startswith(ct + ':') or text.lower().startswith(ct + '(') for ct in commit_types)
        
        # Extract commit type if present
        for ct in commit_types:
            if text.lower().startswith(ct + ':') or text.lower().startswith(ct + '('):
                features['commit_type_prefix'] = ct
                break
        else:
            features['commit_type_prefix'] = 'none'
        
        # Enhanced keyword detection
        bug_keywords = ['fix', 'bug', 'error', 'issue', 'problem', 'resolve', 'solve', 'correct', 'patch', 'hotfix']
        feature_keywords = ['add', 'implement', 'create', 'new', 'feature', 'support', 'introduce', 'enable']
        doc_keywords = ['doc', 'readme', 'comment', 'documentation', 'docs', 'guide', 'manual']
        refactor_keywords = ['refactor', 'restructure', 'reorganize', 'cleanup', 'optimize', 'improve']
        
        text_lower = text.lower()
        features['has_bug_keywords'] = any(keyword in text_lower for keyword in bug_keywords)
        features['has_feature_keywords'] = any(keyword in text_lower for keyword in feature_keywords)
        features['has_doc_keywords'] = any(keyword in text_lower for keyword in doc_keywords)
        features['has_refactor_keywords'] = any(keyword in text_lower for keyword in refactor_keywords)
        
        # Sentiment indicators (basic)
        positive_words = ['improve', 'enhance', 'optimize', 'better', 'good', 'success', 'complete', 'finish']
        negative_words = ['remove', 'delete', 'deprecated', 'broken', 'fail', 'error', 'disable', 'revert']
        urgent_words = ['urgent', 'critical', 'hotfix', 'emergency', 'asap', 'immediate', 'quick']
        
        features['positive_sentiment'] = any(word in text_lower for word in positive_words)
        features['negative_sentiment'] = any(word in text_lower for word in negative_words)
        features['urgent_sentiment'] = any(word in text_lower for word in urgent_words)
        
        return features
    
    def clean_commit_message(self, text: str) -> str:
        """
        Enhanced commit message cleaning
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove commit hashes (SHA)
        text = re.sub(r'\b[0-9a-fA-F]{7,40}\b', '', text)
        
        # Remove issue/PR references
        text = re.sub(r'(closes?|fixes?|resolves?|addresses?)\s*#\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'#\d+', '', text)
        
        # Remove co-authored-by lines
        text = re.sub(r'co-authored-by:.*', '', text, flags=re.IGNORECASE)
        
        # Remove merge commit patterns
        text = re.sub(r'merge (branch|pull request) .* into .*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'merge .* of .*', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Build vocabulary with advanced NLTK features
        """
        if self.method != "lstm":
            return
        
        logger.info("🔤 Building enhanced vocabulary with NLTK features...")
        
        all_tokens = []
        for text in texts:
            cleaned_text = self.clean_commit_message(text)
            tokens = self.advanced_tokenize(cleaned_text.lower())
            
            # Filter tokens
            tokens = [token for token in tokens 
                     if token not in self.stop_words 
                     and len(token) > 1 
                     and token.isalpha()]
            
            all_tokens.extend(tokens)
            self.word_counts.update(tokens)
            
            # Collect n-grams if enabled
            if self.enable_ngrams and len(tokens) > 1:
                bigrams = list(nltk.bigrams(tokens))
                self.bigram_counts.update(bigrams)
                
                if len(tokens) > 2:
                    trigrams = list(nltk.trigrams(tokens))
                    self.trigram_counts.update(trigrams)
        
        # Build vocabulary with most frequent words
        most_common = self.word_counts.most_common(self.vocab_size - 4)
        
        for i, (word, count) in enumerate(most_common):
            idx = i + 4  # Start from 4 (after special tokens)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        logger.info(f"✅ Enhanced vocabulary built with {len(self.word_to_idx)} words")
        
        # Log vocabulary statistics
        if NLTK_AVAILABLE:
            logger.info(f"📊 Total unique bigrams: {len(self.bigram_counts)}")
            logger.info(f"📊 Total unique trigrams: {len(self.trigram_counts)}")
    
    def encode_text_lstm(self, text: str) -> torch.Tensor:
        """
        Enhanced LSTM encoding with NLTK preprocessing
        """
        cleaned_text = self.clean_commit_message(text)
        tokens = self.advanced_tokenize(cleaned_text.lower())
        
        # Filter tokens
        tokens = [token for token in tokens 
                 if token not in self.stop_words 
                 and len(token) > 1 
                 and token.isalpha()]
        
        # Convert to indices
        indices = [self.word_to_idx.get(token, 1) for token in tokens]  # 1 is UNK
        
        # Add start and end tokens
        indices = [2] + indices + [3]  # 2 is START, 3 is END
        
        # Pad or truncate
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices.extend([0] * (self.max_length - len(indices)))  # 0 is PAD
        
        return torch.tensor(indices, dtype=torch.long)
    
    def get_collocations(self, texts: List[str], n: int = 10) -> Dict[str, List[Tuple]]:
        """
        Extract meaningful collocations from commit messages
        """
        if not NLTK_AVAILABLE or not self.enable_ngrams:
            return {'bigrams': [], 'trigrams': []}
        
        all_tokens = []
        for text in texts:
            cleaned_text = self.clean_commit_message(text)
            tokens = self.advanced_tokenize(cleaned_text.lower())
            tokens = [token for token in tokens 
                     if token not in self.stop_words and len(token) > 1]
            all_tokens.extend(tokens)
        
        try:
            # Bigram collocations
            bigram_finder = BigramCollocationFinder.from_words(all_tokens)
            bigram_finder.apply_freq_filter(3)  # Only bigrams appearing 3+ times
            bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, n)
            
            # Trigram collocations
            trigram_finder = TrigramCollocationFinder.from_words(all_tokens)
            trigram_finder.apply_freq_filter(2)  # Only trigrams appearing 2+ times
            trigrams = trigram_finder.nbest(TrigramAssocMeasures.chi_sq, n)
            
            return {'bigrams': bigrams, 'trigrams': trigrams}
            
        except Exception as e:
            logger.warning(f"Collocation extraction failed: {e}")
            return {'bigrams': [], 'trigrams': []}
    
    def fit(self, texts: List[str]) -> 'EnhancedTextProcessor':
        """
        Fit the enhanced text processor to training data
        """
        logger.info("🚀 Fitting enhanced text processor with NLTK features...")
        
        if self.method == "lstm":
            self.build_vocabulary(texts)
            
            # Extract and log collocations for insights
            collocations = self.get_collocations(texts)
            if collocations['bigrams']:
                logger.info(f"📈 Top bigrams: {collocations['bigrams'][:5]}")
            if collocations['trigrams']:
                logger.info(f"📈 Top trigrams: {collocations['trigrams'][:3]}")
        
        logger.info("✅ Enhanced text processor fitted successfully")
        return self
    
    # Keep all other methods from the original TextProcessor
    def encode_text_transformer(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode text for transformer method (unchanged)"""
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
        """Get text embeddings (unchanged)"""
        if self.method == "lstm":
            embeddings = []
            for text in texts:
                embeddings.append(self.encode_text_lstm(text))
            return torch.stack(embeddings)
        
        elif self.method in ["distilbert", "transformer"]:
            self.model.eval()
            self.model.to(device)
            
            embeddings = []
            with torch.no_grad():
                for text in texts:
                    encoding = self.encode_text_transformer(text)
                    input_ids = encoding['input_ids'].unsqueeze(0).to(device)
                    attention_mask = encoding['attention_mask'].unsqueeze(0).to(device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    embedding = outputs.last_hidden_state[:, 0, :]
                    embeddings.append(embedding.cpu())
            
            return torch.cat(embeddings, dim=0)
    
    def process_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Process a batch of texts with enhanced features"""
        results = {
            'text_features': [],
            'embeddings': None,
            'enhanced_features': []
        }
        
        # Extract enhanced features
        for text in texts:
            basic_features = self.extract_basic_features(text)
            if NLTK_AVAILABLE:
                enhanced_features = self.extract_advanced_features(text)
                results['enhanced_features'].append(enhanced_features)
            else:
                results['enhanced_features'].append(basic_features)
            
            results['text_features'].append(basic_features)
        
        # Get embeddings
        if self.method == "lstm":
            embeddings = []
            for text in texts:
                embeddings.append(self.encode_text_lstm(text))
            results['embeddings'] = torch.stack(embeddings)
        else:
            results['embeddings'] = self.get_text_embeddings(texts)
        
        return results
    
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
