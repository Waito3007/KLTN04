"""
NLTK Data Downloader
Downloads all required NLTK data for enhanced text processing
"""

import nltk
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_nltk_data():
    """Download all required NLTK data"""
    
    required_data = [
        'punkt',           # Sentence tokenizer models
        'stopwords',       # Stopwords corpus
        'wordnet',         # WordNet corpus for lemmatization
        'averaged_perceptron_tagger',  # POS tagger
        'maxent_ne_chunker',  # Named entity chunker
        'words',           # Word corpus
        'vader_lexicon',   # VADER sentiment analysis lexicon
        'opinion_lexicon', # Opinion lexicon for sentiment
        'omw-1.4',         # Open Multilingual Wordnet
        'brown',           # Brown corpus (for additional features)
        'reuters',         # Reuters corpus (for additional features)
        'movie_reviews',   # Movie reviews corpus (for sentiment)
    ]
    
    logger.info("🔽 Starting NLTK data download...")
    
    for data_item in required_data:
        try:
            logger.info(f"📦 Downloading {data_item}...")
            nltk.download(data_item, quiet=False)
            logger.info(f"✅ Successfully downloaded {data_item}")
        except Exception as e:
            logger.error(f"❌ Failed to download {data_item}: {e}")
    
    logger.info("🎉 NLTK data download completed!")
    
    # Test if everything works
    try:
        logger.info("🧪 Testing NLTK components...")
        
        # Test tokenization
        from nltk.tokenize import word_tokenize, sent_tokenize
        test_text = "This is a test. It should work properly!"
        tokens = word_tokenize(test_text)
        sentences = sent_tokenize(test_text)
        logger.info(f"✅ Tokenization test passed: {len(tokens)} tokens, {len(sentences)} sentences")
        
        # Test stopwords
        from nltk.corpus import stopwords
        stop_words = stopwords.words('english')
        logger.info(f"✅ Stopwords test passed: {len(stop_words)} stop words loaded")
        
        # Test POS tagging
        from nltk.tag import pos_tag
        pos_tags = pos_tag(tokens)
        logger.info(f"✅ POS tagging test passed: {len(pos_tags)} tagged tokens")
        
        # Test lemmatization
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        lemma = lemmatizer.lemmatize("running", "v")
        logger.info(f"✅ Lemmatization test passed: 'running' -> '{lemma}'")
        
        # Test sentiment analysis
        from nltk.sentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores("This is a great improvement!")
        logger.info(f"✅ Sentiment analysis test passed: {sentiment}")
        
        logger.info("🎉 All NLTK components are working correctly!")
        
    except Exception as e:
        logger.error(f"❌ NLTK component test failed: {e}")

if __name__ == "__main__":
    download_nltk_data()
