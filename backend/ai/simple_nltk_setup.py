"""
Simple NLTK Setup Script
Downloads essential NLTK data components one by one
"""

import sys
import subprocess

def install_package_if_needed(package):
    """Install package if not available"""
    try:
        __import__(package)
        print(f"✅ {package} is available")
        return True
    except ImportError:
        print(f"📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False

def download_nltk_essentials():
    """Download essential NLTK data"""
    
    # First, try to import nltk with minimal dependencies
    try:
        import nltk
        print("✅ NLTK imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import NLTK: {e}")
        return False
    
    # Download essential data one by one
    essential_data = [
        'punkt',           # Sentence tokenizer
        'stopwords',       # Stopwords
        'wordnet',         # WordNet
        'vader_lexicon',   # Sentiment analysis
    ]
    
    print("🔽 Downloading essential NLTK data...")
    
    for data_item in essential_data:
        try:
            print(f"📦 Downloading {data_item}...")
            nltk.download(data_item, quiet=True)
            print(f"✅ Downloaded {data_item}")
        except Exception as e:
            print(f"⚠️ Failed to download {data_item}: {e}")
    
    # Test basic functionality
    try:
        print("🧪 Testing basic NLTK functionality...")
        
        # Test tokenization
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize("Hello world! This is a test.")
        print(f"✅ Tokenization works: {len(tokens)} tokens")
        
        # Test stopwords
        try:
            from nltk.corpus import stopwords
            stop_words = stopwords.words('english')
            print(f"✅ Stopwords work: {len(stop_words)} words")
        except Exception as e:
            print(f"⚠️ Stopwords not available: {e}")
        
        # Test sentiment
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores("This is great!")
            print(f"✅ Sentiment analysis works: {sentiment}")
        except Exception as e:
            print(f"⚠️ Sentiment analysis not available: {e}")
            
        print("🎉 Basic NLTK setup completed!")
        return True
        
    except Exception as e:
        print(f"❌ NLTK testing failed: {e}")
        return False

if __name__ == "__main__":
    # Install TextBlob separately
    install_package_if_needed("textblob")
    
    # Setup NLTK
    download_nltk_essentials()
