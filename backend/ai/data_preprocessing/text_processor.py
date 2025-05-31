import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK resources
resources = ['punkt', 'averaged_perceptron_tagger', 'stopwords']
for resource in resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Xử lý văn bản: làm sạch, tách câu, token hóa, padding
class TextProcessor:
    def __init__(self, max_sent_len=30, max_word_len=20):
        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len

    def clean_text(self, text):
        text = text.lower()
        # Giữ lại chữ, số, dấu cách, dấu chấm, dấu gạch dưới, dấu ngoặc, dấu =, :, ;, #, @, /, \, ', ", <, >, -, +, *, %, &, |
        text = re.sub(r'[^\w\s\.\_\(\)\[\]\{\}\=\:\;\#\@\/\\\'\"\<\>\-\+\*\%\&\|]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def sent_tokenize(self, text):
        return sent_tokenize(text)

    def word_tokenize(self, sentence):
        return word_tokenize(sentence)

    def pad_sequences(self, sequences, maxlen, pad_value=0):
        padded = np.full((len(sequences), maxlen), pad_value)
        for i, seq in enumerate(sequences):
            trunc = seq[:maxlen]
            padded[i, :len(trunc)] = trunc
        return padded

    def chunk_text(self, text, max_tokens=500):
        """Chia văn bản dài thành các đoạn nhỏ hơn theo số lượng từ."""
        words = self.word_tokenize(text)
        chunks = []
        for i in range(0, len(words), max_tokens):
            chunk = ' '.join(words[i:i+max_tokens])
            chunks.append(chunk)
        return chunks

    def process_document(self, text, word2idx=None):
        text = self.clean_text(text)
        sents = self.sent_tokenize(text)[:self.max_sent_len]
        doc = []
        for sent in sents:
            words = self.word_tokenize(sent)[:self.max_word_len]
            if word2idx:
                words = [word2idx.get(w, word2idx.get('<unk>', 0)) for w in words]
            doc.append(words)
        # Padding từng câu
        doc = [w + [0]*(self.max_word_len-len(w)) if len(w)<self.max_word_len else w for w in doc]
        # Padding số câu
        if len(doc) < self.max_sent_len:
            doc += [[0]*self.max_word_len]*(self.max_sent_len-len(doc))
        return np.array(doc)
