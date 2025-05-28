import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Tải embedding pre-trained (CodeBERT/FastText/GloVe)
class EmbeddingLoader:
    def __init__(self, embedding_type='codebert', model_name='microsoft/codebert-base'):
        self.embedding_type = embedding_type
        self.model_name = model_name
        self.embeddings = None
        self.tokenizer = None
        self.model = None

    def load(self, path=None):
        if self.embedding_type == 'codebert':
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
        elif self.embedding_type in ['glove', 'fasttext']:
            self.embeddings = {}
            with open(path, encoding='utf-8') as f:
                for line in f:
                    values = line.strip().split()
                    word = values[0]
                    vec = np.asarray(values[1:], dtype='float32')
                    self.embeddings[word] = vec
        else:
            raise ValueError('Unknown embedding type')

    def get_word_embedding(self, word):
        if self.embedding_type == 'codebert':
            # Trả về embedding của từ từ transformers
            tokens = self.tokenizer(word, return_tensors='pt')
            with torch.no_grad():
                output = self.model(**tokens)
            return output.last_hidden_state[0, 1:-1, :].mean(dim=0).numpy()
        elif self.embedding_type in ['glove', 'fasttext']:
            return self.embeddings.get(word, np.zeros(300))
        else:
            return None
