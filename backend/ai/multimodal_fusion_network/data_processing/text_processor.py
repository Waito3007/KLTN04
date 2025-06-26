"""
Module xử lý văn bản cho commit messages.
"""
import re
import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import Counter

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextProcessor:
    """Class xử lý văn bản cho commit messages."""
    
    def __init__(self, max_vocab_size: int = 10000, max_sequence_length: int = 100):
        """
        Khởi tạo text processor.
        
        Args:
            max_vocab_size: Kích thước tối đa của từ điển
            max_sequence_length: Độ dài tối đa của chuỗi sau khi tokenize
        """
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocab_size = 0
        self.is_fitted = False
    
    def clean_text(self, text: str) -> str:
        """
        Làm sạch văn bản.
        
        Args:
            text: Văn bản cần làm sạch
            
        Returns:
            Văn bản đã làm sạch
        """
        # Chuyển về chữ thường
        text = text.lower()
        
        # Loại bỏ các ký tự đặc biệt và giữ lại chữ cái, số, khoảng trắng và một số ký tự đặc biệt
        text = re.sub(r'[^\w\s\.\,\-\#\:\(\)]', ' ', text)
        
        # Loại bỏ nhiều khoảng trắng liên tiếp
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize văn bản thành danh sách từ.
        
        Args:
            text: Văn bản cần tokenize
            
        Returns:
            Danh sách các từ
        """
        # Làm sạch văn bản
        cleaned_text = self.clean_text(text)
        
        # Tách thành từng từ
        tokens = cleaned_text.split()
        
        return tokens
    
    def fit(self, texts: List[str]) -> None:
        """
        Xây dựng từ điển từ tập văn bản.
        
        Args:
            texts: Danh sách các văn bản
        """
        # Tokenize tất cả văn bản
        all_tokens = []
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
        
        # Đếm tần suất từng từ
        counter = Counter(all_tokens)
        
        # Lấy các từ phổ biến nhất
        most_common = counter.most_common(self.max_vocab_size - 2)  # Trừ 2 cho <PAD> và <UNK>
        
        # Tạo từ điển
        self.word_to_index = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in most_common:
            self.word_to_index[word] = len(self.word_to_index)
        
        # Tạo index_to_word
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}
        
        # Cập nhật kích thước từ điển
        self.vocab_size = len(self.word_to_index)
        
        self.is_fitted = True
        logger.info(f"Đã xây dựng từ điển với {self.vocab_size} từ")
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Chuyển đổi danh sách văn bản thành danh sách các chuỗi số.
        
        Args:
            texts: Danh sách các văn bản
            
        Returns:
            Danh sách các chuỗi số
        """
        if not self.is_fitted:
            raise ValueError("Text processor chưa được fit. Hãy gọi phương thức fit trước.")
        
        sequences = []
        for text in texts:
            tokens = self.tokenize(text)
            sequence = [self.word_to_index.get(token, 1) for token in tokens]  # 1 là index của <UNK>
            sequences.append(sequence)
        
        return sequences
    
    def pad_sequences(self, sequences: List[List[int]]) -> np.ndarray:
        """
        Đệm các chuỗi để có cùng độ dài.
        
        Args:
            sequences: Danh sách các chuỗi số
            
        Returns:
            Mảng numpy các chuỗi đã đệm
        """
        padded_sequences = []
        
        for sequence in sequences:
            # Cắt hoặc đệm chuỗi
            if len(sequence) > self.max_sequence_length:
                padded_sequence = sequence[:self.max_sequence_length]
            else:
                padded_sequence = sequence + [0] * (self.max_sequence_length - len(sequence))
            
            padded_sequences.append(padded_sequence)
        
        return np.array(padded_sequences)
    
    def process(self, texts: List[str]) -> np.ndarray:
        """
        Xử lý danh sách văn bản thành mảng numpy.
        
        Args:
            texts: Danh sách các văn bản
            
        Returns:
            Mảng numpy các chuỗi đã xử lý
        """
        sequences = self.texts_to_sequences(texts)
        padded_sequences = self.pad_sequences(sequences)
        
        return padded_sequences
    
    def decode_sequence(self, sequence: List[int]) -> str:
        """
        Giải mã chuỗi số thành văn bản.
        
        Args:
            sequence: Chuỗi số
            
        Returns:
            Văn bản đã giải mã
        """
        if not self.is_fitted:
            raise ValueError("Text processor chưa được fit. Hãy gọi phương thức fit trước.")
        
        words = [self.index_to_word.get(index, "<UNK>") for index in sequence if index != 0]  # Bỏ qua <PAD>
        return " ".join(words)
    
    def save(self, filepath: str) -> None:
        """
        Lưu text processor vào file.
        
        Args:
            filepath: Đường dẫn file
        """
        data = {
            'max_vocab_size': self.max_vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'word_to_index': self.word_to_index,
            'index_to_word': {int(k): v for k, v in self.index_to_word.items()},  # Convert keys to int
            'vocab_size': self.vocab_size,
            'is_fitted': self.is_fitted
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Đã lưu text processor vào {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TextProcessor':
        """
        Tải text processor từ file.
        
        Args:
            filepath: Đường dẫn file
            
        Returns:
            TextProcessor
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processor = cls(
            max_vocab_size=data['max_vocab_size'],
            max_sequence_length=data['max_sequence_length']
        )
        
        processor.word_to_index = data['word_to_index']
        processor.index_to_word = {int(k): v for k, v in data['index_to_word'].items()}  # Convert keys to int
        processor.vocab_size = data['vocab_size']
        processor.is_fitted = data['is_fitted']
        
        logger.info(f"Đã tải text processor từ {filepath}")
        return processor


if __name__ == "__main__":
    # Ví dụ sử dụng
    texts = [
        "fix: resolve bug in authentication module",
        "feat: add new login page",
        "chore: update dependencies",
        "docs: update README.md with installation instructions"
    ]
    
    processor = TextProcessor(max_vocab_size=100, max_sequence_length=10)
    processor.fit(texts)
    
    # Xử lý văn bản
    sequences = processor.process(texts)
    print(sequences.shape)
    
    # Giải mã chuỗi
    decoded = processor.decode_sequence(sequences[0])
    print(decoded)
    
    # Lưu và tải
    processor.save("text_processor.json")
    loaded_processor = TextProcessor.load("text_processor.json")
