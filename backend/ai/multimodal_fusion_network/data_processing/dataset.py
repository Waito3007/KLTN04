"""
Module tạo dataset cho huấn luyện mô hình.
"""
import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Any, Tuple

from data_processing.text_processor import TextProcessor
from data_processing.metadata_processor import MetadataProcessor

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CommitDataset(Dataset):
    """Dataset cho dữ liệu commit."""
    
    def __init__(self, data_path: str, text_processor: TextProcessor, metadata_processor: MetadataProcessor):
        """
        Khởi tạo dataset.
        
        Args:
            data_path: Đường dẫn đến file dữ liệu JSON
            text_processor: Đối tượng TextProcessor đã được fit
            metadata_processor: Đối tượng MetadataProcessor đã được fit
        """
        self.data_path = data_path
        self.text_processor = text_processor
        self.metadata_processor = metadata_processor
        self.data = None
        self.task_names = []
        self._load_data()
    
    def _load_data(self) -> None:
        """Đọc dữ liệu từ file JSON."""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            # Lấy tên các task từ mẫu đầu tiên
            if self.data and 'data' in self.data and self.data['data']:
                sample = self.data['data'][0]
                if 'labels' in sample:
                    self.task_names = list(sample['labels'].keys())
            
            logger.info(f"Đã đọc {len(self.data.get('data', []))} mẫu từ {self.data_path}")
            logger.info(f"Các task: {self.task_names}")
        except Exception as e:
            logger.error(f"Lỗi khi đọc dữ liệu: {str(e)}")
            self.data = {'metadata': {}, 'data': []}
    
    def __len__(self) -> int:
        """Trả về số lượng mẫu trong dataset."""
        return len(self.data.get('data', []))
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Lấy một mẫu dữ liệu đã được xử lý.
        
        Args:
            idx: Chỉ số của mẫu
            
        Returns:
            Dict chứa các tensor cho text, metadata và labels
        """
        sample = self.data['data'][idx]
        
        # Xử lý text
        text = sample.get('text', '')
        text_tensor = torch.tensor(self.text_processor.process([text])[0], dtype=torch.long)
        
        # Xử lý metadata
        features = sample.get('features', {})
        metadata_tensor = torch.tensor(self.metadata_processor.process_single(features), dtype=torch.float)
        
        # Xử lý labels
        labels = sample.get('labels', {})
        labels_dict = {}
        for task_name in self.task_names:
            if task_name in labels:
                try:
                    labels_dict[task_name] = torch.tensor(labels[task_name], dtype=torch.float)
                except Exception as e:
                    print(f"Lỗi nhãn ở index {idx}, task {task_name}, value: {labels[task_name]}")
                    raise
            else:
                # Nếu không có nhãn, trả về vector 0 đúng chiều dài
                # Số chiều lấy từ metadata_processor hoặc label_maps nếu có
                # Tạm thời lấy chiều dài từ sample đầu tiên
                zero_len = len(self.data['data'][0]['labels'][task_name]) if self.data['data'] and task_name in self.data['data'][0]['labels'] else 1
                labels_dict[task_name] = torch.zeros(zero_len, dtype=torch.float)
        return {
            'text': text_tensor,
            'metadata': metadata_tensor,
            'labels': labels_dict,
            'raw_text': text
        }


def create_data_loaders(
    train_path: str,
    val_path: str,
    test_path: str,
    text_processor: TextProcessor,
    metadata_processor: MetadataProcessor,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo data loaders cho tập train, validation và test.
    
    Args:
        train_path: Đường dẫn đến file dữ liệu train
        val_path: Đường dẫn đến file dữ liệu validation
        test_path: Đường dẫn đến file dữ liệu test
        text_processor: Đối tượng TextProcessor đã được fit
        metadata_processor: Đối tượng MetadataProcessor đã được fit
        batch_size: Kích thước batch
        num_workers: Số lượng worker
        
    Returns:
        Tuple (train_loader, val_loader, test_loader)
    """
    # Tạo datasets
    train_dataset = CommitDataset(train_path, text_processor, metadata_processor)
    val_dataset = CommitDataset(val_path, text_processor, metadata_processor)
    test_dataset = CommitDataset(test_path, text_processor, metadata_processor)
    
    # Tạo data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Đã tạo data loaders: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Ví dụ sử dụng
    # Giả sử đã có text_processor và metadata_processor đã được fit
    text_processor = TextProcessor.load("text_processor.json")
    metadata_processor = MetadataProcessor.load("metadata_processor.json")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_path="datasets/train.json",
        val_path="datasets/val.json",
        test_path="datasets/test.json",
        text_processor=text_processor,
        metadata_processor=metadata_processor,
        batch_size=32
    )
    
    # Kiểm tra data loader
    for batch in train_loader:
        print(f"Text shape: {batch['text'].shape}")
        print(f"Metadata shape: {batch['metadata'].shape}")
        print(f"Labels: {batch['labels']}")
        break
