import json
import torch
from torch.utils.data import Dataset, DataLoader

class CommitFusionDataset(Dataset):
    def __init__(self, json_path, text_processor, metadata_processor, label_name=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Nếu có metadata, lấy data['data'], nếu không thì data là list
        self.samples = data['data'] if isinstance(data, dict) and 'data' in data else data
        self.text_processor = text_processor
        self.metadata_processor = metadata_processor
        self.label_name = label_name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item['text']
        metadata = item['metadata']
        # Xử lý text và metadata
        text_tensor = self.text_processor.encode(text)  # Trả về tensor (token ids)
        metadata_tensor = self.metadata_processor.encode(metadata)  # Trả về tensor (vector đặc trưng)
        # Xử lý label nếu có
        if self.label_name and 'labels' in item:
            label = item['labels'][self.label_name]
            label_tensor = torch.tensor(label, dtype=torch.long)
            return text_tensor, metadata_tensor, label_tensor
        return text_tensor, metadata_tensor