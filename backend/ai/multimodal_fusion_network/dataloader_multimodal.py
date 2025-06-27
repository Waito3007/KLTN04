import json
import torch
from torch.utils.data import Dataset, DataLoader
# import tensorflow as tf
import numpy as np

# PyTorch Dataset
class MultimodalCommitDataset(Dataset):
    def __init__(self, json_file, text_key='text', label_key='labels', transform=None):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)['data']
        self.text_key = text_key
        self.label_key = label_key
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get(self.text_key, "")
        labels = item.get(self.label_key, {})
        meta = item.get('metadata', {})
        # Tùy chỉnh tiền xử lý ở đây
        if self.transform:
            text = self.transform(text)
        return {
            'text': text,
            'labels': labels,
            'meta': meta
        }

# TensorFlow Dataset
# class MultimodalCommitTFDataset:
#     def __init__(self, json_file, text_key='text', label_key='labels'):
#         with open(json_file, 'r', encoding='utf-8') as f:
#             self.data = json.load(f)['data']
#         self.text_key = text_key
#         self.label_key = label_key

#     def generator(self):
#         for item in self.data:
#             text = item.get(self.text_key, "")
#             labels = item.get(self.label_key, {})
#             meta = item.get('metadata', {})
#             yield text, labels, meta

#     def get_tf_dataset(self):
#         return tf.data.Dataset.from_generator(
#             self.generator,
#             output_signature=(
#                 tf.TensorSpec(shape=(), dtype=tf.string),
#                 tf.TensorSpec(shape=(), dtype=tf.string),  # labels as JSON string
#                 tf.TensorSpec(shape=(), dtype=tf.string),  # meta as JSON string
#             )
#         )
