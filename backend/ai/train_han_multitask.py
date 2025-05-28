# Pipeline huấn luyện HAN đa nhiệm cho commit message
import torch
from torch.utils.data import DataLoader, Dataset
from ai.models.hierarchical_attention import HierarchicalAttentionNetwork
from ai.data_preprocessing.text_processor import TextProcessor
from ai.data_preprocessing.embedding_loader import EmbeddingLoader
from ai.training.multitask_trainer import MultiTaskTrainer
from ai.training.loss_functions import UncertaintyWeightingLoss
from ai.evaluation.metrics_calculator import calc_metrics
import numpy as np
import json
import glob
import os

# Dummy dataset cho ví dụ
class CommitDataset(Dataset):
    def __init__(self, texts, labels, processor, embed_loader):
        self.texts = texts
        self.labels = labels
        self.processor = processor
        self.embed_loader = embed_loader
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        doc = self.processor.process_document(self.texts[idx])
        # Chuyển từng từ thành embedding vector
        embed_doc = np.zeros((self.processor.max_sent_len, self.processor.max_word_len, 768))
        for i, sent in enumerate(doc):
            for j, word in enumerate(sent):
                # word là chỉ số, cần ánh xạ lại sang từ nếu dùng word2idx
                embed_doc[i, j] = self.embed_loader.get_word_embedding(str(word))
        item = {
            'input': torch.tensor(embed_doc, dtype=torch.float32),
            'labels': {k: torch.tensor(v[idx]) for k, v in self.labels.items()}
        }
        return item

def load_commit_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['text'] for item in data]
    # Danh sách nhãn mục đích commit
    purpose_keys = ['feat','fix','docs','style','refactor','chore','test','uncategorized']
    # Danh sách nhãn tech_tag (ví dụ, lấy từ các key phụ trong cats)
    tech_tag_keys = ['auth','search','cart','order','profile','product','api','ui','notification','dashboard']
    purpose_labels = []
    suspicious_labels = []
    tech_tag_labels = []
    sentiment_labels = []
    for item in data:
        cats = item['cats']
        # Mục đích commit: nhãn lớn nhất trong nhóm purpose_keys
        purpose = max(purpose_keys, key=lambda k: cats.get(k, 0))
        purpose_labels.append(purpose_keys.index(purpose))
        # Suspicious: nếu có trường 'suspicious', lấy nhãn, không thì gán 0
        suspicious_labels.append(int(cats.get('suspicious', 0)))
        # Tech tag: lấy nhãn lớn nhất trong nhóm tech_tag_keys
        tech_tag = max(tech_tag_keys, key=lambda k: cats.get(k, 0))
        tech_tag_labels.append(tech_tag_keys.index(tech_tag))
        # Sentiment: nếu có trường 'sentiment', lấy nhãn, không thì gán 1 (neutral)
        sentiment_labels.append(int(cats.get('sentiment', 1)))
    labels = {
        'purpose': purpose_labels,
        'suspicious': suspicious_labels,
        'tech_tag': tech_tag_labels,
        'sentiment': sentiment_labels
    }
    return texts, labels

def load_unified_data(data_dir):
    all_texts = []
    all_labels = { 'purpose': [], 'suspicious': [], 'tech_tag': [], 'sentiment': [] }
    all_types = []
    for file in glob.glob(os.path.join(data_dir, '*.json')):
        with open(file, 'r', encoding='utf-8') as f:
            items = json.load(f)
        for item in items:
            all_texts.append(item['raw_text'])
            all_types.append(item.get('data_type', 'unknown'))
            # Nếu có nhãn thật thì lấy, không thì gán None hoặc 0/1 phù hợp
            labels = item.get('labels', {})
            all_labels['purpose'].append(labels.get('purpose', 0) if labels.get('purpose') is not None else 0)
            all_labels['suspicious'].append(labels.get('suspicious', 0) if labels.get('suspicious') is not None else 0)
            all_labels['tech_tag'].append(labels.get('tech_tag', 0) if labels.get('tech_tag') is not None else 0)
            all_labels['sentiment'].append(labels.get('sentiment', 1) if labels.get('sentiment') is not None else 1)
    return all_texts, all_labels, all_types

def main():
    # 1. Load dữ liệu thật từ train_data.json
    texts, labels = load_commit_data('ai/train_data.json')
    processor = TextProcessor()
    embed_loader = EmbeddingLoader(embedding_type='codebert')
    embed_loader.load()
    dataset = CommitDataset(texts, labels, processor, embed_loader)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 2. Khởi tạo mô hình
    num_classes_dict = {
        'purpose': 8,  # feat, fix, docs, style, refactor, chore, test, uncategorized
        'tech_tag': 10, # auth, search, cart, order, profile, product, api, ui, notification, dashboard
        'suspicious': 2,
        'sentiment': 3
    }
    model = HierarchicalAttentionNetwork(embed_dim=768, hidden_dim=128, num_classes_dict=num_classes_dict).to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fns = {
        'purpose': torch.nn.CrossEntropyLoss(),
        'suspicious': torch.nn.CrossEntropyLoss(),
        'tech_tag': torch.nn.CrossEntropyLoss(),
        'sentiment': torch.nn.CrossEntropyLoss()
    }
    multitask_loss = UncertaintyWeightingLoss(num_tasks=4)
    trainer = MultiTaskTrainer(model, optimizer, loss_fns, device='cpu')

    # 3. Train 1 epoch (demo)
    loss = trainer.train_epoch(dataloader)
    print(f"Train loss: {loss}")

    # 4. Validate (demo)
    val_loss, preds, labels = trainer.validate(dataloader)
    print(f"Val loss: {val_loss}")
    for task in preds:
        metrics = calc_metrics(labels[task].numpy(), preds[task].argmax(-1).numpy())
        print(f"Task {task}: {metrics}")

if __name__ == "__main__":
    main()
