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
    def __init__(self, samples, processor, embed_loader):
        self.samples = samples
        self.processor = processor
        self.embed_loader = embed_loader
        # Chuẩn hóa ánh xạ nhãn cho từng task
        self.purpose_map = {
            'Feature Implementation': 0,
            'Bug Fix': 1,
            'Refactoring': 2,
            'Documentation Update': 3,
            'Test Update': 4,
            'Security Patch': 5,
            'Code Style/Formatting': 6,
            'Build/CI/CD Script Update': 7,
            'Other': 8
        }
        self.sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
        self.tech_vocab = [
            'python', 'fastapi', 'react', 'javascript', 'typescript', 'docker', 'sqlalchemy', 'pytorch', 'spacy',
            'css', 'html', 'postgresql', 'mysql', 'mongodb', 'redis', 'vue', 'angular', 'flask', 'django',
            'node', 'express', 'graphql', 'rest', 'api', 'gitlab', 'github', 'ci', 'cd', 'kubernetes', 'helm',
            'pytest', 'unittest', 'junit', 'cicd', 'github actions', 'travis', 'jenkins', 'circleci', 'webpack',
            'babel', 'vite', 'npm', 'yarn', 'pip', 'poetry', 'black', 'flake8', 'isort', 'prettier', 'eslint',
            'jwt', 'oauth', 'sso', 'celery', 'rabbitmq', 'kafka', 'grpc', 'protobuf', 'swagger', 'openapi',
            'sentry', 'prometheus', 'grafana', 'nginx', 'apache', 'linux', 'ubuntu', 'windows', 'macos',
            'aws', 'azure', 'gcp', 'firebase', 'heroku', 'netlify', 'vercel', 'tailwind', 'bootstrap', 'material ui'
        ]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        doc = self.processor.process_document(sample['raw_text'])
        embed_doc = np.zeros((self.processor.max_sent_len, self.processor.max_word_len, 768))
        for i, sent in enumerate(doc):
            for j, word in enumerate(sent):
                embed_doc[i, j] = self.embed_loader.get_word_embedding(str(word))
        l = sample['labels']
        # purpose
        purpose = self.purpose_map.get(l.get('purpose', 'Other'), 8)
        # suspicious
        suspicious = int(l.get('suspicious', 0))
        # tech_tag
        tech_tags = l.get('tech_tag', [])
        if isinstance(tech_tags, list) and len(tech_tags) > 0:
            tech_idx = self.tech_vocab.index(tech_tags[0]) if techTags[0] in self.tech_vocab else 0
        else:
            tech_idx = 0
        # sentiment
        sentiment = self.sentiment_map.get(l.get('sentiment', 'neutral'), 1)
        labels_tensor_dict = {
            'purpose': torch.tensor(purpose),
            'suspicious': torch.tensor(suspicious),
            'tech_tag': torch.tensor(tech_idx),
            'sentiment': torch.tensor(sentiment)
        }
        return {
            'input': torch.tensor(embed_doc, dtype=torch.float32),
            'labels': labels_tensor_dict
        }

def load_commit_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Chuẩn hóa ánh xạ nhãn cho từng task
    purpose_map = {
        'Feature Implementation': 0,
        'Bug Fix': 1,
        'Refactoring': 2,
        'Documentation Update': 3,
        'Test Update': 4,
        'Security Patch': 5,
        'Code Style/Formatting': 6,
        'Build/CI/CD Script Update': 7,
        'Other': 8
    }
    sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    texts = []
    labels = {'purpose': [], 'suspicious': [], 'tech_tag': [], 'sentiment': []}
    for item in data:
        texts.append(item['raw_text'])
        l = item['labels']
        # purpose
        labels['purpose'].append(purpose_map.get(l.get('purpose', 'Other'), 8))
        # suspicious
        labels['suspicious'].append(int(l.get('suspicious', 0)))
        # tech_tag: lấy index của tag đầu tiên nếu có, không thì 0
        tech_tags = l.get('tech_tag', [])
        if isinstance(tech_tags, list) and len(tech_tags) > 0:
            # Định nghĩa vocab cho tech_tag
            tech_vocab = [
                'python', 'fastapi', 'react', 'javascript', 'typescript', 'docker', 'sqlalchemy', 'pytorch', 'spacy',
                'css', 'html', 'postgresql', 'mysql', 'mongodb', 'redis', 'vue', 'angular', 'flask', 'django',
                'node', 'express', 'graphql', 'rest', 'api', 'gitlab', 'github', 'ci', 'cd', 'kubernetes', 'helm',
                'pytest', 'unittest', 'junit', 'cicd', 'github actions', 'travis', 'jenkins', 'circleci', 'webpack',
                'babel', 'vite', 'npm', 'yarn', 'pip', 'poetry', 'black', 'flake8', 'isort', 'prettier', 'eslint',
                'jwt', 'oauth', 'sso', 'celery', 'rabbitmq', 'kafka', 'grpc', 'protobuf', 'swagger', 'openapi',
                'sentry', 'prometheus', 'grafana', 'nginx', 'apache', 'linux', 'ubuntu', 'windows', 'macos',
                'aws', 'azure', 'gcp', 'firebase', 'heroku', 'netlify', 'vercel', 'tailwind', 'bootstrap', 'material ui'
            ]
            # Lấy index của tag đầu tiên tìm thấy trong vocab, nếu không có thì 0
            idx = tech_vocab.index(tech_tags[0]) if tech_tags[0] in tech_vocab else 0
            labels['tech_tag'].append(idx)
        else:
            labels['tech_tag'].append(0)
        # sentiment
        labels['sentiment'].append(sentiment_map.get(l.get('sentiment', 'neutral'), 1))
    return texts, labels

def load_unified_data(data_dir, filter_type=None):
    all_texts = []
    all_labels = { 'purpose': [], 'suspicious': [], 'tech_tag': [], 'sentiment': [] }
    all_types = []
    for file in os.listdir(data_dir):
        if not file.endswith('.json'):
            continue
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
            items = json.load(f)
        for item in items:
            if filter_type and item.get('data_type') != filter_type:
                continue
            all_texts.append(item['raw_text'])
            all_types.append(item.get('data_type', 'unknown'))
            labels = item.get('labels', {})
            all_labels['purpose'].append(labels.get('purpose', 0) if labels.get('purpose') is not None else 0)
            all_labels['suspicious'].append(labels.get('suspicious', 0) if labels.get('suspicious') is not None else 0)
            all_labels['tech_tag'].append(labels.get('tech_tag', 0) if labels.get('tech_tag') is not None else 0)
            all_labels['sentiment'].append(labels.get('sentiment', 1) if labels.get('sentiment') is not None else 1)
    return all_texts, all_labels, all_types

def main():
    # 1. Load dữ liệu đã gán nhãn từ training_data/han_training_samples.json
    # Sửa đường dẫn tuyệt đối dựa trên vị trí file script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "training_data", "han_training_samples.json")
    if not os.path.exists(json_path):
        # fallback: thử đường dẫn cũ
        json_path = os.path.join(base_dir, "../training_data/han_training_samples.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu huấn luyện: {json_path}")
    texts, labels = load_commit_data(json_path)
    processor = TextProcessor()
    embed_loader = EmbeddingLoader(embedding_type='codebert')
    embed_loader.load()
    # Đảm bảo truyền đúng tham số: samples là list các dict có 'raw_text' và 'labels'
    samples = []
    for i in range(len(texts)):
        samples.append({'raw_text': texts[i], 'labels': {
            'purpose': labels['purpose'][i],
            'suspicious': labels['suspicious'][i],
            'tech_tag': labels['tech_tag'][i],
            'sentiment': labels['sentiment'][i]
        }})
    dataset = CommitDataset(samples, processor, embed_loader)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 2. Khởi tạo mô hình
    num_classes_dict = {
        'purpose': 9,  # feat, fix, docs, style, refactor, chore, test, uncategorized, other
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
