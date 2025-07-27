import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

class CommitDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128):
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.scaler = StandardScaler()
        num_features = [self._get_num_features(x) for x in self.data]
        self.scaler.fit(num_features)
    def _get_num_features(self, item):
        return [
            int(item.get('files_count', 0)),
            int(item.get('lines_added', 0)),
            int(item.get('lines_removed', 0)),
            int(item.get('total_changes', 0))
        ]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        text = (item.get('commit_message', '') + ' ' + item.get('diff_content', ''))[:512]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        num_feat = self._get_num_features(item)
        num_feat = self.scaler.transform([num_feat])[0]
        label = 1 if item['risk'] == 'highrisk' else 0
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'num_feat': torch.tensor(num_feat, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

class MultiFusionModel(nn.Module):
    def __init__(self, bert_name='distilbert-base-uncased', num_num_features=4, hidden_size=128):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(bert_name)
        self.mlp = nn.Sequential(
            nn.Linear(num_num_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.classifier = nn.Linear(self.bert.config.hidden_size + hidden_size, 2)
    def forward(self, input_ids, attention_mask, num_feat):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_feat = bert_out.last_hidden_state[:, 0]
        mlp_feat = self.mlp(num_feat)
        fusion = torch.cat([bert_feat, mlp_feat], dim=1)
        logits = self.classifier(fusion)
        return logits

def train_model(train_path, val_path, test_path, epochs=10, batch_size=8, lr=1e-5):
    from tqdm import tqdm
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_ds = CommitDataset(train_path, tokenizer)
    val_ds = CommitDataset(val_path, tokenizer)
    test_ds = CommitDataset(test_path, tokenizer)
    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = MultiFusionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_macro_f1 = 0.0
    best_model_path = 'risk_model.pt'
    patience = 3
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Train Epoch {epoch+1}'):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            num_feat = batch['num_feat'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, attention_mask, num_feat)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1} train loss: {total_loss/len(train_loader):.4f}')
        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Val Epoch {epoch+1}'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                num_feat = batch['num_feat'].to(device)
                labels = batch['label'].to(device)
                logits = model(input_ids, attention_mask, num_feat)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(pred)
                trues.extend(labels.cpu().numpy())
        report = classification_report(trues, preds, target_names=['lowrisk', 'highrisk'], output_dict=True)
        macro_f1 = report['macro avg']['f1-score']
        print(classification_report(trues, preds, target_names=['lowrisk', 'highrisk']))
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved at epoch {epoch+1} with macro F1: {macro_f1:.4f}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'No improvement in macro F1 for {epochs_no_improve} epoch(s).')
        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1} (no macro F1 improvement in {patience} epochs).')
            break
    # Đánh giá trên test set
    print('Evaluating on test set...')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Test'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            num_feat = batch['num_feat'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, attention_mask, num_feat)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(labels.cpu().numpy())
    print('Test set results:')
    print(classification_report(trues, preds, target_names=['lowrisk', 'highrisk']))

if __name__ == '__main__':
    train_model(
        train_path=r'train.jsonl',
        val_path=r'val.jsonl',
        test_path=r'test.jsonl',
        epochs=10,
        batch_size=8,
        lr=1e-5
    )
