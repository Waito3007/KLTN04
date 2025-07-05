import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình
DATA_DIR = r"c:\SAN\KLTN\KLTN04\ai\model\multifusion\data"
MODEL_SAVE_PATH = r"c:\SAN\KLTN\KLTN04\ai\model\multifusion\multifusionV2.pth"
BERT_MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

# 1. Định nghĩa Dataset
class CommitDataset(Dataset):
    def __init__(self, data, tokenizer, label_encoder_lang, label_encoder_type, scaler):
        self.data = data
        self.tokenizer = tokenizer
        self.label_encoder_lang = label_encoder_lang
        self.label_encoder_type = label_encoder_type
        self.scaler = scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        commit = self.data[idx]

        # Text features (commit_message)
        message = commit.get("commit_message", "")
        encoding = self.tokenizer.encode_plus(
            message,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Numerical features
        lines_added = commit.get("lines_added", 0)
        lines_removed = commit.get("lines_removed", 0)
        files_count = commit.get("files_count", 0)
        total_changes = lines_added + lines_removed
        
        # Handle division by zero for ratio
        if lines_removed == 0:
            ratio_added_removed = lines_added # If no lines removed, ratio is just added lines
        else:
            ratio_added_removed = lines_added / lines_removed

        numerical_features = torch.tensor([
            lines_added, lines_removed, files_count, total_changes, ratio_added_removed
        ], dtype=torch.float32)
        
        # Scale numerical features
        numerical_features_scaled = torch.tensor(self.scaler.transform(numerical_features.reshape(1, -1)).flatten(), dtype=torch.float32)

        # Categorical features (detected_language)
        lang = commit["metadata"].get("detected_language", "unknown_language")
        lang_encoded = self.label_encoder_lang.transform([lang])[0]
        lang_one_hot = torch.zeros(len(self.label_encoder_lang.classes_))
        lang_one_hot[lang_encoded] = 1

        # Labels (commit_type)
        commit_type = commit.get("commit_type", "other_type").lower() # Ensure consistency
        label = self.label_encoder_type.transform([commit_type])[0]

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'numerical_features': numerical_features_scaled,
            'language_one_hot': lang_one_hot,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. Định nghĩa Mô hình Multifusion
class MultiFusionModel(nn.Module):
    def __init__(self, bert_model_name, num_numerical_features, num_language_features, num_classes):
        super(MultiFusionModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)

        # MLP for numerical and language features
        self.mlp = nn.Sequential(
            nn.Linear(num_numerical_features + num_language_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Fusion layer
        self.classifier = nn.Linear(self.bert.config.hidden_size + 64, num_classes)

    def forward(self, input_ids, attention_mask, numerical_features, language_one_hot):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.last_hidden_state[:, 0, :]
        
        # Process numerical and language features
        combined_structured_features = torch.cat((numerical_features, language_one_hot), dim=1)
        mlp_output = self.mlp(combined_structured_features)

        # Concatenate outputs from both branches
        combined_features = torch.cat((pooled_output, mlp_output), dim=1)
        
        return self.classifier(self.dropout(combined_features))

# 3. Tải và tiền xử lý dữ liệu
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    train_data = load_data(os.path.join(DATA_DIR, "train_commits_balanced_multi.json"))
    val_data = load_data(os.path.join(DATA_DIR, "val_commits_balanced_multi.json"))

    # Tiền xử lý nhãn (Label Encoding)
    all_languages = [commit["metadata"].get("detected_language", "unknown_language") for commit in train_data + val_data]
    label_encoder_lang = LabelEncoder()
    label_encoder_lang.fit(all_languages)

    all_commit_types = [commit.get("commit_type", "other_type").lower() for commit in train_data + val_data]
    label_encoder_type = LabelEncoder()
    label_encoder_type.fit(all_commit_types)

    num_classes = len(label_encoder_type.classes_)
    num_language_features = len(label_encoder_lang.classes_)

    # Tiền xử lý đặc trưng số (Scaling)
    # Collect all numerical features first to fit the scaler
    all_numerical_features = []
    for commit in train_data + val_data:
        lines_added = commit.get("lines_added", 0)
        lines_removed = commit.get("lines_removed", 0)
        files_count = commit.get("files_count", 0)
        total_changes = lines_added + lines_removed
        if lines_removed == 0:
            ratio_added_removed = lines_added
        else:
            ratio_added_removed = lines_added / lines_removed
        all_numerical_features.append([lines_added, lines_removed, files_count, total_changes, ratio_added_removed])
    
    scaler = StandardScaler()
    scaler.fit(all_numerical_features)
    num_numerical_features = len(all_numerical_features[0])

    # Khởi tạo Tokenizer và Dataset
    tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)
    train_dataset = CommitDataset(train_data, tokenizer, label_encoder_lang, label_encoder_type, scaler)
    val_dataset = CommitDataset(val_data, tokenizer, label_encoder_lang, label_encoder_type, scaler)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Khởi tạo Mô hình, Optimizer, Loss Function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    model = MultiFusionModel(BERT_MODEL_NAME, num_numerical_features, num_language_features, num_classes)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 4. Huấn luyện Mô hình
    print("\n--- Bắt đầu huấn luyện mô hình ---")
    best_val_loss = float('inf')


    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch + 1}/{EPOCHS} ==========")
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            language_one_hot = batch['language_one_hot'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, numerical_features, language_one_hot)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            # In tiến trình mỗi 50 batch
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_dataloader):
                print(f"[Epoch {epoch + 1}/{EPOCHS}] Batch {batch_idx + 1}/{len(train_dataloader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"[Epoch {epoch + 1}] Average Train Loss: {avg_train_loss:.4f}")

        # Đánh giá trên tập xác thực
        print(f"[Epoch {epoch + 1}] Đánh giá trên tập xác thực...")
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for val_batch_idx, batch in enumerate(val_dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                numerical_features = batch['numerical_features'].to(device)
                language_one_hot = batch['language_one_hot'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask, numerical_features, language_one_hot)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # In tiến trình mỗi 20 batch validation
                if (val_batch_idx + 1) % 20 == 0 or (val_batch_idx + 1) == len(val_dataloader):
                    print(f"[Epoch {epoch + 1}] Validation batch {val_batch_idx + 1}/{len(val_dataloader)}")

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"[Epoch {epoch + 1}] Average Validation Loss: {avg_val_loss:.4f}")
        
        # Lưu mô hình nếu validation loss tốt hơn
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"[Epoch {epoch + 1}] Mô hình đã được lưu vào {MODEL_SAVE_PATH} với Validation Loss: {best_val_loss:.4f}")

    print("\n--- Huấn luyện mô hình hoàn thành ---")

    # 5. Đánh giá cuối cùng trên tập xác thực (sử dụng mô hình tốt nhất đã lưu)
    print("\n--- Đánh giá cuối cùng trên tập xác thực ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            language_one_hot = batch['language_one_hot'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, numerical_features, language_one_hot)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Báo cáo phân loại trên tập xác thực:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder_type.classes_, labels=label_encoder_type.transform(label_encoder_type.classes_)))