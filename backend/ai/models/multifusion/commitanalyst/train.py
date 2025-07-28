import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.preprocessing import StandardScaler
from torch import nn
from tqdm.auto import tqdm
import json
import joblib
import numpy as np
import os
import re

# Thêm import cho Automatic Mixed Precision
from torch.cuda.amp import autocast, GradScaler

# --- 1. Cấu hình và Tham số ---
class Config:
    def __init__(self):
        self.PRETRAINED_MODEL_NAME = 'microsoft/codebert-base'
        self.MAX_TOKEN_LEN = 512
        self.BATCH_SIZE = 64 # Đã tăng lên để tận dụng A100 VRAM
        self.NUM_EPOCHS = 10
        self.LEARNING_RATE = 2e-5
        self.NUM_CLASSES = 7 # Sẽ được cập nhật tự động dựa trên nhãn trong dữ liệu

        # Có thể thử giảm dropout nếu batch size lớn, nhưng giữ nguyên để an toàn
        self.DROPOUT_RATE_NUMERIC_MLP = 0.1
        self.DROPOUT_RATE_CLASSIFIER = 0.2

        self.RANDOM_SEED = 42

        self.NUMERIC_FEATURES = []
        self.NUM_NUMERIC_FEATURES = 0 # Sẽ được cập nhật tự động

        self.CODEBERT_OUTPUT_DIM = 768
        self.MLP_HIDDEN_DIM = 128

        self.TRAIN_FILE = 'train_set_ver2.jsonl'
        self.VAL_FILE = 'val_set_ver2.jsonl'

        self.BEST_MODEL_SAVE_PATH = 'best_multi_modal_fusion_model.pth'
        self.CHECKPOINT_SAVE_PATH = 'latest_checkpoint.pth'
        self.LABEL_MAP_SAVE_PATH = 'label_map.json'
        self.REVERSE_LABEL_MAP_SAVE_PATH = 'reverse_label_map.json'
        self.SCALER_SAVE_PATH = 'scaler.pkl'


config = Config()
torch.manual_seed(config.RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Sử dụng thiết bị: {device}")

# --- 2. Chuẩn bị Dữ liệu ---

print("Đang tải Tokenizer và CodeBERT model...")
tokenizer = RobertaTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
print("Đã tải xong.")

print(f"Đang đọc dữ liệu từ '{config.TRAIN_FILE}' và '{config.VAL_FILE}'...")
try:
    train_df = pd.read_json(config.TRAIN_FILE, lines=True)
    val_df = pd.read_json(config.VAL_FILE, lines=True)
    print("Đã đọc dữ liệu thành công.")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file dữ liệu. Đảm bảo '{config.TRAIN_FILE}' và '{config.VAL_FILE}' nằm trong cùng thư mục với script hoặc cung cấp đường dẫn đầy đủ.")
    exit()
except Exception as e:
    print(f"Lỗi khi đọc file JSONL: {e}")
    exit()

# --- Tự động phát hiện các thuộc tính số ---
print("Đang tự động phát hiện các thuộc tính số...")
fixed_numeric_features = ['file_count', 'lines_added', 'lines_removed', 'total_changes', 'num_dirs_changed']
detected_all_numeric_features = []

detected_all_numeric_features.extend(fixed_numeric_features)

for col in train_df.columns:
    if col not in fixed_numeric_features:
        sample_values = train_df[col].head(min(len(train_df), 100))

        if not sample_values.empty:
            converted_sample = pd.to_numeric(sample_values, errors='coerce')
            if converted_sample.count() / len(converted_sample) > 0.90:
                detected_all_numeric_features.append(col)

config.NUMERIC_FEATURES = sorted(list(set(detected_all_numeric_features)))
config.NUM_NUMERIC_FEATURES = len(config.NUMERIC_FEATURES)

print(f"Các thuộc tính số được phát hiện: {config.NUMERIC_FEATURES}")
print(f"Tổng số thuộc tính số: {config.NUM_NUMERIC_FEATURES}")


# Ánh xạ nhãn string sang int
all_unique_labels = pd.concat([train_df['commit_type'], val_df['commit_type']]).unique()
label_map = {label: i for i, label in enumerate(sorted(all_unique_labels))}
reverse_label_map = {i: label for label, i in label_map.items()}

config.NUM_CLASSES = len(label_map)
print(f"Số lượng lớp (commit types) được phát hiện: {config.NUM_CLASSES}")
print("Ánh xạ nhãn:", label_map)

train_df['label'] = train_df['commit_type'].map(label_map)
val_df['label'] = val_df['commit_type'].map(label_map)

print(f"Kích thước tập huấn luyện: {len(train_df)}")
print(f"Kích thước tập kiểm tra: {len(val_df)}")

# --- Xử lý dữ liệu số: Chuyển đổi kiểu và điền giá trị thiếu ---
print("Đang xử lý dữ liệu số (chuyển đổi kiểu và điền giá trị thiếu)...")
for col in config.NUMERIC_FEATURES:
    if col not in train_df.columns:
        print(f"Cảnh báo: Cột '{col}' không tồn tại trong train_df. Thêm cột với giá trị 0.")
        train_df[col] = 0.0
    if col not in val_df.columns:
        print(f"Cảnh báo: Cột '{col}' không tồn tại trong val_df. Thêm cột với giá trị 0.")
        val_df[col] = 0.0

    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    val_df[col] = pd.to_numeric(val_df[col], errors='coerce')

    train_df[col] = train_df[col].fillna(0)
    val_df[col] = val_df[col].fillna(0)

    train_df[col] = train_df[col].astype(np.float32) # Có thể dùng float32 để tiết kiệm bộ nhớ
    val_df[col] = val_df[col].astype(np.float32) # Có thể dùng float32 để tiết kiệm bộ nhớ

print("Đã xử lý dữ liệu số xong.")


# --- Tính toán trọng số lớp từ tập huấn luyện ---
print("Đang tính toán trọng số lớp từ tập huấn luyện...")
class_counts = train_df['label'].value_counts().sort_index()
print("Số lượng mẫu của mỗi lớp trong tập huấn luyện:\n", class_counts)

for i in range(config.NUM_CLASSES):
    if i not in class_counts.index:
        class_counts.loc[i] = 0
class_counts = class_counts.sort_index()
print("Số lượng mẫu của mỗi lớp (sau khi kiểm tra):", class_counts.values)

total_train_samples = len(train_df)
class_weights_raw = []
for count in class_counts.values:
    if count > 0:
        class_weights_raw.append(total_train_samples / (count * config.NUM_CLASSES))
    else:
        # Nếu một lớp không có mẫu nào, gán trọng số cao để khuyến khích mô hình học lớp đó
        class_weights_raw.append(total_train_samples / (1 * config.NUM_CLASSES))

class_weights = torch.tensor(class_weights_raw, dtype=torch.float32).to(device)

print("Trọng số lớp đã tính toán và đưa lên thiết bị:", class_weights)

with open(config.LABEL_MAP_SAVE_PATH, 'w') as f:
    json.dump(label_map, f)
print(f"Đã lưu label_map vào: {config.LABEL_MAP_SAVE_PATH}")

with open(config.REVERSE_LABEL_MAP_SAVE_PATH, 'w') as f:
    json.dump(reverse_label_map, f)
print(f"Đã lưu reverse_label_map vào: {config.REVERSE_LABEL_MAP_SAVE_PATH}")

scaler_obj = StandardScaler() # Đổi tên biến để tránh trùng với GradScaler
train_df[config.NUMERIC_FEATURES] = scaler_obj.fit_transform(train_df[config.NUMERIC_FEATURES])
val_df[config.NUMERIC_FEATURES] = scaler_obj.transform(val_df[config.NUMERIC_FEATURES])
print("Đã chuẩn hóa xong.")

joblib.dump(scaler_obj, config.SCALER_SAVE_PATH)
print(f"Đã lưu StandardScaler vào: {config.SCALER_SAVE_PATH}")


class CommitDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_token_len, numeric_features):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.numeric_features = numeric_features

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        commit_message = str(row['commit_message'])
        diff_content = str(row['diff_content'])
        label = row['label']

        numeric_values = []
        for feature in self.numeric_features:
            value = row[feature] if feature in row else 0.0
            numeric_values.append(value)

        numeric_data = np.array(numeric_values, dtype=np.float32) # Đảm bảo float32
        numeric_data = torch.tensor(numeric_data)

        text = commit_message + self.tokenizer.sep_token + diff_content

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'numeric_data': numeric_data,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Tạo Dataset và DataLoader
# Có thể thêm num_workers nếu CPU không phải là nút thắt cổ chai,
# nhưng thường không cần thiết cho Colab.
train_dataset = CommitDataset(train_df, tokenizer, config.MAX_TOKEN_LEN, config.NUMERIC_FEATURES)
val_dataset = CommitDataset(val_df, tokenizer, config.MAX_TOKEN_LEN, config.NUMERIC_FEATURES)

train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

print("Đã tạo DataLoader.")

# --- 3. Định nghĩa Mô hình Fusion ---
class MultiModalFusionModel(nn.Module):
    def __init__(self, config):
        super(MultiModalFusionModel, self).__init__()
        self.codebert = RobertaModel.from_pretrained(config.PRETRAINED_MODEL_NAME)

        self.numeric_mlp = nn.Sequential(
            nn.Linear(config.NUM_NUMERIC_FEATURES, config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE_NUMERIC_MLP), # Sử dụng dropout rate từ config
            nn.Linear(config.MLP_HIDDEN_DIM, config.MLP_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE_NUMERIC_MLP) # Sử dụng dropout rate từ config
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.CODEBERT_OUTPUT_DIM + (config.MLP_HIDDEN_DIM // 2), 256),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE_CLASSIFIER), # Sử dụng dropout rate từ config
            nn.Linear(256, config.NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, numeric_data):
        # CodeBERT forward pass
        codebert_output = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = codebert_output.last_hidden_state[:, 0, :] # Lấy [CLS] token representation

        # Numeric MLP forward pass
        numeric_output = self.numeric_mlp(numeric_data)

        # Concatenate features
        combined_features = torch.cat((pooled_output, numeric_output), dim=1)

        # Classifier
        logits = self.classifier(combined_features)
        return logits

print("Đã định nghĩa mô hình.")

# --- 4. Hàm Huấn luyện và Đánh giá ---
# Thêm `scaler` vào tham số của `train_epoch`
def train_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        numeric_data = batch['numeric_data'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Bọc forward pass trong autocast() để sử dụng Mixed Precision
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, numeric_data=numeric_data)
            loss = criterion(outputs, labels)

        # Thực hiện backward pass và cập nhật trọng số với scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() # Cập nhật scale factor

        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / total_samples
    return avg_loss, accuracy

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numeric_data = batch['numeric_data'].to(device)
            labels = batch['labels'].to(device)

            # Bọc forward pass trong autocast()
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, numeric_data=numeric_data)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / total_samples
    return avg_loss, accuracy

# --- Hàm lưu và tải Checkpoint ---
def save_checkpoint(epoch, model, optimizer, best_val_accuracy, filepath):
    print(f"Đang lưu checkpoint cho Epoch {epoch+1} vào: {filepath}")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_accuracy': best_val_accuracy,
        'numeric_features': config.NUMERIC_FEATURES
    }
    torch.save(checkpoint, filepath)

# Tinh chỉnh hàm load_checkpoint để sử dụng model và optimizer thực sự sau khi tải
def load_checkpoint(filepath_checkpoint, filepath_best_model, model_ref, optimizer_ref): # Chấp nhận tham chiếu
    start_epoch = 0
    best_val_accuracy = 0.0

    if os.path.exists(filepath_checkpoint):
        print(f"Đang tải checkpoint đầy đủ từ: {filepath_checkpoint}")
        try:
            checkpoint = torch.load(filepath_checkpoint, map_location=device)
            model_ref.load_state_dict(checkpoint['model_state_dict'])
            optimizer_ref.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_accuracy = checkpoint['best_val_accuracy']
            loaded_numeric_features = checkpoint.get('numeric_features', None)

            if loaded_numeric_features is not None:
                if loaded_numeric_features != config.NUMERIC_FEATURES:
                    print(f"Cảnh báo: NUMERIC_FEATURES từ checkpoint ({loaded_numeric_features}) khác với config hiện tại ({config.NUMERIC_FEATURES}).")
                    print(f"Nếu cấu trúc dữ liệu không thay đổi, có thể tiếp tục. Nếu không, hãy kiểm tra lại.")
                config.NUMERIC_FEATURES = loaded_numeric_features
                config.NUM_NUMERIC_FEATURES = len(config.NUMERIC_FEATURES)
                print(f"Cập nhật NUMERIC_FEATURES từ checkpoint: {config.NUMERIC_FEATURES}")
            else:
                print("Cảnh báo: Checkpoint cũ không có 'numeric_features'. Sử dụng config hiện tại.")

            print(f"Đã tải checkpoint. Tiếp tục huấn luyện từ Epoch {start_epoch} với best_val_accuracy {best_val_accuracy:.4f}.")
            return start_epoch, best_val_accuracy
        except Exception as e:
            print(f"Lỗi khi tải checkpoint từ {filepath_checkpoint}: {e}. Sẽ thử tải best model hoặc bắt đầu từ đầu.")

    if os.path.exists(filepath_best_model):
        print(f"Không tìm thấy hoặc không thể tải checkpoint đầy đủ. Đang tải trọng số từ best model: {filepath_best_model}")
        try:
            model_ref.load_state_dict(torch.load(filepath_best_model, map_location=device))
            # Khi chỉ tải best model, optimizer cần được reset
            start_epoch = 0
            best_val_accuracy = 0.0
            print(f"Đã tải trọng số từ best model. Bắt đầu huấn luyện từ Epoch {start_epoch} (optimizer được reset).")
            return start_epoch, best_val_accuracy
        except Exception as e:
            print(f"Lỗi khi tải best model từ {filepath_best_model}: {e}. Bắt đầu từ đầu.")

    print("Không tìm thấy checkpoint hoặc best model. Bắt đầu huấn luyện từ Epoch 0 với trọng số mặc định.")
    return 0, 0.0


# --- 5. Chạy Huấn luyện ---
print("Bắt đầu huấn luyện mô hình...")

# Khởi tạo model và optimizer ban đầu (nếu cần sau đó sẽ load state_dict)
model = MultiModalFusionModel(config)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

# Khởi tạo GradScaler
scaler_amp = GradScaler() # Đổi tên biến để tránh trùng với scaler_obj của StandardScaler

# Tải checkpoint hoặc best model (cập nhật trạng thái cho model và optimizer hiện tại)
start_epoch, best_val_accuracy = load_checkpoint(
    config.CHECKPOINT_SAVE_PATH,
    config.BEST_MODEL_SAVE_PATH,
    model, # Truyền tham chiếu model
    optimizer # Truyền tham chiếu optimizer
)

criterion = nn.CrossEntropyLoss(weight=class_weights)

for epoch in range(start_epoch, config.NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
    # Truyền scaler_amp vào hàm train_epoch
    train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion, device, scaler_amp)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    val_loss, val_acc = eval_epoch(model, val_dataloader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), config.BEST_MODEL_SAVE_PATH)
        print(f"Lưu mô hình tốt nhất vào: {config.BEST_MODEL_SAVE_PATH} với Accuracy: {best_val_accuracy:.4f}")

    save_checkpoint(epoch, model, optimizer, best_val_accuracy, config.CHECKPOINT_SAVE_PATH)


print("\nQuá trình huấn luyện hoàn tất!")
print(f"Accuracy tốt nhất trên tập kiểm tra: {best_val_accuracy:.4f}")

# --- Đánh giá chi tiết mô hình sau khi huấn luyện (tùy chọn) ---
from sklearn.metrics import classification_report, confusion_matrix

print("\n--- Đánh giá chi tiết mô hình trên tập validation (từ model tốt nhất đã lưu) ---")
try:
    if os.path.exists(config.BEST_MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.BEST_MODEL_SAVE_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"Đã tải model tốt nhất từ {config.BEST_MODEL_SAVE_PATH} để đánh giá cuối cùng.")

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Final Evaluation on Val Set"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                numeric_data = batch['numeric_data'].to(device)
                labels = batch['labels'].to(device)

                with autocast(): # Sử dụng autocast cho đánh giá cuối cùng
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, numeric_data=numeric_data)
                _, preds = torch.max(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print("Classification Report (Validation Set):")
        # Đảm bảo reverse_label_map được tải nếu chưa có
        if 'reverse_label_map' not in locals():
            try:
                with open(config.REVERSE_LABEL_MAP_SAVE_PATH, 'r') as f:
                    reverse_label_map = json.load(f)
            except FileNotFoundError:
                print(f"Cảnh báo: Không tìm thấy {config.REVERSE_LABEL_MAP_SAVE_PATH}. Báo cáo sẽ hiển thị nhãn số.")
                reverse_label_map = {str(i): str(i) for i in range(config.NUM_CLASSES)}

        target_names_list = [reverse_label_map[str(i)] for i in range(config.NUM_CLASSES)]
        print(classification_report(all_labels, all_preds, target_names=target_names_list))

        print("\nConfusion Matrix (Validation Set):")
        print(confusion_matrix(all_labels, all_preds))
    else:
        print(f"Không tìm thấy model tốt nhất tại {config.BEST_MODEL_SAVE_PATH} để đánh giá chi tiết.")
except Exception as e:
    print(f"Lỗi khi đánh giá chi tiết mô hình: {e}")