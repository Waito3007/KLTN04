import json
import os
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch.nn as nn
import numpy as np

# Cấu hình
MODEL_SAVE_PATH = r"C:\SAN\KLTN\KLTN04\backend\ai\models\multifusion\multifusionV2.pth"
BERT_MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 128
DATA_DIR = r"C:\SAN\KLTN\KLTN04\backend\ai\models\multifusion\data" # Path to your data directory

# 2. Định nghĩa Mô hình Multifusion (phải giống với lúc train)
class MultiFusionModel(nn.Module):
    def __init__(self, bert_model_name, num_numerical_features, num_language_features, num_classes):
        super(MultiFusionModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.3)

        self.mlp = nn.Sequential(
            nn.Linear(num_numerical_features + num_language_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(self.bert.config.hidden_size + 64, num_classes)

    def forward(self, input_ids, attention_mask, numerical_features, language_one_hot):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.last_hidden_state[:, 0, :]
        
        combined_structured_features = torch.cat((numerical_features, language_one_hot), dim=1)
        mlp_output = self.mlp(combined_structured_features)

        combined_features = torch.cat((pooled_output, mlp_output), dim=1)
        
        return self.classifier(self.dropout(combined_features))

class MultiFusionAIService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder_lang = None
        self.label_encoder_type = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_resources()

    def _load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _load_resources(self):
        print("Loading MultiFusion AI resources...")
        # Load data to fit encoders and scaler
        train_data = self._load_data(os.path.join(DATA_DIR, "train_commits_balanced_multi.json"))
        val_data = self._load_data(os.path.join(DATA_DIR, "val_commits_balanced_multi.json"))
        all_data = train_data + val_data

        # Initialize Tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)

        # Fit LabelEncoder for languages
        all_languages = [commit["metadata"].get("detected_language", "unknown_language") for commit in all_data]
        self.label_encoder_lang = LabelEncoder()
        self.label_encoder_lang.fit(all_languages)

        # Fit LabelEncoder for commit types
        all_commit_types = [commit.get("commit_type", "other_type").lower() for commit in all_data]
        self.label_encoder_type = LabelEncoder()
        self.label_encoder_type.fit(all_commit_types)
        
        num_classes = len(self.label_encoder_type.classes_)
        num_language_features = len(self.label_encoder_lang.classes_)

        # Fit StandardScaler for numerical features
        all_numerical_features = []
        for commit in all_data:
            lines_added = commit.get("lines_added", 0)
            lines_removed = commit.get("lines_removed", 0)
            files_count = commit.get("files_count", 0)
            total_changes = lines_added + lines_removed
            if lines_removed == 0:
                ratio_added_removed = lines_added
            else:
                ratio_added_removed = lines_added / lines_removed
            all_numerical_features.append([lines_added, lines_removed, files_count, total_changes, ratio_added_removed])
        
        self.scaler = StandardScaler()
        self.scaler.fit(all_numerical_features)
        num_numerical_features = len(all_numerical_features[0])

        # Load Model
        self.model = MultiFusionModel(BERT_MODEL_NAME, num_numerical_features, num_language_features, num_classes)
        self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("MultiFusion AI resources loaded.")

    def predict_commit_type(self, commit_message: str, lines_added: int, lines_removed: int, files_count: int, detected_language: str):
        if self.model is None:
            raise RuntimeError("MultiFusion AI model not loaded.")

        # Preprocess text features
        encoding = self.tokenizer.encode_plus(
            commit_message,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].flatten().unsqueeze(0).to(self.device)
        attention_mask = encoding['attention_mask'].flatten().unsqueeze(0).to(self.device)

        # Preprocess numerical features
        total_changes = lines_added + lines_removed
        if lines_removed == 0:
            ratio_added_removed = lines_added
        else:
            ratio_added_removed = lines_added / lines_removed
        
        numerical_features_raw = np.array([[lines_added, lines_removed, files_count, total_changes, ratio_added_removed]], dtype=np.float32)
        numerical_features_scaled = torch.tensor(self.scaler.transform(numerical_features_raw), dtype=torch.float32).to(self.device)

        # Preprocess categorical features (language)
        lang_encoded = self.label_encoder_lang.transform([detected_language])[0]
        lang_one_hot = torch.zeros(len(self.label_encoder_lang.classes_)).unsqueeze(0)
        lang_one_hot[0, lang_encoded] = 1
        lang_one_hot = lang_one_hot.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, numerical_features_scaled, lang_one_hot)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            
        predicted_commit_type = self.label_encoder_type.inverse_transform([predicted_class_idx])[0]
        return predicted_commit_type
