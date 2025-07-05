import json
import os
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# --- Định nghĩa lại các class và cấu hình từ script training ---

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

class CommitClassificationService:
    def __init__(self):
        """
        Initializes the service by loading the model and all necessary preprocessors.
        It re-fits the preprocessors on the same data used for training to ensure consistency.
        """
        # --- Cấu hình --- (Nên được quản lý tốt hơn trong ứng dụng thực tế)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "model", "multifusion", "data")
        self.model_path = os.path.join(self.base_dir, "model", "multifusion", "multifusionV2.pth")
        self.bert_model_name = 'distilbert-base-uncased'
        self.max_len = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("--- Initializing Commit Classification Service ---")
        print(f"Using device: {self.device}")

        # --- Tải và fit lại các bộ tiền xử lý ---
        self._fit_preprocessors()

        # --- Tải mô hình --- 
        self.model = MultiFusionModel(
            bert_model_name=self.bert_model_name,
            num_numerical_features=self.num_numerical_features,
            num_language_features=len(self.label_encoder_lang.classes_),
            num_classes=len(self.label_encoder_type.classes_)
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

        # --- Tải tokenizer ---
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.bert_model_name)
        print("Tokenizer loaded successfully.")
        print("--- Service Initialized ---")

    def _load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _fit_preprocessors(self):
        """Fits the LabelEncoders and StandardScaler on the training and validation data."""
        print("Fitting preprocessors...")
        train_data = self._load_data(os.path.join(self.data_dir, "train_commits_balanced_multi.json"))
        val_data = self._load_data(os.path.join(self.data_dir, "val_commits_balanced_multi.json"))
        full_data = train_data + val_data

        # Fit LabelEncoder for language
        all_languages = [commit["metadata"].get("detected_language", "unknown_language") for commit in full_data]
        self.label_encoder_lang = LabelEncoder()
        self.label_encoder_lang.fit(all_languages)

        # Fit LabelEncoder for commit type
        all_commit_types = [commit.get("commit_type", "other_type").lower() for commit in full_data]
        self.label_encoder_type = LabelEncoder()
        self.label_encoder_type.fit(all_commit_types)

        # Fit StandardScaler for numerical features
        all_numerical_features = []
        for commit in full_data:
            lines_added = commit.get("lines_added", 0)
            lines_removed = commit.get("lines_removed", 0)
            files_count = commit.get("files_count", 0)
            total_changes = lines_added + lines_removed
            ratio = lines_added / lines_removed if lines_removed != 0 else lines_added
            all_numerical_features.append([lines_added, lines_removed, files_count, total_changes, ratio])
        
        self.scaler = StandardScaler()
        self.scaler.fit(all_numerical_features)
        self.num_numerical_features = len(all_numerical_features[0])
        print("Preprocessors fitted successfully.")

    def classify_commit(self, commit_data: dict) -> dict:
        """
        Classifies a single commit using the loaded MultiFusion model.
        """
        # 1. Preprocess input
        message = commit_data.get("message", "")
        encoding = self.tokenizer.encode_plus(
            message,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        lines_added = commit_data.get("insertions", 0)
        lines_removed = commit_data.get("deletions", 0)
        files_count = commit_data.get("files_changed", 0)
        total_changes = lines_added + lines_removed
        ratio = lines_added / lines_removed if lines_removed != 0 else lines_added
        numerical_features_raw = np.array([[lines_added, lines_removed, files_count, total_changes, ratio]])
        numerical_features_scaled = torch.tensor(self.scaler.transform(numerical_features_raw).flatten(), dtype=torch.float32).unsqueeze(0)

        # For language, we need to handle cases where the language was not seen during training
        lang = commit_data.get("file_types", {}).get("dominant_language", "unknown_language")
        if lang not in self.label_encoder_lang.classes_:
            lang = "unknown_language" # Default to a known category
        lang_encoded = self.label_encoder_lang.transform([lang])[0]
        lang_one_hot = torch.zeros(1, len(self.label_encoder_lang.classes_))
        lang_one_hot[0, lang_encoded] = 1

        # 2. Move tensors to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        numerical_features = numerical_features_scaled.to(self.device)
        language_one_hot = lang_one_hot.to(self.device)

        # 3. Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, numerical_features, language_one_hot)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=1)
            
            predicted_class = self.label_encoder_type.inverse_transform([predicted_class_idx.item()])[0]
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence.item()
            }

# Singleton instance
commit_classification_service = CommitClassificationService()