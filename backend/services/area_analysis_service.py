import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from typing import Dict, Any, List

# Assuming the model is saved in this path relative to the backend directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ai', 'models', 'multifusion', 'areaAnalyst', 'best_area_classifier.pt')
# Assuming training data for LabelEncoder and StandardScaler is available
TRAIN_DATA_PATH = os.path.join(os.path.dirname(__file__), '..','ai', 'models', 'multifusion', 'areaAnalyst', 'train.jsonl')

class MultiFusionAreaModel(nn.Module):
    def __init__(self, bert_name='distilbert-base-uncased', num_num_features=4, hidden_size=128, num_classes=5):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(bert_name)
        self.mlp = nn.Sequential(
            nn.Linear(num_num_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.classifier = nn.Linear(self.bert.config.hidden_size + hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, num_feat):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_feat = bert_out.last_hidden_state[:, 0]
        mlp_feat = self.mlp(num_feat)
        fusion = torch.cat([bert_feat, mlp_feat], dim=1)
        logits = self.classifier(fusion)
        return logits

class AreaAnalysisService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AreaAnalysisService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Placeholder for LabelEncoder and StandardScaler ---
        # In a real scenario, these should be saved during training and loaded here.
        # For demonstration, we'll re-fit them using sample data.
        # This is NOT robust for production if the training data changes or is not available.
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self._fit_encoders_from_sample_data()
        # --- End Placeholder ---

        # Determine num_classes from label_encoder after fitting
        num_classes = len(self.label_encoder.classes_)
        self.model = MultiFusionAreaModel(num_classes=num_classes).to(self.device)
        
        if os.path.exists(MODEL_PATH):
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.eval()
            print(f"AreaAnalysisService: Model loaded from {MODEL_PATH}")
        else:
            print(f"AreaAnalysisService: Model not found at {MODEL_PATH}. Please train the model first.")
            # Optionally, raise an error or handle gracefully
            self.model = None # Indicate that model is not loaded

    def _fit_encoders_from_sample_data(self):
        # This is a temporary solution for demonstration.
        # In production, LabelEncoder and StandardScaler should be saved/loaded.
        areas = []
        num_features_data = []
        try:
            with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if 'dev_area' in item:
                            areas.append(item['dev_area'])
                        num_features_data.append([
                            int(item.get('files_count', 0)),
                            int(item.get('lines_added', 0)),
                            int(item.get('lines_removed', 0)),
                            int(item.get('total_changes', 0))
                        ])
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON line in {TRAIN_DATA_PATH}: {line.strip()}")
            
            if areas:
                self.label_encoder.fit(areas)
                print(f"AreaAnalysisService: LabelEncoder fitted with classes: {self.label_encoder.classes_}")
            else:
                print(f"AreaAnalysisService: No 'dev_area' found in {TRAIN_DATA_PATH} to fit LabelEncoder.")
                # Fallback for label encoder if no data
                self.label_encoder.fit(['unknown']) # Default class

            if num_features_data:
                self.scaler.fit(num_features_data)
                print(f"AreaAnalysisService: StandardScaler fitted.")
            else:
                print(f"AreaAnalysisService: No numerical features found in {TRAIN_DATA_PATH} to fit StandardScaler.")
                # Fallback for scaler if no data
                self.scaler.mean_ = [0.0, 0.0, 0.0, 0.0]
                self.scaler.scale_ = [1.0, 1.0, 1.0, 1.0]

        except FileNotFoundError:
            print(f"AreaAnalysisService: Training data not found at {TRAIN_DATA_PATH}. Cannot fit encoders.")
            # Fallback for label encoder and scaler if file not found
            self.label_encoder.fit(['unknown'])
            self.scaler.mean_ = [0.0, 0.0, 0.0, 0.0]
            self.scaler.scale_ = [1.0, 1.0, 1.0, 1.0]
        except Exception as e:
            print(f"AreaAnalysisService: Error fitting encoders: {e}")
            self.label_encoder.fit(['unknown'])
            self.scaler.mean_ = [0.0, 0.0, 0.0, 0.0]
            self.scaler.scale_ = [1.0, 1.0, 1.0, 1.0]


    def predict_area(self, commit_data: Dict[str, Any]) -> str:
        if self.model is None:
            return "Model not loaded" # Or raise an error

        text = (commit_data.get('commit_message', '') + ' ' + commit_data.get('diff_content', ''))[:512]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

        num_feat = [
            int(commit_data.get('files_count', 0)),
            int(commit_data.get('lines_added', 0)),
            int(commit_data.get('lines_removed', 0)),
            int(commit_data.get('total_changes', 0))
        ]
        num_feat = self.scaler.transform([num_feat])[0]

        input_ids = inputs['input_ids'].squeeze(0).to(self.device)
        attention_mask = inputs['attention_mask'].squeeze(0).to(self.device)
        num_feat_tensor = torch.tensor(num_feat, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0), num_feat_tensor)
            predicted_label_idx = torch.argmax(logits, dim=1).item()
            predicted_area = self.label_encoder.inverse_transform([predicted_label_idx])[0]
        
        return predicted_area
