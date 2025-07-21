import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import json
from sklearn.preprocessing import StandardScaler
import os
from typing import Dict, Any

# Define the MultiFusionModel architecture (copied from train.py)
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
        self.classifier = nn.Linear(self.bert.config.hidden_size + hidden_size, 2) # 2 classes: lowrisk, highrisk

    def forward(self, input_ids, attention_mask, num_feat):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_feat = bert_out.last_hidden_state[:, 0]
        mlp_feat = self.mlp(num_feat)
        fusion = torch.cat([bert_feat, mlp_feat], dim=1)
        logits = self.classifier(fusion)
        return logits

class RiskAnalysisService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RiskAnalysisService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model and scaler paths
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'ai', 'models', 'multifusion', 'riskAnalyst', 'risk_model.pt')
        self.train_data_path = os.path.join(os.path.dirname(__file__), '..', 'ai', 'models', 'multifusion', 'riskAnalyst', 'train.jsonl')

        self.scaler = StandardScaler()
        self._fit_scaler_from_sample_data() # Fit scaler using sample data

        self.model = MultiFusionModel().to(self.device)
        
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print(f"RiskAnalysisService: Model loaded from {self.model_path}")
        else:
            print(f"RiskAnalysisService: Model not found at {self.model_path}. Please train the model first.")
            self.model = None # Indicate that model is not loaded

    def _fit_scaler_from_sample_data(self):
        # This is a temporary solution for demonstration.
        # In production, StandardScaler should be saved/loaded.
        num_features_data = []
        try:
            with open(self.train_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        num_features_data.append([
                            int(item.get('files_count', 0)),
                            int(item.get('lines_added', 0)),
                            int(item.get('lines_removed', 0)),
                            int(item.get('total_changes', 0))
                        ])
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON line in {self.train_data_path}: {line.strip()}")
            
            if num_features_data:
                self.scaler.fit(num_features_data)
                print(f"RiskAnalysisService: StandardScaler fitted from {self.train_data_path}.")
            else:
                print(f"RiskAnalysisService: No numerical features found in {self.train_data_path} to fit StandardScaler.")
                # Fallback for scaler if no data
                self.scaler.mean_ = [0.0, 0.0, 0.0, 0.0]
                self.scaler.scale_ = [1.0, 1.0, 1.0, 1.0]

        except FileNotFoundError:
            print(f"RiskAnalysisService: Training data not found at {self.train_data_path}. Cannot fit scaler.")
            # Fallback for scaler if file not found
            self.scaler.mean_ = [0.0, 0.0, 0.0, 0.0]
            self.scaler.scale_ = [1.0, 1.0, 1.0, 1.0]
        except Exception as e:
            print(f"RiskAnalysisService: Error fitting scaler: {e}")
            self.scaler.mean_ = [0.0, 0.0, 0.0, 0.0]
            self.scaler.scale_ = [1.0, 1.0, 1.0, 1.0]

    def predict_risk(self, commit_data: Dict[str, Any]) -> str:
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
            
            # Map 0 to lowrisk, 1 to highrisk
            predicted_risk = "highrisk" if predicted_label_idx == 1 else "lowrisk"
        
        return predicted_risk
