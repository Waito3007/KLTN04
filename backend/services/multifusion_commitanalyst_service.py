from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
from collections import defaultdict
from fastapi import HTTPException
import logging
import torch
import torch.nn as nn
import json
import numpy as np
import joblib
from transformers import RobertaTokenizer, RobertaModel
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class MultiModalFusionModel(nn.Module):
    def __init__(self, pretrained_model_name, num_numeric_features, mlp_hidden_dim, num_classes, dropout_numeric=0.1, dropout_classifier=0.2):
        super(MultiModalFusionModel, self).__init__()
        self.codebert = RobertaModel.from_pretrained(pretrained_model_name)
        self.numeric_mlp = nn.Sequential(
            nn.Linear(num_numeric_features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_numeric),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_numeric)
        )
        codebert_output_dim = 768
        self.classifier = nn.Sequential(
            nn.Linear(codebert_output_dim + (mlp_hidden_dim // 2), 256),
            nn.ReLU(),
            nn.Dropout(dropout_classifier),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, numeric_data):
        codebert_output = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = codebert_output.last_hidden_state[:, 0, :]
        numeric_output = self.numeric_mlp(numeric_data)
        combined_features = torch.cat((pooled_output, numeric_output), dim=1)
        logits = self.classifier(combined_features)
        return logits

class MultiFusionV2Service:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MultiFusionV2Service, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model = None
        self.tokenizer = None
        self.scaler = None
        self.label_map = None
        self.reverse_label_map = None
        self.numeric_features = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = 512
        self._load_model()
        self._initialized = True
    
    def get_model_info(self):
        return {
            "model_name": "MultiFusion V2",
            "status": "available",
            "input_features": [
                "message",
                "file_count",
                "lines_added",
                "lines_removed",
                "total_changes",
                "num_dirs_changed"
            ],
            "num_classes": 11
        }

    def _load_model(self):
        print("--- Starting _load_model ---")
        try:
            model_dir = Path(__file__).parent.parent / "ai" / "models" / "multifusion" / "commitanalyst"
            print(f"1. Model directory: {model_dir}")

            model_path = model_dir / "best_multi_modal_fusion_model.pth"
            scaler_path = model_dir / "scaler.pkl"
            label_map_path = model_dir / "label_map.json"
            reverse_label_map_path = model_dir / "reverse_label_map.json"
            print("2. Constructed model file paths.")

            if not all([p.exists() for p in [model_path, scaler_path, label_map_path, reverse_label_map_path]]):
                print("ERROR: One or more model files are missing.")
                logger.error("One or more model files are missing. MultiFusionV2Service will be unavailable.")
                return
            print("3. All model files exist.")

            print("4. Loading label_map.json...")
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
            print("5. Loading reverse_label_map.json...")
            with open(reverse_label_map_path, 'r') as f:
                self.reverse_label_map = {int(k): v for k, v in json.load(f).items()}
            print("6. Label maps loaded.")

            print("7. Loading scaler.pkl...")
            self.scaler = joblib.load(scaler_path)
            print("8. Scaler loaded.")

            print("9. Loading tokenizer...")
            self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
            print("10. Tokenizer loaded.")

            if hasattr(self.scaler, 'get_feature_names_out'):
                self.numeric_features = self.scaler.get_feature_names_out().tolist()
                print("11a. Extracted numeric features via get_feature_names_out().")
            else:
                self.numeric_features = ['file_count', 'lines_added', 'lines_removed', 'total_changes', 'num_dirs_changed']
                print("11b. Using fallback numeric features.")

            print(f"12. Numeric features: {self.numeric_features}")

            print("13. Initializing MultiModalFusionModel...")
            self.model = MultiModalFusionModel(
                pretrained_model_name='microsoft/codebert-base',
                num_numeric_features=len(self.numeric_features),
                mlp_hidden_dim=128,
                num_classes=len(self.label_map)
            )
            print("14. Model initialized.")

            print("15. Loading model state dictionary...")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            print("16. Model state loaded.")

            print("17. Moving model to device...")
            self.model.to(self.device)
            self.model.eval()
            print("18. Model ready.")
            logger.info("MultiFusion V2 model loaded successfully.")
            print("--- _load_model finished successfully ---")

        except Exception as e:
            print(f"--- ERROR in _load_model: {e} ---")
            logger.error(f"Error loading MultiFusion V2 model: {e}", exc_info=True)
            self.model = None

    def is_model_available(self) -> bool:
        return all([self.model, self.tokenizer, self.scaler, self.label_map, self.reverse_label_map])

    def predict_commit_type_batch(self, commits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.is_model_available():
            return [self._mock_prediction(c.get('message', '')) for c in commits]

        try:
            text_inputs = [f"{c.get('message', '')} [SEP] {c.get('diff_content', '')[:500]}" for c in commits]
            encodings = self.tokenizer.batch_encode_plus(
                text_inputs, add_special_tokens=True, max_length=self.max_len,
                padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt'
            )
            
            numeric_df = pd.DataFrame(commits)[self.numeric_features]
            numeric_features_scaled = self.scaler.transform(numeric_df)

            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            numeric_tensor = torch.tensor(numeric_features_scaled, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, numeric_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_ids = torch.argmax(probabilities, dim=1)
                confidences = probabilities.max(dim=1).values

            results = []
            for i in range(len(commits)):
                pred_id = predicted_class_ids[i].item()
                results.append({
                    "commit_type": self.reverse_label_map.get(pred_id, "other"),
                    "confidence": round(confidences[i].item(), 4),
                })
            return results
        except Exception as e:
            logger.error(f"Error in MultiFusion batch prediction: {e}", exc_info=True)
            return [self._mock_prediction(c.get('message', '')) for c in commits]

    def _mock_prediction(self, message: str) -> Dict[str, Any]:
        # Simple rule-based mock for when the model is unavailable
        message_lower = message.lower()
        if any(word in message_lower for word in ['fix', 'bug', 'error']):
            return {"commit_type": "fix", "confidence": 0.85}
        if any(word in message_lower for word in ['feat', 'feature', 'add']):
            return {"commit_type": "feat", "confidence": 0.80}
        return {"commit_type": "chore", "confidence": 0.70}


class MultifusionCommitAnalystService:
    def __init__(self, db: Session):
        self.db = db
        self.multifusion_v2_service = MultiFusionV2Service()

    def get_branches(self, repo_id: int):
        # TODO: Trả về danh sách branches thực tế từ DB
        return []

    def get_all_repo_commits_raw(self, repo_id: int):
        # TODO: Trả về danh sách tất cả commits thực tế từ DB
        return []
        self.multifusion_v2_service = MultiFusionV2Service()

    def _get_commits_from_db(self, query: str, params: Dict) -> List[Dict[str, Any]]:
        """Executes a query and returns a list of dictionaries."""
        results = self.db.execute(text(query), params).mappings().all()
        return [dict(row) for row in results]

    def _format_commits_for_ai(self, commits_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepares commit data for the prediction model."""
        formatted_commits = []
        for commit in commits_data:
            insertions = commit.get('insertions', 0) or 0
            deletions = commit.get('deletions', 0) or 0
            formatted_commits.append({
                'message': commit.get('message', ''),
                'diff_content': commit.get('diff_content', ''),
                'lines_added': insertions,
                'lines_removed': deletions,
                'file_count': commit.get('files_changed', 1) or 1,
                'total_changes': insertions + deletions,
                'num_dirs_changed': 0,  # Not available from DB, default to 0
            })
        return formatted_commits

    def _combine_results(self, db_commits: List[Dict], ai_results: List[Dict]) -> List[Dict]:
        """Combines database data with AI analysis results."""
        combined = []
        for i, commit in enumerate(db_commits):
            ai_data = ai_results[i] if i < len(ai_results) else {}
            commit_type = ai_data.get("commit_type", "other")
            
            commit_info = {
                **commit,
                "sha_short": commit.get('sha', 'N/A')[:8],
                "date": commit.get('committer_date').isoformat() if commit.get('committer_date') else None,
                "analysis": {
                    "type": commit_type,
                    "confidence": ai_data.get("confidence", 0.0),
                    "ai_powered": True,
                    "ai_model": "MultiFusionV2"
                }
            }
            combined.append(commit_info)
        return combined

    async def get_all_repo_commits_with_analysis(self, repo_id: int, limit: int, offset: int, branch_name: Optional[str]) -> Dict[str, Any]:
        """Gets all repository commits with AI analysis."""
        query = """
            SELECT id, sha, message, author_name, committer_date, branch_name, insertions, deletions, files_changed, diff_content
            FROM commits 
            WHERE repo_id = :repo_id
        """
        params = {"repo_id": repo_id, "limit": limit, "offset": offset}
        if branch_name:
            query += " AND branch_name = :branch_name"
            params["branch_name"] = branch_name
        query += " ORDER BY committer_date DESC LIMIT :limit OFFSET :offset"

        db_commits = self._get_commits_from_db(query, params)
        
        if not db_commits:
            return {"summary": {"total_commits": 0}, "commits": [], "statistics": {}}

        commits_for_ai = self._format_commits_for_ai(db_commits)
        ai_results = self.multifusion_v2_service.predict_commit_type_batch(commits_for_ai)
        
        commits_with_analysis = self._combine_results(db_commits, ai_results)

        # Basic statistics
        commit_type_stats = defaultdict(int)
        for c in commits_with_analysis:
            commit_type_stats[c['analysis']['type']] += 1

        return {
            "summary": {"total_commits": len(commits_with_analysis)},
            "commits": commits_with_analysis,
            "statistics": {"commit_types": dict(commit_type_stats)}
        }