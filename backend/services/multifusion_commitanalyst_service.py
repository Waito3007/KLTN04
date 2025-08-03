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
import pandas as pd
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

            # model_path = model_dir / "commitanalyst.pth"
            model_path = model_dir / "ver4.3.pth"
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


            # SỬA: Sử dụng đúng 10 đặc trưng số học như khi train model, sắp xếp theo alphabet
            self.numeric_features = [
                'confidence_score',
                'file_count',
                'lines_added',
                'lines_removed',
                'num_build_files',
                'num_dirs_changed',
                'num_doc_files',
                'num_test_files',
                'risk',
                'total_changes'
            ]
            print("11. Using 10 numeric features for commit analysis.")
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


def extract_commit_numeric_features(commit):
    """
    Sinh các đặc trưng số học từ commit: num_{ext}_files, num_build_files, num_test_files, num_doc_files, num_dirs_changed.
    """
    import json, os
    modified_files = commit.get('modified_files', [])
    if isinstance(modified_files, str):
        try:
            modified_files = json.loads(modified_files)
        except Exception:
            modified_files = []
    file_types = commit.get('file_types', {})
    if isinstance(file_types, str):
        try:
            file_types = json.loads(file_types)
        except Exception:
            file_types = {}

    filetype_counts = {}
    dirs = set()
    num_build_files = 0
    num_test_files = 0
    num_doc_files = 0

    for file in modified_files:
        fname = file.lower()
        ext = os.path.splitext(fname)[1][1:]
        if ext:
            filetype_counts[ext] = filetype_counts.get(ext, 0) + 1
        if os.path.basename(fname) in {'package.json', 'pom.xml', 'build.gradle', 'makefile', 'cmakelists.txt'} or ext in {'yml', 'yaml', 'xml', 'gradle'}:
            num_build_files += 1
        if 'test' in fname or 'spec' in fname:
            num_test_files += 1
        if ext in {'md', 'rst'} or 'doc' in fname:
            num_doc_files += 1
        dir_path = os.path.dirname(fname)
        if dir_path:
            dirs.add(dir_path)

    for ext, count in file_types.items():
        ext_key = ext.lstrip('.')
        filetype_counts[ext_key] = filetype_counts.get(ext_key, 0) + count

    features = {f'num_{ext}_files': count for ext, count in filetype_counts.items() if ext}
    features['num_build_files'] = num_build_files
    features['num_test_files'] = num_test_files
    features['num_doc_files'] = num_doc_files
    features['num_dirs_changed'] = len(dirs)
    return features

# dịch vụ commit analyst cho MultiFusion
class MultifusionCommitAnalystService:
    def __init__(self, db: Session):
        self.db = db
        self.multifusion_v2_service = MultiFusionV2Service()
        print("Session DB:", self.db)

    def get_branches(self, repo_id: int):
        """Trả về danh sách branches thực tế từ DB"""
        query = """
            SELECT id, name, sha, is_default, is_protected, created_at, last_commit_date, commits_count, contributors_count
            FROM branches
            WHERE repo_id = :repo_id
            ORDER BY is_default DESC, name ASC
        """
        params = {"repo_id": repo_id}
        return self._get_commits_from_db(query, params)

    def get_all_repo_commits_raw(self, repo_id: int):
        """Trả về danh sách tất cả commits thực tế từ DB"""
        query = """
            SELECT id, sha, message, author_name, committer_date, branch_name, insertions, deletions, files_changed, diff_content, modified_files, file_types
            FROM commits
            WHERE repo_id = :repo_id
            ORDER BY committer_date DESC
        """
        params = {"repo_id": repo_id}
        return self._get_commits_from_db(query, params)

    def _get_commits_from_db(self, query: str, params: Dict) -> List[Dict[str, Any]]:
        """Executes a query and returns a list of dictionaries."""
        results = self.db.execute(text(query), params).mappings().all()
        return [dict(row) for row in results]

    # Đã tách logic ra ngoài, không cần hàm này nữa

    def _format_commits_for_ai(self, commits_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepares commit data for the prediction model, tự động sinh đặc trưng số học từ filetype."""
        formatted_commits = []
        for commit in commits_data:
            insertions = commit.get('insertions', 0) or 0
            deletions = commit.get('deletions', 0) or 0
            numeric_features = extract_commit_numeric_features(commit)
            formatted = {
                'message': commit.get('message', ''),
                'diff_content': commit.get('diff_content', ''),
                'lines_added': insertions,
                'lines_removed': deletions,
                'file_count': commit.get('files_changed', commit.get('files_count', 1)) or 1,
                'total_changes': insertions + deletions,
                'num_dirs_changed': commit.get('num_dirs_changed', 0),
                'branch_name': commit.get('branch_name', ''),
                'author_name': commit.get('author_name', ''),
                'sha': commit.get('sha', ''),
                'date': commit.get('committer_date', None),
                **numeric_features
            }
            # Add missing features with defaults to prevent KeyError
            formatted.setdefault('risk', 0)
            formatted.setdefault('confidence_score', 1.0)
            formatted_commits.append(formatted)
        return formatted_commits

    def _combine_results(self, db_commits: List[Dict], ai_results: List[Dict], commits_for_ai: List[Dict] = None) -> List[Dict]:
        """Combines database data with AI analysis results and merges numeric features."""
        combined = []
        # If commits_for_ai is not provided, fallback to old behavior
        if commits_for_ai is None:
            commits_for_ai = [{} for _ in db_commits]
        for i, commit in enumerate(db_commits):
            ai_data = ai_results[i] if i < len(ai_results) else {}
            commit_type = ai_data.get("commit_type", "other")
            # Merge numeric features from commits_for_ai
            numeric_features = {}
            if i < len(commits_for_ai):
                # Only include keys that are numeric features (num_*, *_files, num_dirs_changed)
                numeric_features = {k: v for k, v in commits_for_ai[i].items() if (k.startswith("num_") or k.endswith("_files") or k == "num_dirs_changed")}
            # Rename 'modified_files' to 'files_changed' in output
            commit_dict = dict(commit)
            if 'modified_files' in commit_dict:
                commit_dict['files_changed'] = commit_dict.pop('modified_files')
            commit_info = {
                **commit_dict,
                **numeric_features,
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
        # Get total commits count
        count_query = "SELECT COUNT(id) FROM commits WHERE repo_id = :repo_id"
        count_params = {"repo_id": repo_id}
        if branch_name:
            count_query += " AND LOWER(branch_name) = LOWER(:branch_name)"
            count_params["branch_name"] = branch_name
        
        total_commits = self.db.execute(text(count_query), count_params).scalar() or 0

        # Get paginated commits
        query = """
            SELECT id, sha, message, author_name, committer_date, branch_name, insertions, deletions, files_changed, diff_content, modified_files, file_types
            FROM commits 
            WHERE repo_id = :repo_id
        """
        params = {"repo_id": repo_id, "limit": limit, "offset": offset}
        if branch_name:
            query += " AND LOWER(branch_name) = LOWER(:branch_name)"
            params["branch_name"] = branch_name
        query += " ORDER BY committer_date DESC LIMIT :limit OFFSET :offset"

        db_commits = self._get_commits_from_db(query, params)
        
        if not db_commits:
            return {
                "summary": {"total_commits": total_commits, "page_commits": 0},
                "commits": [],
                "statistics": {}
            }

        commits_for_ai = self._format_commits_for_ai(db_commits)
        ai_results = self.multifusion_v2_service.predict_commit_type_batch(commits_for_ai)
        # Pass commits_for_ai to _combine_results to merge numeric features
        commits_with_analysis = self._combine_results(db_commits, ai_results, commits_for_ai)

        # Basic statistics
        commit_type_stats = defaultdict(int)
        for c in commits_with_analysis:
            commit_type_stats[c['analysis']['type']] += 1

        return {
            "summary": {"total_commits": total_commits, "page_commits": len(commits_with_analysis)},
            "commits": commits_with_analysis,
            "statistics": {"commit_types": dict(commit_type_stats)}
        }