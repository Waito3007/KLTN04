# backend/services/multifusion_v2_service.py
"""
MultiFusion V2 AI Service - Advanced commit analysis with BERT + structured features
Supports commit type classification, member analysis, and productivity insights
"""

import torch
import torch.nn as nn
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MultiFusionV2Model(nn.Module):
    """MultiFusion V2 Model Architecture"""
    def __init__(self, bert_model_name, num_numerical_features, num_language_features, num_classes):
        super(MultiFusionV2Model, self).__init__()
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

class MultiFusionV2Service:
    """Service for MultiFusion V2 model operations"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder_lang = None
        self.label_encoder_type = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = 128
        self.model_path = Path("c:/SAN/KLTN/KLTN04/backend/ai/models/multifusion/multifusionV2.pth")
        self.metadata_path = Path("c:/SAN/KLTN/KLTN04/backend/ai/models/multifusion/metadata_v2.json")
        
        # Load model and metadata
        self._load_model()
    
    def _load_model(self):
        """Load the trained MultiFusion V2 model and metadata"""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model file not found: {self.model_path}")
                return False
            
            # Load metadata (encoders, scaler info)
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Recreate label encoders
                self.label_encoder_lang = LabelEncoder()
                self.label_encoder_lang.classes_ = np.array(metadata['language_classes'])
                
                self.label_encoder_type = LabelEncoder()
                self.label_encoder_type.classes_ = np.array(metadata['commit_type_classes'])
                
                # Recreate scaler
                self.scaler = StandardScaler()
                self.scaler.mean_ = np.array(metadata['scaler_mean'])
                self.scaler.scale_ = np.array(metadata['scaler_scale'])
                
                num_numerical_features = metadata['num_numerical_features']
                num_language_features = metadata['num_language_features']
                num_classes = metadata['num_classes']
                
            else:
                # Fallback default values
                logger.warning("Metadata file not found, using default values")
                self.label_encoder_lang = LabelEncoder()
                self.label_encoder_lang.classes_ = np.array(['python', 'javascript', 'java', 'cpp', 'unknown_language'])
                
                self.label_encoder_type = LabelEncoder()
                self.label_encoder_type.classes_ = np.array(['feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore', 'build', 'ci', 'perf', 'other_type'])
                
                self.scaler = StandardScaler()
                self.scaler.mean_ = np.array([10.0, 5.0, 2.0, 15.0, 2.0])
                self.scaler.scale_ = np.array([50.0, 25.0, 5.0, 75.0, 10.0])
                
                num_numerical_features = 5
                num_language_features = len(self.label_encoder_lang.classes_)
                num_classes = len(self.label_encoder_type.classes_)
            
            # Initialize tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
            # Initialize and load model
            self.model = MultiFusionV2Model(
                'distilbert-base-uncased',
                num_numerical_features,
                num_language_features,
                num_classes
            )
            
            # Load model weights
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("MultiFusion V2 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading MultiFusion V2 model: {e}")
            return False
    
    def is_model_available(self) -> bool:
        """Check if model is loaded and available"""
        return self.model is not None and self.tokenizer is not None
    
    def predict_commit_type(self, commit_message: str, lines_added: int = 0, 
                          lines_removed: int = 0, files_count: int = 1, 
                          detected_language: str = "unknown_language") -> Dict[str, Any]:
        """
        Predict commit type using MultiFusion V2 model
        
        Args:
            commit_message: The commit message text
            lines_added: Number of lines added
            lines_removed: Number of lines removed  
            files_count: Number of files changed
            detected_language: Programming language detected
            
        Returns:
            Dict with prediction results
        """
        if not self.is_model_available():
            return {"error": "Model not available", "commit_type": "unknown"}
        
        try:
            # Prepare text features
            encoding = self.tokenizer.encode_plus(
                commit_message,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            
            # Prepare numerical features
            total_changes = lines_added + lines_removed
            if lines_removed == 0:
                ratio_added_removed = lines_added
            else:
                ratio_added_removed = lines_added / lines_removed
            
            numerical_features = np.array([[lines_added, lines_removed, files_count, total_changes, ratio_added_removed]])
            numerical_features_scaled = torch.tensor(
                self.scaler.transform(numerical_features).flatten(), 
                dtype=torch.float32
            ).unsqueeze(0)
            
            # Prepare language features
            if detected_language in self.label_encoder_lang.classes_:
                lang_encoded = self.label_encoder_lang.transform([detected_language])[0]
            else:
                # Handle unknown language
                if "unknown_language" in self.label_encoder_lang.classes_:
                    lang_encoded = self.label_encoder_lang.transform(["unknown_language"])[0]
                else:
                    lang_encoded = 0  # First class as fallback
            
            lang_one_hot = torch.zeros(len(self.label_encoder_lang.classes_))
            lang_one_hot[lang_encoded] = 1
            lang_one_hot = lang_one_hot.unsqueeze(0)
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            numerical_features_scaled = numerical_features_scaled.to(self.device)
            lang_one_hot = lang_one_hot.to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, numerical_features_scaled, lang_one_hot)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_id = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_id].item()
            
            predicted_type = self.label_encoder_type.classes_[predicted_class_id]
            
            # Get all class probabilities
            all_probabilities = {}
            for i, class_name in enumerate(self.label_encoder_type.classes_):
                all_probabilities[class_name] = probabilities[0][i].item()
            
            return {
                "commit_type": predicted_type,
                "confidence": confidence,
                "all_probabilities": all_probabilities,
                "input_features": {
                    "lines_added": lines_added,
                    "lines_removed": lines_removed,
                    "files_count": files_count,
                    "detected_language": detected_language,
                    "total_changes": total_changes,
                    "ratio_added_removed": ratio_added_removed
                }
            }
            
        except Exception as e:
            logger.error(f"Error in MultiFusion V2 prediction: {e}")
            return {"error": str(e), "commit_type": "unknown"}
    
    def analyze_member_commits(self, commits_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze multiple commits for a member
        
        Args:
            commits_data: List of commit data dictionaries
            
        Returns:
            Analysis results with statistics and insights
        """
        if not self.is_model_available():
            return {"error": "Model not available"}
        
        try:
            results = []
            commit_type_counts = {}
            total_changes = 0
            total_files = 0
            languages_used = set()
            
            for commit in commits_data:
                # Extract commit features
                message = commit.get('message', '')
                lines_added = commit.get('lines_added', 0)
                lines_removed = commit.get('lines_removed', 0)
                files_count = commit.get('files_count', 1)
                detected_language = commit.get('detected_language', 'unknown_language')
                
                # Predict commit type
                prediction = self.predict_commit_type(
                    message, lines_added, lines_removed, files_count, detected_language
                )
                
                commit_type = prediction.get('commit_type', 'unknown')
                confidence = prediction.get('confidence', 0.0)
                
                # Accumulate statistics
                commit_type_counts[commit_type] = commit_type_counts.get(commit_type, 0) + 1
                total_changes += lines_added + lines_removed
                total_files += files_count
                if detected_language != 'unknown_language':
                    languages_used.add(detected_language)
                
                results.append({
                    'commit_id': commit.get('id', ''),
                    'message': message[:100] + '...' if len(message) > 100 else message,
                    'predicted_type': commit_type,
                    'confidence': round(confidence, 3),
                    'lines_added': lines_added,
                    'lines_removed': lines_removed,
                    'files_count': files_count,
                    'detected_language': detected_language,
                    'date': commit.get('date', '')
                })
            
            # Calculate insights
            total_commits = len(commits_data)
            avg_changes_per_commit = total_changes / total_commits if total_commits > 0 else 0
            avg_files_per_commit = total_files / total_commits if total_commits > 0 else 0
            
            # Find dominant commit type
            dominant_type = max(commit_type_counts.items(), key=lambda x: x[1]) if commit_type_counts else ('unknown', 0)
            
            return {
                "total_commits": total_commits,
                "commit_type_distribution": commit_type_counts,
                "dominant_commit_type": {
                    "type": dominant_type[0],
                    "count": dominant_type[1],
                    "percentage": round((dominant_type[1] / total_commits) * 100, 2) if total_commits > 0 else 0
                },
                "productivity_metrics": {
                    "total_changes": total_changes,
                    "total_files_modified": total_files,
                    "avg_changes_per_commit": round(avg_changes_per_commit, 2),
                    "avg_files_per_commit": round(avg_files_per_commit, 2)
                },
                "languages_used": list(languages_used),
                "commits": results
            }
            
        except Exception as e:
            logger.error(f"Error analyzing member commits: {e}")
            return {"error": str(e)}
    
    def batch_analyze_commits(self, commit_messages: List[str]) -> List[Dict[str, Any]]:
        """
        Batch analyze multiple commit messages (simple text-only analysis)
        
        Args:
            commit_messages: List of commit message strings
            
        Returns:
            List of prediction results
        """
        results = []
        for message in commit_messages:
            prediction = self.predict_commit_type(message)
            results.append({
                "message": message[:100] + '...' if len(message) > 100 else message,
                "predicted_type": prediction.get("commit_type", "unknown"),
                "confidence": prediction.get("confidence", 0.0)
            })
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": "MultiFusion V2",
            "architecture": "BERT + MLP Fusion",
            "version": "2.0",
            "is_available": self.is_model_available(),
            "device": str(self.device),
            "supported_languages": list(self.label_encoder_lang.classes_) if self.label_encoder_lang else [],
            "supported_commit_types": list(self.label_encoder_type.classes_) if self.label_encoder_type else [],
            "features": [
                "Commit message semantic analysis (BERT)",
                "Code metrics integration (lines, files)",
                "Programming language detection",
                "Multi-modal fusion",
                "High-accuracy commit type classification"
            ]
        }
